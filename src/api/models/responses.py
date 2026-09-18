"""Translate OpenAI Responses API traffic onto the Bedrock Converse API.

The gateway already speaks Chat Completions, but the Responses API differs on both ends:
a request carries a polymorphic ``input`` list instead of ``messages``, and a response is a
list of typed output items delivered over SSE as named events rather than a stream of
message deltas.

Requests are converted into the internal :class:`ChatRequest` so every bit of existing
request-side handling (model validation, inference profiles, prompt caching, reasoning
budgets) is reused. Responses are built from the raw Bedrock Converse output, because the
Chat Completions rendering flattens reasoning into ``<think>`` tags inside the text and the
Responses API needs it kept as a separate item.

Ref: https://platform.openai.com/docs/api-reference/responses
"""

import json
import logging
import time
import uuid
from typing import AsyncIterable

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.models.bedrock import ENCODER, BedrockModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    DeveloperMessage,
    Function,
    ImageContent,
    ImageUrl,
    ResponseFunction,
    ResponsesFunctionCall,
    ResponsesInputTokensDetails,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesOutputTokensDetails,
    ResponsesReasoningItem,
    ResponsesReasoningSummary,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from api.setting import DEBUG, DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)

# "minimal" has no Bedrock equivalent; treat it as the smallest budget we can express.
REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# Input item types that cannot be replayed to Bedrock.
# A reasoning block is only accepted back with the signature Bedrock issued it with, which
# is not part of the Responses wire format, and an item reference needs the server-side
# storage this gateway does not have.
IGNORED_INPUT_ITEM_TYPES = frozenset({"reasoning", "item_reference"})

TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text", "summary_text"})

# Output item id prefixes, mirroring the ones OpenAI uses per item type.
ITEM_ID_PREFIX = {"text": "msg", "reasoning": "rs", "tool": "fc"}


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _stringify(value) -> str:
    """Flatten a tool output into the single text block Bedrock's toolResult expects."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)
    return json.dumps(value)


def _content_parts(content) -> list[TextContent | ImageContent]:
    """Convert the content of an input message item into chat content parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContent(text=content)] if content else []

    parts: list[TextContent | ImageContent] = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append(TextContent(text=part))
            continue
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type in TEXT_PART_TYPES:
            text = part.get("text") or ""
            if text:
                parts.append(TextContent(text=text))
        elif part_type == "refusal":
            refusal = part.get("refusal") or ""
            if refusal:
                parts.append(TextContent(text=refusal))
        elif part_type == "input_image":
            url = part.get("image_url")
            if isinstance(url, dict):
                # Not part of the Responses schema, but tolerated: some clients reuse the
                # nested Chat Completions shape.
                url = url.get("url")
            if url:
                parts.append(ImageContent(image_url=ImageUrl(url=url, detail=part.get("detail") or "auto")))
        else:
            logger.warning("Ignoring unsupported Responses content part of type %s", part_type)
    return parts


def _text_of(parts: list[TextContent | ImageContent]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextContent))


def _convert_message_item(item: dict) -> list:
    role = item.get("role") or "user"
    parts = _content_parts(item.get("content"))

    if role == "system":
        text = _text_of(parts)
        return [SystemMessage(content=text)] if text else []
    if role == "developer":
        text = _text_of(parts)
        return [DeveloperMessage(content=text)] if text else []
    if role == "assistant":
        text = _text_of(parts)
        return [AssistantMessage(content=text)] if text else []
    return [UserMessage(content=parts)] if parts else []


def _convert_input_item(item: dict) -> list:
    """Convert one Responses input item into zero or more chat messages."""
    item_type = item.get("type")

    if item_type == "function_call":
        # An empty argument string is valid on the wire but not valid JSON, and the chat
        # layer json.loads() it on the way to Bedrock.
        arguments = item.get("arguments") or "{}"
        return [
            AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id=item.get("call_id") or item.get("id") or "",
                        type="function",
                        function=ResponseFunction(name=item.get("name"), arguments=arguments),
                    )
                ]
            )
        ]
    if item_type == "function_call_output":
        return [
            ToolMessage(
                tool_call_id=item.get("call_id") or "",
                content=_stringify(item.get("output")),
            )
        ]
    if item_type in IGNORED_INPUT_ITEM_TYPES:
        return []
    if item_type in (None, "message") or "role" in item:
        return _convert_message_item(item)

    logger.warning("Ignoring unsupported Responses input item of type %s", item_type)
    return []


def _convert_input(value: str | list[dict]) -> list:
    if isinstance(value, str):
        return [UserMessage(content=value)] if value else []

    messages = []
    for item in value:
        if isinstance(item, dict):
            messages.extend(_convert_input_item(item))
    return messages


def _convert_tools(tools: list[dict] | None) -> list[Tool] | None:
    if not tools:
        return None

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            # Hosted tools (web_search, file_search, computer_use, ...) run inside OpenAI
            # and have no Bedrock counterpart.
            logger.warning("Ignoring unsupported Responses tool of type %s", tool.get("type"))
            continue
        # Responses declares function tools flat; accept the nested Chat Completions shape too.
        spec = tool["function"] if isinstance(tool.get("function"), dict) else tool
        name = spec.get("name")
        if not name:
            continue
        converted.append(
            Tool(
                function=Function(
                    name=name,
                    description=spec.get("description"),
                    parameters=spec.get("parameters") or {"type": "object", "properties": {}},
                )
            )
        )
    return converted or None


def _convert_tool_choice(tool_choice: str | dict | None) -> str | dict:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            # Bedrock's toolChoice has no "none"; the closest behaviour is to let the model decide.
            logger.warning('tool_choice "none" is not supported by Bedrock, falling back to "auto"')
            return "auto"
        return tool_choice
    if tool_choice.get("type") == "function" and tool_choice.get("name"):
        return {"function": {"name": tool_choice["name"]}}
    if "function" in tool_choice:
        return tool_choice
    logger.warning("Ignoring unsupported tool_choice %s", tool_choice)
    return "auto"


async def _aiter(stream) -> AsyncIterable[dict]:
    """Iterate the blocking botocore event stream without blocking the event loop."""
    iterator = iter(stream)
    sentinel = object()
    while True:
        chunk = await run_in_threadpool(next, iterator, sentinel)
        if chunk is sentinel:
            return
        yield chunk


class BedrockResponsesModel:
    """Serves the Responses API on top of :class:`BedrockModel`."""

    def __init__(self, chat_model: BedrockModel | None = None):
        self.chat_model = chat_model or BedrockModel()

    def build_chat_request(self, request: ResponsesRequest) -> ChatRequest:
        """Convert and validate a Responses request. Raises HTTPException on bad input."""
        if DEBUG:
            logger.info("Raw Responses request: " + request.model_dump_json())

        messages = []
        if request.instructions:
            messages.append(SystemMessage(content=request.instructions))
        messages.extend(_convert_input(request.input))

        # instructions alone is not a conversation: Bedrock needs at least one message.
        if not any(message.role not in ("system", "developer") for message in messages):
            raise HTTPException(
                status_code=400,
                detail="input must contain at least one user, assistant or tool item",
            )

        reasoning_effort = None
        if request.reasoning and request.reasoning.effort:
            effort = request.reasoning.effort.lower()
            reasoning_effort = REASONING_EFFORT_MAP.get(effort)
            if reasoning_effort is None:
                logger.info("Ignoring unsupported reasoning effort %s", effort)

        max_tokens = request.max_output_tokens
        if reasoning_effort and max_tokens is None:
            # Claude rejects an enabled reasoning_config without maxTokens.
            max_tokens = DEFAULT_MAX_TOKENS

        chat_request = ChatRequest(
            messages=messages,
            model=request.model,
            stream=bool(request.stream),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tools=_convert_tools(request.tools),
            tool_choice=_convert_tool_choice(request.tool_choice),
            extra_body=request.extra_body,
        )
        self.chat_model.validate(chat_request)
        return chat_request

    def _base_response(self, request: ResponsesRequest, response_id: str, created_at: int) -> ResponsesResponse:
        """A response shell echoing the request, shared by the final object and the events."""
        return ResponsesResponse(
            id=response_id,
            created_at=created_at,
            model=request.model,
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            parallel_tool_calls=bool(request.parallel_tool_calls),
            previous_response_id=request.previous_response_id,
            reasoning=request.reasoning,
            store=False,
            temperature=request.temperature,
            text=request.text,
            tool_choice=request.tool_choice if request.tool_choice is not None else "auto",
            tools=request.tools or [],
            top_p=request.top_p,
            truncation=request.truncation or "disabled",
            metadata=request.metadata or {},
        )

    @staticmethod
    def _usage(usage: dict, reasoning_tokens: int = 0) -> ResponsesUsage:
        """Map Bedrock token counts onto the Responses usage object.

        Bedrock's totalTokens covers input, cache reads/writes and output, so the input
        side is derived from the total the same way the Chat Completions path does it.
        """
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", 0)
        input_tokens = total_tokens - output_tokens
        return ResponsesUsage(
            input_tokens=input_tokens,
            input_tokens_details=ResponsesInputTokensDetails(cached_tokens=usage.get("cacheReadInputTokens", 0)),
            output_tokens=output_tokens,
            output_tokens_details=ResponsesOutputTokensDetails(reasoning_tokens=reasoning_tokens),
            total_tokens=total_tokens if total_tokens > 0 else input_tokens + output_tokens,
        )

    @staticmethod
    def _apply_stop_reason(response: ResponsesResponse, stop_reason: str | None) -> None:
        if stop_reason == "max_tokens":
            response.status = "incomplete"
            response.incomplete_details = {"reason": "max_output_tokens"}
        elif stop_reason == "content_filtered":
            response.status = "incomplete"
            response.incomplete_details = {"reason": "content_filter"}
        else:
            response.status = "completed"

    async def respond(self, request: ResponsesRequest, chat_request: ChatRequest | None = None) -> ResponsesResponse:
        """Handle a non-streaming Responses request."""
        chat_request = chat_request or self.build_chat_request(request)
        raw = await self.chat_model.invoke(chat_request)

        response = self._base_response(request, generate_id("resp"), int(time.time()))

        reasoning_text = ""
        text = ""
        tool_calls = []
        for block in raw["output"]["message"].get("content", []):
            if "reasoningContent" in block:
                reasoning_text += block["reasoningContent"].get("reasoningText", {}).get("text", "")
            elif "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                tool_calls.append(block["toolUse"])
            else:
                logger.warning("Unknown tag in message content " + ",".join(block.keys()))

        if reasoning_text:
            response.output.append(
                ResponsesReasoningItem(
                    id=generate_id("rs"),
                    summary=[ResponsesReasoningSummary(text=reasoning_text)],
                    status="completed",
                )
            )
        if text:
            response.output.append(
                ResponsesOutputMessage(
                    id=generate_id("msg"),
                    content=[ResponsesOutputText(text=text)],
                    status="completed",
                )
            )
        for tool_use in tool_calls:
            response.output.append(
                ResponsesFunctionCall(
                    id=generate_id("fc"),
                    call_id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    arguments=json.dumps(tool_use.get("input") or {}),
                    status="completed",
                )
            )

        # Bedrock does not report reasoning tokens separately, so estimate them the same
        # way the Chat Completions path does.
        reasoning_tokens = len(ENCODER.encode(reasoning_text)) if reasoning_text else 0
        response.usage = self._usage(raw.get("usage", {}), reasoning_tokens)
        self._apply_stop_reason(response, raw.get("stopReason"))

        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response

    async def respond_stream(
        self, request: ResponsesRequest, chat_request: ChatRequest | None = None
    ) -> AsyncIterable[bytes]:
        """Handle a streaming Responses request, emitting the named SSE events.

        Once the stream has started the status line is already sent, so a failure here can
        only be reported as a response.failed event rather than an HTTP error.
        """
        response_id = generate_id("resp")
        session = _StreamSession(self._base_response(request, response_id, int(time.time())))
        chat_request = chat_request or self.build_chat_request(request)

        # response.created has to come first: a client that sees any other event before it
        # has nothing to attach the rest of the stream to, and the OpenAI SDK errors out.
        yield session.event("response.created", {"response": session.snapshot("in_progress")})
        yield session.event("response.in_progress", {"response": session.snapshot("in_progress")})

        try:
            raw = await self.chat_model.invoke(chat_request, stream=True)

            async for chunk in _aiter(raw.get("stream")):
                if DEBUG:
                    logger.info("Bedrock response chunk: " + str(chunk))
                for event in session.handle(chunk):
                    yield event

            for event in session.finish():
                yield event
        except HTTPException as e:
            logger.error("Stream error for model %s: %s", request.model, str(e.detail))
            for event in session.fail(str(e.detail)):
                yield event
        except Exception as e:
            logger.error("Stream error for model %s: %s", request.model, str(e))
            for event in session.fail(str(e)):
                yield event


class _OpenBlock:
    """A Bedrock content block that is currently streaming, and the item it maps to."""

    def __init__(self, kind: str, output_index: int, item_id: str, call_id: str = "", name: str = ""):
        self.kind = kind  # "text" | "reasoning" | "tool"
        self.output_index = output_index
        self.item_id = item_id
        self.call_id = call_id
        self.name = name
        self.buffer = ""


class _StreamSession:
    """Turns a Bedrock Converse stream into Responses API events.

    Bedrock reports content blocks by index and streams reasoning, text and tool input as
    deltas on those blocks. Each block becomes one Responses output item, opened when its
    first delta arrives and closed on contentBlockStop.
    """

    def __init__(self, response: ResponsesResponse):
        self.response = response
        self.sequence_number = 0
        self.blocks: dict[int, _OpenBlock] = {}
        self.next_output_index = 0
        self.stop_reason: str | None = None
        self.usage: dict = {}
        self.reasoning_tokens = 0

    def event(self, event_type: str, payload: dict) -> bytes:
        self.sequence_number += 1
        data = {"type": event_type, "sequence_number": self.sequence_number, **payload}
        if DEBUG:
            logger.info("Proxy response event: " + event_type)
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

    def snapshot(self, status: str) -> dict:
        """The response object as it currently stands, for the lifecycle events."""
        return self.response.model_copy(update={"status": status}).model_dump()

    def handle(self, chunk: dict) -> list[bytes]:
        if "contentBlockStart" in chunk:
            start = chunk["contentBlockStart"]["start"]
            if "toolUse" in start:
                return self._open(
                    chunk["contentBlockStart"]["contentBlockIndex"],
                    "tool",
                    call_id=start["toolUse"]["toolUseId"],
                    name=start["toolUse"]["name"],
                )
            return []

        if "contentBlockDelta" in chunk:
            index = chunk["contentBlockDelta"]["contentBlockIndex"]
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                events = self._open(index, "text")
                return events + self._delta(index, delta["text"])
            if "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if "text" not in reasoning:
                    # The signature only matters for replaying a block back to Bedrock,
                    # which the Responses wire format cannot carry.
                    return []
                self.reasoning_tokens += len(ENCODER.encode(reasoning["text"]))
                events = self._open(index, "reasoning")
                return events + self._delta(index, reasoning["text"])
            if "toolUse" in delta:
                return self._delta(index, delta["toolUse"].get("input") or "")
            return []

        if "contentBlockStop" in chunk:
            return self._close(chunk["contentBlockStop"]["contentBlockIndex"])

        if "messageStop" in chunk:
            self.stop_reason = chunk["messageStop"].get("stopReason")
            return []

        if "metadata" in chunk:
            self.usage = chunk["metadata"].get("usage", {})
            return []

        return []

    def _open(self, index: int, kind: str, call_id: str = "", name: str = "") -> list[bytes]:
        if index in self.blocks:
            return []

        block = _OpenBlock(kind, self.next_output_index, generate_id(ITEM_ID_PREFIX[kind]), call_id, name)
        self.blocks[index] = block
        self.next_output_index += 1

        events = [
            self.event(
                "response.output_item.added",
                {"output_index": block.output_index, "item": self._item(block, "in_progress").model_dump()},
            )
        ]
        if kind == "text":
            events.append(
                self.event(
                    "response.content_part.added",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "content_index": 0,
                        "part": ResponsesOutputText(text="").model_dump(),
                    },
                )
            )
        elif kind == "reasoning":
            events.append(
                self.event(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "summary_index": 0,
                        "part": ResponsesReasoningSummary(text="").model_dump(),
                    },
                )
            )
        return events

    def _delta(self, index: int, text: str) -> list[bytes]:
        block = self.blocks.get(index)
        if block is None or not text:
            return []
        block.buffer += text

        if block.kind == "text":
            return [
                self.event(
                    "response.output_text.delta",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "content_index": 0,
                        "delta": text,
                    },
                )
            ]
        if block.kind == "reasoning":
            return [
                self.event(
                    "response.reasoning_summary_text.delta",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "summary_index": 0,
                        "delta": text,
                    },
                )
            ]
        return [
            self.event(
                "response.function_call_arguments.delta",
                {"item_id": block.item_id, "output_index": block.output_index, "delta": text},
            )
        ]

    def _close(self, index: int) -> list[bytes]:
        block = self.blocks.pop(index, None)
        if block is None:
            return []

        item = self._item(block, "completed")
        events = []
        if block.kind == "text":
            events.append(
                self.event(
                    "response.output_text.done",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "content_index": 0,
                        "text": block.buffer,
                    },
                )
            )
            events.append(
                self.event(
                    "response.content_part.done",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "content_index": 0,
                        "part": ResponsesOutputText(text=block.buffer).model_dump(),
                    },
                )
            )
        elif block.kind == "reasoning":
            events.append(
                self.event(
                    "response.reasoning_summary_text.done",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "summary_index": 0,
                        "text": block.buffer,
                    },
                )
            )
            events.append(
                self.event(
                    "response.reasoning_summary_part.done",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "summary_index": 0,
                        "part": ResponsesReasoningSummary(text=block.buffer).model_dump(),
                    },
                )
            )
        else:
            events.append(
                self.event(
                    "response.function_call_arguments.done",
                    {
                        "item_id": block.item_id,
                        "output_index": block.output_index,
                        "arguments": item.arguments,
                    },
                )
            )

        self.response.output.append(item)
        events.append(
            self.event(
                "response.output_item.done",
                {"output_index": block.output_index, "item": item.model_dump()},
            )
        )
        return events

    def _item(self, block: _OpenBlock, status: str):
        if block.kind == "text":
            content = [ResponsesOutputText(text=block.buffer)] if status == "completed" else []
            return ResponsesOutputMessage(id=block.item_id, content=content, status=status)
        if block.kind == "reasoning":
            summary = [ResponsesReasoningSummary(text=block.buffer)] if status == "completed" else []
            return ResponsesReasoningItem(id=block.item_id, summary=summary, status=status)
        return ResponsesFunctionCall(
            id=block.item_id,
            call_id=block.call_id,
            name=block.name,
            # An empty argument string is valid on the wire but clients json.loads() it.
            arguments=block.buffer or "{}",
            status=status,
        )

    def finish(self) -> list[bytes]:
        events = []
        # Bedrock always closes its blocks, but never leave an item half-open if it does not.
        for index in sorted(self.blocks):
            events.extend(self._close(index))

        self.response.usage = BedrockResponsesModel._usage(self.usage, self.reasoning_tokens)
        BedrockResponsesModel._apply_stop_reason(self.response, self.stop_reason)

        event_type = "response.completed" if self.response.status == "completed" else "response.incomplete"
        events.append(self.event(event_type, {"response": self.response.model_dump()}))
        return events

    def fail(self, message: str) -> list[bytes]:
        error = {"code": "server_error", "message": message}
        self.response.status = "failed"
        self.response.error = error
        return [
            self.event("error", error),
            self.event("response.failed", {"response": self.response.model_dump()}),
        ]
