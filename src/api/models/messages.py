"""Translate Anthropic Messages API traffic onto the Bedrock Converse API.

This is what lets Claude Code (and any other Anthropic SDK client) run against the gateway.

Unlike the Responses API, requests are converted straight into Converse arguments instead of
going through the internal :class:`ChatRequest`. The Chat Completions shape cannot carry what
an agentic Anthropic client depends on: thinking blocks with their signatures (Claude rejects
a tool-use turn replayed without them), ``cache_control`` breakpoints, images inside tool
results, and Claude-only request fields such as ``thinking`` and ``output_config``.

Ref: https://docs.anthropic.com/en/api/messages
"""

import asyncio
import base64
import json
import logging
import re
import uuid
from typing import AsyncIterable

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.image_url import parse_image_url
from api.models import bedrock
from api.models.bedrock import (
    ENCODER,
    TEMPERATURE_TOPP_CONFLICT_MODELS,
    TEMPERATURE_UNSUPPORTED_MODELS,
    BedrockModel,
    bedrock_runtime,
)
from api.schema import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from api.setting import ANTHROPIC_BETA_ALLOWLIST, DEBUG

logger = logging.getLogger(__name__)

STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "guardrail_intervened": "refusal",
    "content_filtered": "refusal",
    "model_context_window_exceeded": "model_context_window_exceeded",
}

# Request fields that only Claude understands. They are passed through to Claude as
# additionalModelRequestFields and dropped for every other model.
CLAUDE_ONLY_FIELDS = ("thinking", "output_config", "context_management", "top_k")

# Converse only accepts these letters in a document name.
DOCUMENT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9\s\-\(\)\[\]]")

# How long a stream may go quiet before a ping is sent. Claude can think for a long time
# without streaming anything, which trips idle timeouts on load balancers and clients.
PING_INTERVAL_SECONDS = 15

# Bedrock ids an Anthropic model name may map to: an optional date and version suffix.
_VERSION_SUFFIX = r"(-\d{8})?(-v\d+(:\d+)?)?"


def generate_message_id() -> str:
    return f"msg_bdrk_{uuid.uuid4().hex[:24]}"


def resolve_model(model: str) -> str:
    """Map an Anthropic model name such as claude-sonnet-4-5 onto a Bedrock model id.

    Claude Code sends first-party names by default. A name that is already a Bedrock id
    (or inference profile) is used as is. Otherwise the matching Claude model is looked up,
    preferring a global inference profile, then a regional one, then the bare model id.
    Anything unresolved is returned unchanged for validation to reject.
    """
    model_list = bedrock.bedrock_model_list
    if model in model_list or not model.startswith("claude-"):
        return model

    pattern = re.compile(rf"^(?:([a-z]+)\.)?anthropic\.{re.escape(model)}{_VERSION_SUFFIX}$")
    candidates = []
    for model_id in model_list:
        match = pattern.match(model_id)
        if match:
            geo = match.group(1)
            rank = 0 if geo == "global" else 1 if geo else 2
            candidates.append((rank, model_id))
    if not candidates:
        return model
    # Newest version first within a rank (so v2 wins over v1); the second sort is stable.
    candidates.sort(key=lambda c: c[1], reverse=True)
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _decode_base64(data: str, what: str) -> bytes:
    try:
        return base64.b64decode(data)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"Invalid base64 data in {what}")


def _cache_point(cache_control) -> dict | None:
    if not isinstance(cache_control, dict):
        return None
    point = {"type": "default"}
    if cache_control.get("ttl") in ("5m", "1h"):
        point["ttl"] = cache_control["ttl"]
    return {"cachePoint": point}


async def _aiter_with_pings(stream) -> AsyncIterable[dict | None]:
    """Iterate the blocking botocore event stream, yielding None whenever it goes quiet."""
    iterator = iter(stream)
    sentinel = object()
    pending = asyncio.ensure_future(run_in_threadpool(next, iterator, sentinel))
    while True:
        done, _ = await asyncio.wait({pending}, timeout=PING_INTERVAL_SECONDS)
        if not done:
            yield None
            continue
        chunk = pending.result()
        if chunk is sentinel:
            return
        yield chunk
        pending = asyncio.ensure_future(run_in_threadpool(next, iterator, sentinel))


class _RequestConverter:
    """Converts one Anthropic request into Converse arguments."""

    def __init__(self, chat_model: BedrockModel, model_id: str):
        self.chat_model = chat_model
        self.model_id = model_id
        resolved = chat_model._resolve_to_foundation_model(model_id).lower()
        self.resolved_model = resolved
        self.is_claude = "anthropic.claude" in resolved
        self.cache_supported = chat_model._supports_prompt_caching(model_id)
        self.document_count = 0
        self.tool_names_used: list[str] = []

    def cache_point(self, cache_control) -> list[dict]:
        if not self.cache_supported:
            return []
        point = _cache_point(cache_control)
        return [point] if point else []

    def system(self, system: str | list[dict] | None) -> list[dict]:
        if system is None:
            return []
        if isinstance(system, str):
            return [{"text": system}] if system.strip() else []
        blocks = []
        for block in system:
            text = block.get("text") or ""
            if text.strip():
                blocks.append({"text": text})
            blocks.extend(self.cache_point(block.get("cache_control")))
        return blocks

    def image(self, source: dict) -> dict:
        if not self.chat_model.is_supported_modality(self.model_id, modality="IMAGE"):
            raise HTTPException(
                status_code=400, detail=f"Multimodal message is currently not supported by {self.model_id}"
            )
        if source.get("type") == "base64":
            data = _decode_base64(source.get("data") or "", "image")
            content_type = source.get("media_type") or "image/png"
        elif source.get("type") == "url":
            # Fetched through the same checks as the Chat Completions image urls.
            data, content_type = parse_image_url(source.get("url") or "")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported image source type {source.get('type')}")
        return {"image": {"format": content_type.split("/")[-1], "source": {"bytes": data}}}

    def document(self, block: dict) -> list[dict]:
        source = block.get("source") or {}
        source_type = source.get("type")
        if source_type == "content":
            # A document made of content blocks is just its text as far as Bedrock goes.
            return [
                {"text": part["text"]}
                for part in source.get("content") or []
                if isinstance(part, dict) and (part.get("text") or "").strip()
            ]
        if source_type == "base64" and source.get("media_type") == "application/pdf":
            doc_format, data = "pdf", _decode_base64(source.get("data") or "", "document")
        elif source_type == "text":
            doc_format, data = "txt", (source.get("data") or "").encode("utf-8")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document source {source_type} {source.get('media_type') or ''}".strip(),
            )

        # Converse wants a name per document and is picky about its characters.
        self.document_count += 1
        name = DOCUMENT_NAME_PATTERN.sub(" ", block.get("title") or "").strip()
        name = f"{name[:100] or 'document'} {self.document_count}"
        return [{"document": {"format": doc_format, "name": name, "source": {"bytes": data}}}]

    def tool_result(self, block: dict) -> list[dict]:
        content = block.get("content")
        parts = []
        cache_points = self.cache_point(block.get("cache_control"))
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    if part.get("text"):
                        parts.append({"text": part["text"]})
                elif part_type == "image":
                    parts.append(self.image(part.get("source") or {}))
                elif part_type == "document":
                    parts.extend(self.document(part))
                else:
                    logger.warning("Ignoring unsupported tool_result content of type %s", part_type)
                # Bedrock has no cache point inside a tool result; put it right after one.
                if not cache_points:
                    cache_points = self.cache_point(part.get("cache_control"))

        # Only Claude takes images and documents inside a tool result; other vision models
        # on Bedrock (GPT, Qwen-VL, ...) reject them there but accept them right after it,
        # in the same user turn. That is where screenshots and images Claude Code reads land.
        attachments = []
        if not self.is_claude:
            attachments = [part for part in parts if "image" in part or "document" in part]
            parts = [part for part in parts if "text" in part]
            if attachments:
                parts.append({"text": f"[{len(attachments)} attachment(s) follow this tool result]"})

        result = {"toolUseId": block.get("tool_use_id") or "", "content": parts}
        if block.get("is_error"):
            result["status"] = "error"
        return [{"toolResult": result}] + attachments + cache_points

    def content_block(self, block: dict) -> list[dict]:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            # Converse rejects blank text blocks, the Messages API tolerates them.
            converted = [{"text": text}] if text.strip() else []
        elif block_type == "image":
            converted = [self.image(block.get("source") or {})]
        elif block_type == "document":
            converted = self.document(block)
        elif block_type == "tool_use":
            self.tool_names_used.append(block.get("name") or "")
            tool_input = block.get("input")
            converted = [
                {
                    "toolUse": {
                        "toolUseId": block.get("id") or "",
                        "name": block.get("name") or "",
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                }
            ]
        elif block_type == "tool_result":
            return self.tool_result(block)
        elif block_type == "thinking":
            signature = block.get("signature") or ""
            # Without a signature Claude refuses the block, and models that do not sign
            # their reasoning (DeepSeek, Qwen, ...) do not want it back anyway.
            if not signature:
                return []
            converted = [
                {"reasoningContent": {"reasoningText": {"text": block.get("thinking") or "", "signature": signature}}}
            ]
        elif block_type == "redacted_thinking":
            converted = [
                {"reasoningContent": {"redactedContent": _decode_base64(block.get("data") or "", "redacted_thinking")}}
            ]
        else:
            # server_tool_use, web_search_tool_result, ... only exist on the first-party API.
            logger.warning("Ignoring unsupported content block of type %s", block_type)
            return []
        return converted + self.cache_point(block.get("cache_control"))

    def messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        # Text of mid-conversation system messages that is waiting for a user turn.
        pending_system: list[dict] = []
        for message in messages:
            role = message.get("role")
            if role not in ("user", "assistant", "system"):
                raise HTTPException(status_code=400, detail=f"Unexpected message role {role}")
            content = message.get("content")
            if isinstance(content, str):
                blocks = [{"text": content}] if content.strip() else []
            else:
                blocks = []
                for block in content or []:
                    if isinstance(block, dict):
                        blocks.extend(self.content_block(block))
            if not blocks:
                continue

            if role == "system":
                # Claude Code sends these with the mid-conversation-system beta; Converse
                # only has user and assistant turns. The text joins the user turn right
                # before it, or else the end of the next one, since tool results have to
                # lead a user turn.
                if converted and converted[-1]["role"] == "user":
                    converted[-1]["content"].extend(blocks)
                else:
                    pending_system.extend(blocks)
                continue
            if role == "user" and pending_system:
                blocks, pending_system = blocks + pending_system, []

            # The Messages API merges consecutive turns of one role, Converse rejects them.
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})
        if pending_system:
            converted.append({"role": "user", "content": pending_system})
        if not converted:
            raise HTTPException(status_code=400, detail="messages must contain at least one non-empty message")
        return converted

    def tool_config(self, tools: list[dict] | None, tool_choice: dict | None) -> dict | None:
        converted = []
        for tool in tools or []:
            if tool.get("type") not in (None, "custom"):
                # Server tools (web_search, code_execution, ...) run on Anthropic's side and
                # have no Bedrock counterpart.
                logger.warning("Ignoring unsupported tool of type %s", tool.get("type"))
                continue
            spec = {
                "name": tool.get("name") or "",
                "inputSchema": {"json": tool.get("input_schema") or {"type": "object", "properties": {}}},
            }
            if tool.get("description"):
                spec["description"] = tool["description"]
            converted.append({"toolSpec": spec})
            converted.extend(self.cache_point(tool.get("cache_control")))

        if not any("toolSpec" in tool for tool in converted):
            if not self.tool_names_used:
                return None
            # Converse refuses toolUse/toolResult blocks without a toolConfig, which the
            # Messages API allows (e.g. a summarising request over a tool-using history).
            converted = [
                {
                    "toolSpec": {
                        "name": name,
                        "description": "Tool used earlier in this conversation.",
                        "inputSchema": {"json": {"type": "object", "properties": {}}},
                    }
                }
                for name in dict.fromkeys(self.tool_names_used)
            ]

        config = {"tools": converted}
        choice_type = (tool_choice or {}).get("type")
        if choice_type == "any":
            config["toolChoice"] = {"any": {}}
        elif choice_type == "tool" and tool_choice.get("name"):
            config["toolChoice"] = {"tool": {"name": tool_choice["name"]}}
        elif choice_type == "none":
            # Converse has no "none"; letting the model decide is the closest.
            logger.info('tool_choice "none" is not supported by Bedrock, using "auto"')
        return config

    def claude_fields(self, request: AnthropicCountTokensRequest, betas: list[str]) -> dict:
        fields = {}
        if not self.is_claude:
            ignored = [name for name in CLAUDE_ONLY_FIELDS if getattr(request, name, None) is not None]
            if ignored and DEBUG:
                logger.info("Ignoring %s for non-Claude model %s", ", ".join(ignored), self.model_id)
            return fields
        for name in CLAUDE_ONLY_FIELDS:
            value = getattr(request, name, None)
            if value is not None:
                fields[name] = value
        forwarded = [beta for beta in betas if beta in ANTHROPIC_BETA_ALLOWLIST]
        if forwarded:
            fields["anthropic_beta"] = forwarded
        return fields

    def inference_config(self, request: AnthropicMessagesRequest, thinking_enabled: bool) -> dict:
        config = {"maxTokens": request.max_tokens}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["topP"] = request.top_p
        if request.stop_sequences:
            config["stopSequences"] = request.stop_sequences

        # Same per-model quirks as the Chat Completions path.
        if any(model in self.resolved_model for model in TEMPERATURE_UNSUPPORTED_MODELS):
            config.pop("temperature", None)
        if "temperature" in config and "topP" in config:
            if any(model in self.resolved_model for model in TEMPERATURE_TOPP_CONFLICT_MODELS):
                config.pop("topP", None)
        if thinking_enabled:
            # Extended thinking does not take top_p below 0.95; dropping it is the safe choice.
            config.pop("topP", None)
        return config


def parse_betas(header: str | None) -> list[str]:
    return [beta.strip() for beta in (header or "").split(",") if beta.strip()]


class BedrockMessagesModel:
    """Serves the Anthropic Messages API on top of :class:`BedrockModel`."""

    def __init__(self, chat_model: BedrockModel | None = None):
        self.chat_model = chat_model or BedrockModel()

    def build_converse_args(self, request: AnthropicCountTokensRequest, betas: list[str] | None = None) -> dict:
        """Convert and validate a request into Converse arguments. Raises HTTPException on bad input."""
        if DEBUG:
            logger.info("Raw Messages request: " + request.model_dump_json())

        model_id = resolve_model(request.model)
        self.chat_model.validate_model(model_id)
        converter = _RequestConverter(self.chat_model, model_id)

        args = {"modelId": model_id, "messages": converter.messages(request.messages)}
        system = converter.system(request.system)
        if system:
            args["system"] = system
        tool_config = converter.tool_config(request.tools, request.tool_choice)
        if tool_config:
            args["toolConfig"] = tool_config

        fields = converter.claude_fields(request, betas or [])
        if fields:
            args["additionalModelRequestFields"] = fields
        if isinstance(request, AnthropicMessagesRequest):
            thinking_enabled = (fields.get("thinking") or {}).get("type") in ("enabled", "adaptive")
            args["inferenceConfig"] = converter.inference_config(request, thinking_enabled)

        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))
        return args

    async def invoke(self, args: dict, stream: bool = False):
        return await self.chat_model.converse(args, stream=stream)

    @staticmethod
    def usage(usage: dict) -> AnthropicUsage:
        # Converse reports inputTokens without the cached part, as the Messages API does.
        return AnthropicUsage(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cache_creation_input_tokens=usage.get("cacheWriteInputTokens", 0),
            cache_read_input_tokens=usage.get("cacheReadInputTokens", 0),
        )

    @staticmethod
    def stop(stop_reason: str | None, additional_fields: dict | None) -> tuple[str, str | None]:
        reason = STOP_REASON_MAP.get(stop_reason or "", "end_turn")
        stop_sequence = (additional_fields or {}).get("stop_sequence") if reason == "stop_sequence" else None
        return reason, stop_sequence

    @staticmethod
    def content(blocks: list[dict]) -> list[dict]:
        content = []
        for block in blocks:
            if "text" in block:
                content.append({"type": "text", "text": block["text"]})
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                if "redactedContent" in reasoning:
                    data = base64.b64encode(reasoning["redactedContent"]).decode("ascii")
                    content.append({"type": "redacted_thinking", "data": data})
                else:
                    text = reasoning.get("reasoningText") or {}
                    content.append(
                        {"type": "thinking", "thinking": text.get("text", ""), "signature": text.get("signature", "")}
                    )
            elif "toolUse" in block:
                tool = block["toolUse"]
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool["toolUseId"],
                        "name": tool["name"],
                        "input": tool.get("input") or {},
                    }
                )
            else:
                logger.warning("Unknown tag in message content " + ",".join(block.keys()))
        return content

    async def respond(self, request: AnthropicMessagesRequest, args: dict) -> AnthropicMessagesResponse:
        """Handle a non-streaming Messages request."""
        raw = await self.invoke(args)
        stop_reason, stop_sequence = self.stop(raw.get("stopReason"), raw.get("additionalModelResponseFields"))
        response = AnthropicMessagesResponse(
            id=generate_message_id(),
            model=request.model,
            content=self.content(raw["output"]["message"].get("content", [])),
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=self.usage(raw.get("usage", {})),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response

    async def respond_stream(self, request: AnthropicMessagesRequest, raw: dict) -> AsyncIterable[bytes]:
        """Turn an open ConverseStream response into Messages API SSE events.

        The stream is opened by the caller so that a failed call still gets a proper HTTP
        error; once events flow, a failure can only be reported as an error event.
        """
        session = _StreamSession(request.model)
        yield session.start()
        try:
            async for chunk in _aiter_with_pings(raw.get("stream")):
                if chunk is None:
                    yield session.event("ping", {})
                    continue
                if DEBUG:
                    logger.info("Bedrock response chunk: " + str(chunk))
                for event in session.handle(chunk):
                    yield event
            for event in session.finish():
                yield event
        except Exception as e:
            logger.error("Stream error for model %s: %s", request.model, str(e))
            yield session.error(e)

    async def count_tokens(self, request: AnthropicCountTokensRequest, betas: list[str] | None = None) -> int:
        args = self.build_converse_args(request, betas)
        converse_input = {key: args[key] for key in ("messages", "system", "toolConfig") if key in args}
        # CountTokens only takes a foundation model id, not an inference profile.
        model_id = self.chat_model._resolve_to_foundation_model(args["modelId"])
        try:
            response = await run_in_threadpool(
                bedrock_runtime.count_tokens, modelId=model_id, input={"converse": converse_input}
            )
            return response["inputTokens"]
        except Exception as e:
            # Not every model supports CountTokens; an estimate beats failing the client.
            if DEBUG:
                logger.info("CountTokens unavailable for %s, estimating: %s", model_id, str(e))
            return self._estimate_tokens(converse_input)

    @staticmethod
    def _estimate_tokens(converse_input: dict) -> int:
        texts = []

        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ("bytes", "redactedContent", "signature"):
                        continue
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, str):
                texts.append(value)

        collect(converse_input)
        return len(ENCODER.encode("\n".join(texts), disallowed_special=()))


class _StreamSession:
    """Turns a Bedrock ConverseStream into Messages API events.

    Bedrock opens text and reasoning blocks implicitly with their first delta and numbers
    blocks its own way, so every Bedrock block index is mapped onto a Messages content
    block index, opened on first sight and closed on contentBlockStop.
    """

    def __init__(self, model: str):
        self.model = model
        self.message_id = generate_message_id()
        self.blocks: dict[int, tuple[int, str]] = {}  # Bedrock index -> (index, type)
        self.next_index = 0
        self.stop_reason: str | None = None
        self.additional_fields: dict | None = None
        self.usage: dict = {}

    @staticmethod
    def event(event_type: str, payload: dict) -> bytes:
        data = {"type": event_type, **payload}
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

    def start(self) -> bytes:
        message = AnthropicMessagesResponse(id=self.message_id, model=self.model).model_dump()
        return self.event("message_start", {"message": message})

    def handle(self, chunk: dict) -> list[bytes]:
        if "contentBlockStart" in chunk:
            start = chunk["contentBlockStart"].get("start", {})
            if "toolUse" in start:
                block = {
                    "type": "tool_use",
                    "id": start["toolUse"]["toolUseId"],
                    "name": start["toolUse"]["name"],
                    "input": {},
                }
                return self._open(chunk["contentBlockStart"]["contentBlockIndex"], block)
            return []

        if "contentBlockDelta" in chunk:
            index = chunk["contentBlockDelta"]["contentBlockIndex"]
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                events = self._open(index, {"type": "text", "text": ""})
                return events + self._delta(index, {"type": "text_delta", "text": delta["text"]})
            if "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if "redactedContent" in reasoning:
                    data = base64.b64encode(reasoning["redactedContent"]).decode("ascii")
                    return self._open(index, {"type": "redacted_thinking", "data": data})
                events = self._open(index, {"type": "thinking", "thinking": "", "signature": ""})
                if reasoning.get("text"):
                    events += self._delta(index, {"type": "thinking_delta", "thinking": reasoning["text"]})
                if "signature" in reasoning:
                    events += self._delta(index, {"type": "signature_delta", "signature": reasoning["signature"]})
                return events
            if "toolUse" in delta:
                return self._delta(
                    index, {"type": "input_json_delta", "partial_json": delta["toolUse"].get("input") or ""}
                )
            return []

        if "contentBlockStop" in chunk:
            return self._close(chunk["contentBlockStop"]["contentBlockIndex"])

        if "messageStop" in chunk:
            self.stop_reason = chunk["messageStop"].get("stopReason")
            self.additional_fields = chunk["messageStop"].get("additionalModelResponseFields")
            return []

        if "metadata" in chunk:
            self.usage = chunk["metadata"].get("usage", {})
            return []

        return []

    def _open(self, bedrock_index: int, block: dict) -> list[bytes]:
        if bedrock_index in self.blocks:
            return []
        index = self.next_index
        self.next_index += 1
        self.blocks[bedrock_index] = (index, block["type"])
        return [self.event("content_block_start", {"index": index, "content_block": block})]

    def _delta(self, bedrock_index: int, delta: dict) -> list[bytes]:
        if bedrock_index not in self.blocks:
            return []
        index, _ = self.blocks[bedrock_index]
        return [self.event("content_block_delta", {"index": index, "delta": delta})]

    def _close(self, bedrock_index: int) -> list[bytes]:
        if bedrock_index not in self.blocks:
            return []
        index, _ = self.blocks.pop(bedrock_index)
        return [self.event("content_block_stop", {"index": index})]

    def finish(self) -> list[bytes]:
        events = []
        # Bedrock always closes its blocks, but never leave one half-open if it does not.
        for bedrock_index in sorted(self.blocks, key=lambda i: self.blocks[i][0]):
            events.extend(self._close(bedrock_index))

        stop_reason, stop_sequence = BedrockMessagesModel.stop(self.stop_reason, self.additional_fields)
        events.append(
            self.event(
                "message_delta",
                {
                    "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                    # Converse only reports usage at the end, so the input side is sent here
                    # too; the Anthropic SDK and Claude Code merge it into the message usage.
                    "usage": BedrockMessagesModel.usage(self.usage).model_dump(),
                },
            )
        )
        events.append(self.event("message_stop", {}))
        return events

    def error(self, exc: Exception) -> bytes:
        message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        error_type = "overloaded_error" if "throttl" in message.lower() else "api_error"
        return self.event("error", {"error": {"type": error_type, "message": message}})
