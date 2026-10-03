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

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterable, Literal

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
    ResponsesCustomToolCall,
    ResponsesFunctionCall,
    ResponsesInputTokensDetails,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesOutputTokensDetails,
    ResponsesReasoningItem,
    ResponsesReasoningSummary,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesToolSearchCall,
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
ITEM_ID_PREFIX = {"text": "msg", "reasoning": "rs", "function": "fc", "custom": "ctc", "tool_search": "tsc"}


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


# Bedrock's toolSpec.name constraint.
VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
INVALID_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
# Leaves room for "_" plus 8 hex digits of hash within the 64-character limit.
HASHED_NAME_PREFIX_LENGTH = 55

EMPTY_SCHEMA = {"type": "object", "properties": {}}
# A custom (freeform) tool takes one free-text input. Converse only has JSON tools, so it is
# exposed as a function with a single string argument.
CUSTOM_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "The raw input for the tool."}},
    "required": ["input"],
}
TOOL_SEARCH_NAME = "tool_search"
# tool_search is not a (namespace, name) the client can call directly, so it gets an
# identity no function can have.
TOOL_SEARCH_IDENTITY = ("\x00tool_search", TOOL_SEARCH_NAME)
TOOL_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query for deferred tools."},
        "limit": {"type": "number", "description": "Maximum number of tools to return."},
    },
    "required": ["query"],
}

ToolKind = Literal["function", "custom", "tool_search"]
PLACEHOLDER_SCHEMAS = {"function": EMPTY_SCHEMA, "custom": CUSTOM_TOOL_SCHEMA, "tool_search": TOOL_SEARCH_SCHEMA}


def _hashed_tool_name(namespace: str | None, name: str, salt: int = 0) -> str:
    """A valid Bedrock name for a tool whose natural name is invalid, too long or taken."""
    base = f"{namespace}__{name}" if namespace else name
    base = INVALID_TOOL_NAME_CHARS.sub("_", base)[:HASHED_NAME_PREFIX_LENGTH]
    key = f"{namespace or ''}\x00{name}"
    if salt:
        key += f"\x00{salt}"
    return f"{base}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"  # nosec B324 - not security relevant


@dataclass
class ToolEntry:
    """One tool as Bedrock sees it, and where it came from on the Responses side."""

    kind: ToolKind
    bedrock_name: str
    name: str
    namespace: str | None
    description: str | None
    parameters: dict
    # The declaration it came from; None for a placeholder standing in for a tool that
    # only appears in replayed history.
    source: dict | None = None

    @property
    def effective(self) -> bool:
        return self.source is not None


@dataclass
class ToolRegistry:
    """The tools of one request, keyed both by Bedrock name and by Responses identity.

    The Responses API groups tools into namespaces and identifies a call by
    (namespace, name), while Bedrock has a single flat list of names limited to
    ``^[a-zA-Z0-9_-]{1,64}$``. The registry assigns every identity a Bedrock name and maps
    tool uses back. It is rebuilt for every request: the mapping depends only on the
    declarations and their order, so a client replaying history gets the same names.
    """

    by_bedrock: dict[str, ToolEntry] = field(default_factory=dict)
    by_identity: dict[tuple[str | None, str], ToolEntry] = field(default_factory=dict)
    # Declarations as echoed back in the response, trimmed to what was accepted.
    declarations: list[dict] = field(default_factory=list)

    @classmethod
    def from_tools(cls, tools: list[dict] | None) -> "ToolRegistry":
        registry = cls()
        for tool in tools or []:
            if isinstance(tool, dict):
                registry.add_declaration(tool)
        return registry

    def add_declaration(self, tool: dict, echo: bool = True) -> bool:
        """Register a top-level tool declaration. Returns whether anything was accepted."""
        tool_type = tool.get("type")
        if tool_type == "namespace":
            namespace = tool.get("name")
            if not namespace:
                return False
            members = [
                member
                for member in tool.get("tools") or []
                if isinstance(member, dict) and self._add_member(member, namespace)
            ]
            if members and echo:
                self.declarations.append({**tool, "tools": members})
            return bool(members)

        if tool_type in ("function", "custom"):
            accepted = self._add_member(tool, None)
            if accepted and echo:
                self.declarations.append(tool)
            return accepted

        if tool_type == TOOL_SEARCH_NAME:
            accepted = self._add_tool_search(tool)
            if accepted and echo:
                self.declarations.append(tool)
            return accepted

        # Hosted tools (web_search, file_search, computer_use, ...) run inside OpenAI and
        # have no Bedrock counterpart.
        logger.warning("Ignoring unsupported Responses tool of type %s", tool_type)
        return False

    def _add_member(self, tool: dict, namespace: str | None) -> bool:
        """Register one callable tool, top-level or inside a namespace."""
        if tool.get("type") == "custom":
            return self._add_custom(tool, namespace)
        if tool.get("type") != "function":
            logger.warning(
                "Ignoring unsupported Responses tool of type %s in namespace %s", tool.get("type"), namespace
            )
            return False
        # Responses declares function tools flat; accept the nested Chat Completions shape too.
        # defer_loading is ignored: Converse cannot load tools lazily, so a deferred tool is
        # exposed up front, which costs input tokens but never hides a tool.
        spec = tool["function"] if isinstance(tool.get("function"), dict) else tool
        name = spec.get("name")
        if not name:
            return False
        return self._register(
            "function",
            namespace,
            name,
            spec.get("description"),
            spec.get("parameters") or EMPTY_SCHEMA,
            tool,
        )

    def _add_custom(self, tool: dict, namespace: str | None) -> bool:
        name = tool.get("name")
        if not name:
            return False
        return self._register("custom", namespace, name, _custom_tool_description(tool), CUSTOM_TOOL_SCHEMA, tool)

    def _add_tool_search(self, tool: dict) -> bool:
        # Only a client-executed search can be served: the model's call is handed back to
        # the client, which runs the search and replays the result as tool_search_output.
        # A hosted search would have to run inside OpenAI.
        if tool.get("execution") != "client":
            logger.warning("Ignoring tool_search with execution %s, only client is supported", tool.get("execution"))
            return False
        return self._register(
            "tool_search",
            None,
            TOOL_SEARCH_NAME,
            tool.get("description"),
            tool.get("parameters") or TOOL_SEARCH_SCHEMA,
            tool,
            identity=TOOL_SEARCH_IDENTITY,
        )

    def _register(
        self,
        kind: ToolKind,
        namespace: str | None,
        name: str,
        description: str | None,
        parameters: dict,
        source: dict | None,
        identity: tuple[str | None, str] | None = None,
    ) -> bool:
        identity = identity or (namespace, name)
        if identity in self.by_identity:
            # The first declaration of an identity wins.
            logger.warning("Ignoring duplicate Responses tool %s", f"{namespace}.{name}" if namespace else name)
            return False
        entry = ToolEntry(
            kind=kind,
            bedrock_name=self._free_name(namespace, name),
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters,
            source=source,
        )
        self.by_identity[identity] = entry
        self.by_bedrock[entry.bedrock_name] = entry
        return True

    def add_loaded_tools(self, tools: list) -> None:
        """Register tools that reach the model outside request.tools.

        That is the additional_tools input item and the tools a tool_search_output loaded.
        They are not echoed: the response's tools field mirrors the request's.
        """
        for tool in tools or []:
            if isinstance(tool, dict):
                self.add_declaration(tool, echo=False)

    def _free_name(self, namespace: str | None, name: str) -> str:
        preferred = f"{namespace}__{name}" if namespace else name
        if VALID_TOOL_NAME.match(preferred) and preferred not in self.by_bedrock:
            return preferred
        salt = 0
        candidate = _hashed_tool_name(namespace, name)
        while candidate in self.by_bedrock:
            salt += 1
            candidate = _hashed_tool_name(namespace, name, salt)
        return candidate

    def resolve(self, namespace: str | None, name: str) -> ToolEntry | None:
        """The declared tool with this identity, ignoring history placeholders."""
        entry = self.by_identity.get((namespace or None, name))
        return entry if entry is not None and entry.effective else None

    def bedrock_name_for(self, namespace: str | None, name: str, kind: ToolKind = "function") -> str:
        """The Bedrock name of a tool replayed from history.

        A call to a tool the request no longer declares still has to be replayed, and
        Converse rejects a toolUse whose name is not in toolConfig, so the identity gets a
        placeholder tool.
        """
        namespace = namespace or None
        identity = TOOL_SEARCH_IDENTITY if kind == "tool_search" else (namespace, name)
        entry = self.by_identity.get(identity)
        if entry is None:
            self._register(
                kind,
                namespace,
                name,
                "Tool used earlier in this conversation.",
                PLACEHOLDER_SCHEMAS[kind],
                None,
                identity=identity,
            )
            entry = self.by_identity[identity]
        return entry.bedrock_name

    def lookup(self, bedrock_name: str) -> ToolEntry | None:
        return self.by_bedrock.get(bedrock_name)

    @property
    def has_effective_tools(self) -> bool:
        return any(entry.effective for entry in self.by_bedrock.values())

    def chat_tools(self) -> list[Tool] | None:
        # Placeholders only matter alongside declared tools. Without any, the chat layer
        # already builds a toolConfig from the history on its own.
        if not self.has_effective_tools:
            return None
        return [
            Tool(
                function=Function(
                    name=entry.bedrock_name,
                    description=entry.description,
                    parameters=entry.parameters,
                )
            )
            for entry in self.by_bedrock.values()
        ]

    def effective_tools(self) -> list[dict]:
        return list(self.declarations)


def _custom_tool_description(tool: dict) -> str | None:
    """The description of a custom tool, with its input format spelled out.

    The format is enforced by OpenAI's constrained decoding; Bedrock can only be told.
    """
    description = tool.get("description")
    tool_format = tool.get("format")
    if not isinstance(tool_format, dict) or tool_format.get("type") != "grammar" or not tool_format.get("definition"):
        return description
    note = (
        "Put the raw tool input in the `input` string. It must match this "
        f"{tool_format.get('syntax') or ''} grammar:\n{tool_format['definition']}"
    )
    return f"{description}\n\n{note}" if description else note


def _loaded_tools_text(tools: list, registry: ToolRegistry) -> str:
    """The toolResult for a tool_search_output: the tools it made callable, one per line."""
    lines = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            members = [(tool.get("name"), member) for member in tool.get("tools") or [] if isinstance(member, dict)]
        else:
            members = [(None, tool)]
        for namespace, member in members:
            spec = member["function"] if isinstance(member.get("function"), dict) else member
            entry = registry.resolve(namespace, spec.get("name") or "")
            if entry is not None:
                lines.append(f"{entry.bedrock_name}: {entry.description or ''}".rstrip())
    if not lines:
        return "No tools were found."
    return "The following tools are now available:\n" + "\n".join(lines)


def _tool_call_message(call_id: str, bedrock_name: str, arguments: str) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=[
            ToolCall(
                id=call_id,
                type="function",
                function=ResponseFunction(name=bedrock_name, arguments=arguments),
            )
        ]
    )


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


def _convert_input_item(item: dict, registry: ToolRegistry) -> list:
    """Convert one Responses input item into zero or more chat messages."""
    item_type = item.get("type")

    call_id = item.get("call_id") or item.get("id") or ""

    if item_type == "function_call":
        # An empty argument string is valid on the wire but not valid JSON, and the chat
        # layer json.loads() it on the way to Bedrock.
        return [
            _tool_call_message(
                call_id,
                registry.bedrock_name_for(item.get("namespace"), item.get("name") or ""),
                item.get("arguments") or "{}",
            )
        ]
    if item_type == "custom_tool_call":
        return [
            _tool_call_message(
                call_id,
                registry.bedrock_name_for(item.get("namespace"), item.get("name") or "", "custom"),
                json.dumps({"input": item.get("input") or ""}),
            )
        ]
    if item_type == "tool_search_call":
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            # Not the wire shape (an object), but cheap to accept.
            arguments = _parse_arguments(arguments)
        return [
            _tool_call_message(
                call_id,
                registry.bedrock_name_for(None, TOOL_SEARCH_NAME, "tool_search"),
                json.dumps(arguments or {}),
            )
        ]
    if item_type in ("function_call_output", "custom_tool_call_output"):
        return [
            ToolMessage(
                tool_call_id=item.get("call_id") or "",
                content=_stringify(item.get("output")),
            )
        ]
    if item_type == "tool_search_output":
        # Its tools were registered before the input was replayed (see _register_input_tools).
        return [
            ToolMessage(tool_call_id=item.get("call_id") or "", content=_loaded_tools_text(item.get("tools"), registry))
        ]
    if item_type == "additional_tools":
        # Codex sends its tools this way instead of in request.tools; they were registered
        # up front and the item itself carries no conversation.
        return []
    if item_type in IGNORED_INPUT_ITEM_TYPES:
        return []
    if item_type in (None, "message") or "role" in item:
        return _convert_message_item(item)

    logger.warning("Ignoring unsupported Responses input item of type %s", item_type)
    return []


def _register_input_tools(value: str | list[dict], registry: ToolRegistry) -> None:
    """Register the tools input items declare, before any history is replayed.

    additional_tools come first, then tools loaded by tool_search_output, so a replayed call
    maps onto the declared tool rather than onto a placeholder.
    """
    if isinstance(value, str):
        return
    items = [item for item in value if isinstance(item, dict)]
    for item_type in ("additional_tools", "tool_search_output"):
        for item in items:
            if item.get("type") == item_type:
                registry.add_loaded_tools(item.get("tools"))


def _convert_input(value: str | list[dict], registry: ToolRegistry) -> list:
    if isinstance(value, str):
        return [UserMessage(content=value)] if value else []

    messages = []
    for item in value:
        if isinstance(item, dict):
            messages.extend(_convert_input_item(item, registry))
    return messages


def _tool_choice_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"message": message, "param": "tool_choice"})


def _convert_tool_choice(tool_choice: str | dict | None, registry: ToolRegistry) -> str | dict:
    """Map tool_choice onto the chat layer, rejecting a choice no declared tool can satisfy.

    Bedrock would otherwise fail the call with a less helpful error, or (for "required"
    with no tools) silently drop the constraint.
    """
    if tool_choice is None or tool_choice == "auto":
        return "auto"
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            # Bedrock's toolChoice has no "none"; the closest behaviour is to let the model decide.
            logger.warning('tool_choice "none" is not supported by Bedrock, falling back to "auto"')
            return "auto"
        if tool_choice == "required":
            if not registry.has_effective_tools:
                raise _tool_choice_error('tool_choice "required" needs at least one supported tool in tools')
            return "required"
        logger.warning("Ignoring unsupported tool_choice %s", tool_choice)
        return "auto"

    # Responses names the tool flat; accept the nested Chat Completions shape too.
    if tool_choice.get("type") == "function" or "function" in tool_choice:
        spec = tool_choice["function"] if isinstance(tool_choice.get("function"), dict) else tool_choice
        name = spec.get("name") or ""
        namespace = spec.get("namespace")
        entry = registry.resolve(namespace, name)
        if entry is None or entry.kind != "function":
            label = f"{namespace}.{name}" if namespace else name
            raise _tool_choice_error(f"tool_choice names function {label!r}, which is not in tools")
        return {"function": {"name": entry.bedrock_name}}
    if tool_choice.get("type") == "custom":
        name = tool_choice.get("name") or ""
        namespace = tool_choice.get("namespace")
        entry = registry.resolve(namespace, name)
        if entry is None or entry.kind != "custom":
            label = f"{namespace}.{name}" if namespace else name
            raise _tool_choice_error(f"tool_choice names custom tool {label!r}, which is not in tools")
        return {"function": {"name": entry.bedrock_name}}
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
        # The registry of the request last converted, so the response side can map tool
        # uses back without converting the request twice.
        self._registry: tuple[ResponsesRequest, ToolRegistry] | None = None

    def _convert(self, request: ResponsesRequest) -> tuple[ToolRegistry, list]:
        # Declared tools are registered before the input is replayed, so history naming a
        # declared tool maps onto it rather than onto a placeholder.
        registry = ToolRegistry.from_tools(request.tools)
        _register_input_tools(request.input, registry)
        messages = []
        if request.instructions:
            messages.append(SystemMessage(content=request.instructions))
        messages.extend(_convert_input(request.input, registry))
        self._registry = (request, registry)
        return registry, messages

    def _registry_for(self, request: ResponsesRequest) -> ToolRegistry:
        if self._registry is not None and self._registry[0] is request:
            return self._registry[1]
        return self._convert(request)[0]

    def build_chat_request(self, request: ResponsesRequest) -> ChatRequest:
        """Convert and validate a Responses request. Raises HTTPException on bad input."""
        if DEBUG:
            logger.info("Raw Responses request: " + request.model_dump_json())

        registry, messages = self._convert(request)

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
            tools=registry.chat_tools(),
            tool_choice=_convert_tool_choice(request.tool_choice, registry),
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
            # Only what reached Bedrock, so a client can tell which tools were dropped.
            tools=self._registry_for(request).effective_tools(),
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
        registry = self._registry_for(request)
        for tool_use in tool_calls:
            response.output.append(
                _tool_call_item(
                    registry,
                    generate_id(ITEM_ID_PREFIX[_tool_kind(registry, tool_use["name"])]),
                    tool_use["toolUseId"],
                    tool_use["name"],
                    json.dumps(tool_use.get("input") or {}),
                    "completed",
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
        chat_request = chat_request or self.build_chat_request(request)
        session = _StreamSession(
            self._base_response(request, response_id, int(time.time())), self._registry_for(request)
        )

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


def _tool_kind(registry: ToolRegistry, bedrock_name: str) -> ToolKind:
    entry = registry.lookup(bedrock_name)
    return entry.kind if entry else "function"


def _parse_arguments(arguments: str) -> dict:
    """The toolUse input as an object; Bedrock's partial JSON only parses once complete."""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        logger.warning("Tool input is not valid JSON: %s", arguments)
        return {}
    return value if isinstance(value, dict) else {}


def _tool_call_item(
    registry: ToolRegistry, item_id: str, call_id: str, bedrock_name: str, arguments: str, status: str
) -> ResponsesFunctionCall | ResponsesCustomToolCall | ResponsesToolSearchCall:
    """The output item for a Bedrock toolUse, typed and named the way the client declared the tool."""
    entry = registry.lookup(bedrock_name)
    if entry is not None and entry.kind == "custom":
        value = _parse_arguments(arguments).get("input", "") if status == "completed" else ""
        return ResponsesCustomToolCall(
            id=item_id,
            call_id=call_id,
            name=entry.name,
            namespace=entry.namespace,
            input=value if isinstance(value, str) else json.dumps(value),
            status=status,
        )
    if entry is not None and entry.kind == "tool_search":
        return ResponsesToolSearchCall(
            id=item_id,
            call_id=call_id,
            arguments=_parse_arguments(arguments) if status == "completed" else {},
            status=status,
        )
    return ResponsesFunctionCall(
        id=item_id,
        call_id=call_id,
        # A name Bedrock made up is passed through as is.
        name=entry.name if entry else bedrock_name,
        namespace=entry.namespace if entry else None,
        arguments=arguments,
        status=status,
    )


class _OpenBlock:
    """A Bedrock content block that is currently streaming, and the item it maps to."""

    def __init__(self, kind: str, output_index: int, item_id: str, call_id: str = "", name: str = ""):
        self.kind = kind  # "text" | "reasoning" | a ToolKind
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

    def __init__(self, response: ResponsesResponse, registry: ToolRegistry | None = None):
        self.response = response
        self.registry = registry or ToolRegistry()
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
                name = start["toolUse"]["name"]
                return self._open(
                    chunk["contentBlockStart"]["contentBlockIndex"],
                    _tool_kind(self.registry, name),
                    call_id=start["toolUse"]["toolUseId"],
                    name=name,
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
        if block.kind == "function":
            return [
                self.event(
                    "response.function_call_arguments.delta",
                    {"item_id": block.item_id, "output_index": block.output_index, "delta": text},
                )
            ]
        # Custom and tool_search input arrives as JSON wrapping the real input, which only
        # parses once complete, so it is buffered and sent with the finished item.
        return []

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
        elif block.kind == "function":
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
        elif block.kind == "custom":
            # The whole input as one delta (Codex previews apply_patch input from these),
            # then the done event that closes the input.
            payload = {"item_id": block.item_id, "output_index": block.output_index}
            if item.input:
                events.append(self.event("response.custom_tool_call_input.delta", {**payload, "delta": item.input}))
            events.append(self.event("response.custom_tool_call_input.done", {**payload, "input": item.input}))

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
        # An empty argument string is valid on the wire but clients json.loads() it.
        return _tool_call_item(self.registry, block.item_id, block.call_id, block.name, block.buffer or "{}", status)

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
