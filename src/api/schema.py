import time
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.setting import DEFAULT_MODEL


class Model(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str | None = "model"
    owned_by: str | None = "bedrock"


class Models(BaseModel):
    object: str | None = "list"
    data: list[Model] = []


class ResponseFunction(BaseModel):
    name: str | None = None
    arguments: str


class ToolCall(BaseModel):
    index: int | None = None
    id: str | None = None
    type: Literal["function"] = "function"
    function: ResponseFunction


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: str | None = "auto"


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class ToolContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class SystemMessage(BaseModel):
    name: str | None = None
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    name: str | None = None
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]


class AssistantMessage(BaseModel):
    name: str | None = None
    role: Literal["assistant"] = "assistant"
    content: str | list[TextContent | ImageContent] | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str | list[ToolContent] | list[dict]
    tool_call_id: str


class DeveloperMessage(BaseModel):
    name: str | None = None
    role: Literal["developer"] = "developer"
    content: str


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: object


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class StreamOptions(BaseModel):
    include_usage: bool = True


class ChatRequest(BaseModel):
    messages: list[SystemMessage | UserMessage | AssistantMessage | ToolMessage | DeveloperMessage]
    model: str = DEFAULT_MODEL
    frequency_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    presence_penalty: float | None = Field(default=0.0, le=2.0, ge=-2.0)  # Not used
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, le=2.0, ge=0.0)
    top_p: float | None = Field(default=None, le=1.0, ge=0.0)
    user: str | None = None  # Not used
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    n: int | None = 1  # Not used
    tools: list[Tool] | None = None
    tool_choice: str | object = "auto"
    stop: list[str] | str | None = None
    extra_body: dict | None = None


class PromptTokensDetails(BaseModel):
    """Details about prompt tokens usage, following OpenAI API format."""
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """Details about completion tokens usage, following OpenAI API format."""
    reasoning_tokens: int = 0
    audio_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class ChatResponseMessage(BaseModel):
    # tool_calls
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


class BaseChoice(BaseModel):
    index: int | None = 0
    finish_reason: str | None = None
    logprobs: dict | None = None


class Choice(BaseChoice):
    message: ChatResponseMessage


class ChoiceDelta(BaseChoice):
    delta: ChatResponseMessage


class BaseChatResponse(BaseModel):
    # id: str = Field(default_factory=lambda: "chatcmpl-" + str(uuid.uuid4())[:8])
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = "fp"


class ChatResponse(BaseChatResponse):
    choices: list[Choice]
    object: Literal["chat.completion"] = "chat.completion"
    usage: Usage


class ChatStreamResponse(BaseChatResponse):
    choices: list[ChoiceDelta]
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    usage: Usage | None = None


class ResponsesReasoningConfig(BaseModel):
    """The `reasoning` object of a Responses request."""

    model_config = ConfigDict(extra="allow")

    effort: str | None = None
    summary: str | None = None
    generate_summary: str | None = None  # Deprecated alias of summary.


class ResponsesTextConfig(BaseModel):
    """The `text` object of a Responses request. Only carried through for echoing back."""

    model_config = ConfigDict(extra="allow")

    format: dict | None = None
    verbosity: str | None = None


class ResponsesRequest(BaseModel):
    """The subset of the OpenAI Responses API this gateway understands.

    `input` items are kept as plain dicts on purpose: the Responses API accepts a dozen
    item types whose shapes keep growing, and a strict union would reject a request over
    a field that never reaches Bedrock anyway. Unknown top-level fields are allowed for
    the same reason.
    """

    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    input: str | list[dict] = ""
    instructions: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, le=2.0, ge=0.0)
    top_p: float | None = Field(default=None, le=1.0, ge=0.0)
    stream: bool | None = False
    reasoning: ResponsesReasoningConfig | None = None
    text: ResponsesTextConfig | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None  # Not used, this gateway is stateless.
    store: bool | None = False  # Not used, nothing is persisted.
    truncation: str | None = "disabled"  # Not used.
    metadata: dict | None = None
    include: list[str] | None = None  # Not used.
    user: str | None = None  # Not used.
    extra_body: dict | None = None


class ResponsesOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict] = []


class ResponsesOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[ResponsesOutputText] = []


class ResponsesReasoningSummary(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponsesReasoningItem(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponsesReasoningSummary] = []
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ResponsesFunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"


ResponsesOutputItem = ResponsesReasoningItem | ResponsesOutputMessage | ResponsesFunctionCall


class ResponsesInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class ResponsesOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponsesUsage(BaseModel):
    input_tokens: int
    input_tokens_details: ResponsesInputTokensDetails = ResponsesInputTokensDetails()
    output_tokens: int
    output_tokens_details: ResponsesOutputTokensDetails = ResponsesOutputTokensDetails()
    total_tokens: int


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "completed"
    model: str
    output: list[ResponsesOutputItem] = []
    usage: ResponsesUsage | None = None
    error: dict | None = None
    incomplete_details: dict | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool = True
    previous_response_id: str | None = None
    reasoning: ResponsesReasoningConfig | None = None
    store: bool = False
    temperature: float | None = None
    text: ResponsesTextConfig | None = None
    tool_choice: str | dict = "auto"
    tools: list[dict] = []
    top_p: float | None = None
    truncation: str = "disabled"
    metadata: dict = {}


class AnthropicCountTokensRequest(BaseModel):
    """The Anthropic Messages API request, minus the generation-only fields.

    Messages, content blocks and tools stay plain dicts for the same reason as the
    Responses input items: the block types keep growing, and most of their fields never
    reach Bedrock. Unknown top-level fields are allowed too.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict]
    system: str | list[dict] | None = None
    tools: list[dict] | None = None
    tool_choice: dict | None = None
    thinking: dict | None = None


class AnthropicMessagesRequest(AnthropicCountTokensRequest):
    max_tokens: int = Field(ge=1)
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    temperature: float | None = Field(default=None, le=1.0, ge=0.0)
    top_p: float | None = Field(default=None, le=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=0)
    output_config: dict | None = None
    context_management: dict | None = None
    metadata: dict | None = None  # Not used.


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict] = []
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = AnthropicUsage()


class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int


class EmbeddingsRequest(BaseModel):
    input: str | list[str] | Iterable[int | Iterable[int]]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None  # Used by Nova embeddings; ignored by other models.
    user: str | None = None  # not used.


class Embedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | bytes
    index: int


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingsUsage


class ErrorMessage(BaseModel):
    message: str


class Error(BaseModel):
    error: ErrorMessage
