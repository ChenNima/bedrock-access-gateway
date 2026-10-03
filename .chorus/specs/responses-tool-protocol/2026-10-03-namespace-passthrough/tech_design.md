---
title: "Tech Design: Responses namespace / tool_search support and native passthrough"
proposalUuid: 06fd6331-0217-4263-bf85-083682ad40f0
documentUuid: a98e9671-fc36-4852-86a2-516139266dc0
---

# Tech Design: Responses namespace / tool_search support and native passthrough

Baseline: `ChenNima/bedrock-access-gateway` @ `dbea1f7`. Files touched:
- `src/api/routers/responses.py`
- `src/api/models/responses.py`
- `src/api/schema.py`
- `src/api/setting.py`
- new `src/api/models/responses_native.py`
- `deployment/*.template`
- docs
- tests under `test/`

## 1. Routing (router)

```
POST /v1/responses
  ├─ alias "gpt-*" → DEFAULT_MODEL           (existing)
  ├─ is_native_responses_model(model)?  ── yes ─▶ NativeResponsesProxy.forward(request)
  └─ no ─▶ BedrockResponsesModel (Converse translation, existing + fixes)
```

`setting.py` adds three settings:
- `RESPONSES_NATIVE_MODEL_PATTERNS`: comma-separated `fnmatch` globs, case-insensitive. Default `*openai.gpt-*`. An empty value disables passthrough.
- `RESPONSES_NATIVE_EXCLUDE_PATTERNS`: default `*gpt-oss*`.
- `BEDROCK_RUNTIME_RESPONSES_URL`: optional override. Default `https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1/responses`.

## 2. Native passthrough (`api/models/responses_native.py`)

- **Body.** `payload = request.model_dump(exclude_unset=True)`; with `extra="allow"`, unknown
  fields are kept. Drop `extra_body`. If `"store"` is not in `request.model_fields_set`, set
  `payload["store"] = False`. Write `payload["model"]` after aliasing.
- **Signing.** `botocore.auth.SigV4Auth(credentials, "bedrock", AWS_REGION)` signs a
  `botocore.awsrequest.AWSRequest`.
  - Credentials come from `boto3.Session().get_credentials().get_frozen_credentials()`,
    re-read on every request so Lambda credential rotation is picked up.
  - Headers: `Content-Type: application/json`, and `Accept: text/event-stream` when streaming.
- **HTTP.** `requests.post(url, data, headers, stream=stream, timeout=(10, 900))` runs in
  `run_in_threadpool`.
  - On a non-2xx status: read the body and return a `Response` with the same status code and
    content type, parsing the JSON body when possible.
  - Non-streaming 2xx: return the upstream JSON unchanged with `JSONResponse`.
  - Streaming 2xx: `StreamingResponse(media_type="text/event-stream")` over an async generator
    that pulls `resp.iter_content(chunk_size=None)` through the existing `_aiter` helper
    (thread pool) and yields the raw bytes. Close the upstream response in a `finally`.
- **Logging.** `DEBUG` logs the payload, mirroring the existing debug behaviour. Never log
  credentials.
- **Auth.** The router keeps `api_key_auth`. The client's `Authorization` header is never
  forwarded.
- **IAM.** Bedrock's native Responses endpoint authorises `bedrock:InvokeModel` (and
  `InvokeModelWithResponseStream` for streaming) against both the model / inference profile
  and the account's default **project** resource.
  - The deploy task confirms the exact project ARN from the AWS docs (inference-prereq) and
    from a live denied call, then adds it to the role policy in both templates.
  - The fallback `arn:aws:bedrock:*:*:project/*` is used only if the docs and the observed
    AccessDenied message both fail to pin the ARN down. That choice must be justified in the
    work report.

## 3. Converse translation fixes (`api/models/responses.py`)

### 3.1 Tool registry (per request, stateless)
```python
@dataclass
class ToolEntry:
    kind: Literal["function", "custom", "tool_search"]
    bedrock_name: str
    name: str                 # original name (tool_search → "tool_search")
    namespace: str | None
    description: str | None
    parameters: dict          # JSON schema; custom → {"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}
    source: dict              # original declaration (for the effective-tools echo)

class ToolRegistry:
    by_bedrock: dict[str, ToolEntry]
    by_identity: dict[tuple[str|None, str], ToolEntry]
    def add_declaration(tool: dict, namespace: str | None = None) -> None  # function | namespace | custom | tool_search | hosted(drop+warn)
    def bedrock_name_for(namespace, name) -> str   # used for replay of unknown identities too
    def lookup(bedrock_name) -> ToolEntry | None
```
Sources are added in this order:
1. `request.tools`
2. `additional_tools` input items
3. `tool_search_output.tools` from replayed history (namespace members may carry
   `defer_loading: true`, which is ignored once the tool is loaded)

Duplicate identities keep the first declaration.

**`defer_loading`.** Namespace members marked `defer_loading: true` are registered eagerly
wherever they appear: in `request.tools`, in `additional_tools`, or in `tool_search_output`.
Converse has no deferred-loading concept, so exposing a deferred tool early only costs input
tokens and never hides a tool the client meant to offer. `tool_search` itself still works as a
discovery step, because the model may call it.

**Unverified wire names.** Some names come from the Codex research and are not yet confirmed:
- the streaming event `response.custom_tool_call_input.done`;
- the `status` and `execution` fields of `tool_search_call`;
- the `namespace` field on `custom_tool_call`.

The tool_search/custom task must check each one against the openai/codex `rust-v0.160.0`
source (`protocol/src/models.rs` plus the SSE parser in `codex-api`) or the OpenAI docs before
encoding it in tests. Fields Codex does not parse are left out.

### 3.2 Name mapping
- **Top-level functions** keep their name when it is valid (`^[a-zA-Z0-9_-]{1,64}$`) and free.
- **Namespace members** become `f"{namespace}__{name}"` when that is valid and free.
- **Otherwise** sanitise (`[^a-zA-Z0-9_-]` → `_`), truncate to 55 characters, and append `_`
  plus 8 hex characters of `sha1(f"{namespace}\x00{name}")`.
  - Collisions within the request are resolved by re-hashing with a counter salt.
  - The same `(namespace, name)` set always yields the same names.
- **tool_search** maps to the Bedrock name `tool_search`. If a user function already uses
  that name, the hashed form is used.
- **Replay.** A replayed `function_call` whose identity is not in the registry is mapped
  through the same pure function, and its tool becomes a placeholder tool so that Converse
  accepts the history.

### 3.3 tool_choice
| Responses | Bedrock |
|---|---|
| `"auto"` / None | `auto` |
| `"none"` | `auto` (existing warning) |
| `"required"` | `any` — 400 if the registry is empty |
| `{"type":"function","name":N,"namespace"?:NS}` | `tool: bedrock_name_for(NS,N)` — 400 if not registered |
| `{"type":"custom","name":N}` | `tool` for that custom entry — 400 if missing |
| other (`allowed_tools`, hosted) | `auto` + warning |

The 400 is raised in `build_chat_request` before any Bedrock call.

**Error body (OpenAI style).** Today, `src/api/app.py` gives a typed error body only to the
Anthropic `/messages` routes. Everything else falls through to FastAPI's
`{"detail": ...}`. Add an `is_responses_route(request)` branch to the
`StarletteHTTPException` and `RequestValidationError` handlers. On those routes it renders:
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "tool_choice", "code": null}}
```
- `type` is `invalid_request_error` for 4xx and `server_error` for 5xx.
- `param` is taken from a dict `detail` (`{"message": ..., "param": ...}`) when one is given,
  otherwise it is `null`.
- Native passthrough errors are upstream bodies and are returned unchanged; they never go
  through this handler.
- The `/chat/completions` error shapes stay as they are.

### 3.4 Input replay
| Input item | Chat message |
|---|---|
| `function_call` (+`namespace`) | `AssistantMessage.tool_calls[id=call_id, name=bedrock_name]` |
| `function_call_output` | `ToolMessage(tool_call_id=call_id)` (unchanged) |
| `custom_tool_call` (`input`) | tool_call with arguments `{"input": input}` |
| `custom_tool_call_output` | `ToolMessage` |
| `tool_search_call` (`arguments` object) | tool_call `tool_search` with `json.dumps(arguments)` |
| `tool_search_output` (`tools`) | `ToolMessage` listing the loaded tools as `<bedrock_name>: <description>` lines; the tools are also registered |
| `additional_tools` | no message; its tools are registered |

Consecutive assistant tool calls that share one response are already merged by the chat
layer. Verify this, and keep parallel calls in a single assistant turn.

### 3.5 Output (non-streaming and streaming)
`toolUse.name` is resolved through `registry.lookup`:
- **function** → `ResponsesFunctionCall(name=entry.name, namespace=entry.namespace, call_id=toolUseId, arguments=json)`. `namespace` is omitted when None (`exclude_none` on that field, or a model serializer).
- **custom** → `ResponsesCustomToolCall(type="custom_tool_call", call_id, name, namespace?, input=<arguments.input>)`.
- **tool_search** → `ResponsesToolSearchCall(type="tool_search_call", call_id, execution="client", status="completed", arguments=<object>)`.
- **unknown name** → `ResponsesFunctionCall` with the raw name (defensive).

Streaming (`_StreamSession`):
- Function blocks keep emitting `function_call_arguments.delta`/`.done`, with `namespace` on the `output_item.added` and `output_item.done` items.
- Custom and tool_search blocks buffer their input and emit only `output_item.added` (in_progress) and `output_item.done`. For custom tools, a single `response.custom_tool_call_input.delta` carrying the full input (Codex reads it for its apply_patch preview, `codex-api/src/sse/responses.rs`) and then `response.custom_tool_call_input.done` are emitted before `output_item.done`.

`schema.py` adds `ResponsesCustomToolCall` and `ResponsesToolSearchCall` and puts them in the
`ResponsesOutputItem` union. It also adds `namespace: str | None = None` to
`ResponsesFunctionCall`, serialised only when set.

### 3.6 Effective tools echo
`_base_response` echoes `registry.effective_tools()`: the original declarations that produced
at least one Bedrock tool. A namespace is echoed with only its accepted members. Hosted tools
are excluded.

## 4. Tests (offline, `test/test_responses.py` / new `test/test_responses_native.py`)
- Name mapping: same name in two namespaces, a name over 64 characters, invalid characters, a top-level/namespace collision, determinism.
- tool_choice: required → any, a namespaced named choice → tool, 400 cases.
- Output, non-streaming and streaming: namespace, call_id, custom, tool_search items.
- Replay: function_call(namespace) + output → follow-up toolUse/toolResult pairing, tool_search_output registering tools, additional_tools expansion.
- Effective tools echo.
- Passthrough:
  - rule matching (gpt-6-astra yes, gpt-oss no, Claude no, env override);
  - the store default and an explicit store;
  - SigV4 headers present (the HTTP call is mocked);
  - upstream error status passthrough;
  - stream bytes relayed unchanged.

## 5. Deployment (Tokyo)
1. Build an arm64 image with `scripts/push-to-ecr.sh` (or an equivalent buildx command) and
   push the tag `bedrock-proxy-api:<date>-namespace` to ECR in 659870570537 / ap-northeast-1.
2. Run `aws cloudformation update-stack --stack-name BedrockProxyAPI` with the updated
   `deployment/BedrockProxy.template` and `ContainerImageUri=<new tag>`, keeping the other
   parameters unchanged.
   - The stack currently points at `:latest` while the running image `20261003-toolconfig`
     was set out of band. The update makes the template authoritative again.
   - Record the previous image digest so the change can be rolled back.
3. Smoke-test with curl against `https://ky7cbpqka8.execute-api.ap-northeast-1.amazonaws.com/...`:
   - passthrough with a namespace and `required`, streaming and non-streaming;
   - Converse with Claude and a namespace. Pick the Claude model id from
     `aws bedrock list-inference-profiles` in Tokyo. The stack's DefaultModelId
     `global.anthropic.claude-opus-5` is the first candidate.
4. Run Codex 0.160.0 locally through a temporary `CODEX_HOME` copy pointed at the gateway,
   allowing only `chorus_checkin`. Capture the `mcpToolCall` success event (App Server event
   or rollout jsonl) for both `global.openai.gpt-6-astra` and a Claude model.
