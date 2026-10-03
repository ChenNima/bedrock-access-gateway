[中文](./Usage_CN.md)

# Usage Guide

Assuming you have set up below environment variables after deployed:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

**API Example:**
- [Models API](#models-api)
- [Responses API](#responses-api)
- [Anthropic Messages API](#anthropic-messages-api)
- [Embedding API](#embedding-api)
- [Multimodal API](#multimodal-api)
- [Tool Call](#tool-call)
- [Reasoning](#reasoning)
- [Interleaved thinking (beta)](#Interleaved thinking (beta))

## Models API

You can use this API to get a list of supported model IDs.

Also, you can use this API to refresh the model list if new models are added to Amazon Bedrock.


**Example Request**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Example Response**

```bash
[
  ...
  {
    "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  {
    "id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  ...
]
```

## Chat Completions API

### Basic Example with Claude Sonnet 4.5

Claude Sonnet 4.5 is Anthropic's most intelligent model, excelling at coding, complex reasoning, and agent-based tasks. It's available via global cross-region inference profiles.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
      }
    ]
  }'
```

**Example SDK Usage**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."}],
)

print(completion.choices[0].message.content)
```

## Responses API

`POST /responses` serves the newer [Responses API](https://platform.openai.com/docs/api-reference/responses)
next to `/chat/completions`. Use it for clients that only speak Responses, such as the Codex CLI.

### Basic Example

```bash
curl $OPENAI_BASE_URL/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "instructions": "You are a helpful assistant.",
    "input": "Hello!"
  }'
```

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    input="Write a Python function to calculate the Fibonacci sequence using dynamic programming.",
)

print(response.output_text)
```

### Streaming

Each Bedrock content block becomes one output item, so reasoning, text and tool calls arrive as
separate items rather than being flattened into one text stream:

```python
with client.responses.stream(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    input="What is 17*23? Think it through.",
    reasoning={"effort": "low"},
) as stream:
    for event in stream:
        if event.type == "response.reasoning_summary_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

### Tool Call

Function tools are declared flat, as the Responses API expects, and the `call_id` of a
`function_call` is the id you send back in the matching `function_call_output`:

```python
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

history = [{"role": "user", "content": "What is the weather in Paris?"}]
response = client.responses.create(model=MODEL, input=history, tools=tools)

call = next(item for item in response.output if item.type == "function_call")
history += [item.model_dump() for item in response.output]
history.append(
    {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": '{"temp_c": 18, "sky": "clear"}',
    }
)

print(client.responses.create(model=MODEL, input=history, tools=tools).output_text)
```

### Using the Codex CLI

Add a provider to `~/.codex/config.toml`:

```toml
model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
model_provider = "bedrock-gateway"

[model_providers.bedrock-gateway]
name = "Bedrock Access Gateway"
base_url = "<API base url>"
env_key = "BEDROCK_GATEWAY_API_KEY"
wire_api = "responses"
```

`wire_api = "responses"` is the important part. Export `BEDROCK_GATEWAY_API_KEY` with your gateway
API key and run `codex`. Codex will warn that it has no metadata for the model, which is harmless.

OpenAI reasoning models on Bedrock work too. Point `model` at their inference-profile ID and,
optionally, set `model_reasoning_effort`:

```toml
model = "global.openai.gpt-6-astra"
model_provider = "bedrock-gateway"
model_reasoning_effort = "medium"

[model_providers.bedrock-gateway]
name = "Bedrock Access Gateway"
base_url = "<API base url>"
env_key = "BEDROCK_GATEWAY_API_KEY"
wire_api = "responses"
```

These models use the native passthrough described below, so Codex's MCP namespaces,
`tool_search` and custom tools reach them unchanged.

To check that Codex really calls MCP tools, do not rely on an HTTP 200 or on Codex listing the
tools: a dropped tool still gives a 200. Run a prompt that needs one MCP tool. Then open the session
rollout (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`) and look for an `item_completed` event
whose item has `"type":"McpToolCall"`, the expected `server` and `tool`, and
`"status":"completed"`. App Server clients receive the same item as `mcpToolCall`.

### Native Passthrough for OpenAI GPT Models

Bedrock serves the OpenAI GPT models through its own
[Responses API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html)
on `bedrock-runtime`. The gateway therefore forwards their requests as is instead of translating
them to Converse. This applies when the model id matches `*openai.gpt-*` and not `*gpt-oss*`; the
match is made after the `gpt-*` → `DEFAULT_MODEL` alias. The request goes to
`https://bedrock-runtime.<AWS_REGION>.amazonaws.com/openai/v1/responses` and is signed with SigV4
using the gateway's AWS credentials. The gateway API key is still checked and is never forwarded.
Streaming events are relayed byte for byte, and upstream errors keep their status code and body.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RESPONSES_NATIVE_MODEL_PATTERNS` | `*openai.gpt-*` | Comma-separated, case-insensitive glob patterns of model ids to pass through. An empty value disables the passthrough. |
| `RESPONSES_NATIVE_EXCLUDE_PATTERNS` | `*gpt-oss*` | Model ids that match these patterns stay on Converse. |
| `BEDROCK_RUNTIME_RESPONSES_URL` | `https://bedrock-runtime.<AWS_REGION>.amazonaws.com/openai/v1/responses` | Overrides the upstream URL. |

- **store.** When the client does not set `store`, the gateway sends `store: false`, so Bedrock
  keeps no conversation data. An explicit `store: true` is honoured, and `previous_response_id`
  then works. Only `POST /responses` is proxied, so stored responses cannot be fetched through the
  gateway.
- **IAM.** Bedrock authorises `bedrock:InvokeModel` (or `InvokeModelWithResponseStream`) on the
  inference profile and on the account's default project,
  `arn:aws:bedrock:*:<account>:project/default`. Both CloudFormation templates grant it. Add it
  yourself if you use your own role.
- **Endpoint limits.** Name a `us.` / `global.` (or `us-gov.`) cross-Region inference profile.
  Foundation-model ids and application inference profiles are rejected, and so is
  `background: true`. Guardrails do not apply.
- **Hosted tools.** The endpoint runs no hosted tools and rejects a whole request that declares
  one. The gateway therefore drops them here too, with a log warning: only `function`,
  `namespace`, `custom` and `tool_search` with `execution: "client"` tools are forwarded, and
  those unchanged. `web_search` (which Codex sends by default), `file_search`, `mcp`,
  `code_interpreter`, hosted `tool_search` and any other type are removed, so Codex works without
  `web_search = "disabled"`. A `tool_choice` that names a removed tool falls back to `"auto"`. GPT OSS models have no Responses support on `bedrock-runtime`, which is why they are
  excluded and stay on Converse.

If you disable the passthrough, GPT-6 / GPT-5.x go through Converse. They reject the `temperature`
field and return encrypted `redactedContent` instead of plaintext reasoning, so the gateway drops
`temperature` for them and skips the encrypted block.

### Tools on the Converse Path

For every other model, the translation carries the tool protocol that Codex uses:

- **Namespace groups** (how Codex sends MCP tools). Each member becomes a Bedrock tool named
  `<namespace>__<name>`. If that name is invalid, longer than 64 characters or taken, it is
  sanitised, truncated and given a short hash. The mapping is rebuilt from each request, so
  replayed history gets the same names. Calls come back as `function_call` items with the
  original `namespace` and `name`, both streaming and non-streaming. `defer_loading` is ignored,
  so deferred tools are offered up front.
- **tool_choice.** `"required"` becomes Bedrock `any`. A named function (with an optional
  `namespace`) or a named custom tool becomes Bedrock `tool`. When nothing is left to force, or the
  named tool is not declared, the gateway returns 400 before calling Bedrock:
  `{"error": {"type": "invalid_request_error", "param": "tool_choice", ...}}`.
- **tool_search** with `execution: "client"` is offered to the model as a function. The model's
  call is returned as a `tool_search_call` item. When a later request replays a
  `tool_search_output`, the tools it lists are added to that request.
- **additional_tools** input items add their tools to the request's tool list.
- **Custom tools** (for example Codex's grammar-based `exec`) are sent as a function that takes a
  single string `input`, and calls come back as `custom_tool_call` items. The grammar goes into
  the tool description, because Bedrock cannot enforce it.

### Limitations

These apply to the Converse path. The gateway is stateless, so `store` and `previous_response_id`
are ignored — send the whole conversation in `input`, which is what the Codex CLI does. Reasoning
items you send back in `input` are dropped, because Bedrock only accepts a reasoning block together
with the signature it issued and that signature has no place in the Responses wire format. Hosted
tools (`web_search`, `file_search`, hosted `tool_search`, ...) are dropped with a log warning,
having no Bedrock counterpart. The response's `tools` echoes only the declarations from the
request's `tools` that took effect. `tool_choice: "none"` falls back to `"auto"`, and
`allowed_tools` is ignored.

When a request enables reasoning without `max_output_tokens`, the gateway has to supply the
maxTokens that Bedrock requires; it uses `DEFAULT_MAX_TOKENS` (32,768 by default).

## Anthropic Messages API

`POST /messages` serves the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
and `POST /messages/count_tokens` its token counting endpoint. Use them for clients that speak
the Anthropic format, such as the Anthropic SDKs and Claude Code. The API key is accepted as
`x-api-key` or as `Authorization: Bearer`.

Anthropic clients append `/v1/messages` to their base URL, so their base URL is the gateway root
without the trailing `/v1` (e.g. `http://localhost:8000/api`). If you change `API_ROUTE_PREFIX`,
keep it ending in `/v1`.

### Basic Example

```bash
curl $ANTHROPIC_BASE_URL/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -d '{
    "model": "claude-haiku-4-5",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2048},
    messages=[{"role": "user", "content": "What is 17*23?"}],
)

for block in message.content:
    print(block.type, getattr(block, "thinking", None) or getattr(block, "text", ""))
```

`model` can be a Bedrock model ID or inference profile, or a first-party name such as
`claude-sonnet-4-5` / `claude-opus-4-6`, which is mapped onto the matching Bedrock inference
profile (global first, then regional). Tool use, images, PDFs, streaming, extended and adaptive
thinking and `cache_control` work as they do on the Anthropic API.

### Using Claude Code

```bash
export ANTHROPIC_BASE_URL=<API base url without /v1>   # e.g. http://localhost:8000/api
export ANTHROPIC_API_KEY=<API key>
claude
```

Make sure `CLAUDE_CODE_USE_BEDROCK` is not set, otherwise Claude Code calls Bedrock directly
instead of the gateway. Claude Code's default model names work unchanged. To use another
Bedrock model, including non-Claude ones:

```bash
export ANTHROPIC_MODEL=global.openai.gpt-6-luna            # or qwen.qwen3-coder-480b-a35b-v1:0, ...
export ANTHROPIC_SMALL_FAST_MODEL=global.anthropic.claude-haiku-4-5-20251001-v1:0
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16000                 # if the model's output limit is below 32k
```

Claude Code warns that a non-Claude model ID is not in its model catalog. The warning is harmless,
but set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the model's real context window so auto-compact
triggers at the right time.

### Limitations

- Claude-only request fields (`thinking`, `output_config`, `context_management`, `top_k`) are
  passed to Claude and dropped for other models.
- Of the `anthropic-beta` flags, only those listed in `ANTHROPIC_BETA_ALLOWLIST` are forwarded,
  because Bedrock rejects a whole request over a flag it does not know. The default covers
  interleaved thinking, 1M context, context management, effort and fine-grained tool streaming.
- Server tools (`web_search`, `web_fetch`, `code_execution`, ...) are dropped with a log warning,
  so Claude Code's WebSearch tool returns nothing. Citations are not supported.
- Images inside a `tool_result` (how Claude Code passes screenshots and images it reads) go to
  Claude as is. Other vision models on Bedrock only accept images outside a tool result, so for
  them the gateway moves the images right after it. Text-only models reject images with a 400.
- Unsigned thinking blocks (from models such as DeepSeek or Qwen) are not sent back to Bedrock.
- `tool_choice: {"type": "none"}` falls back to `auto`.
- `count_tokens` uses Bedrock CountTokens where the model supports it and a tiktoken estimate
  otherwise.

## Embedding API

**Important Notice**: Please carefully review the following points before using this proxy API for embedding.

1. If you have previously used OpenAI embedding models to create vectors, be aware that switching to a new model may not be straightforward. Different models have varying dimensions (e.g., embed-multilingual-v3.0 has 1024 dimensions), and even for the same text, they may produce different results.
2. If you are using OpenAI embedding models for encoded integers (such as with LangChain), this solution will attempt to decode the integers using `tiktoken` to retrieve the original text. However, there is no guarantee that the decoded text will be accurate.
3. If you are using OpenAI embedding models for long texts, you should verify the maximum number of tokens supported for Bedrock models, e.g. for optimal performance, Bedrock recommends limiting the text length to less than 512 tokens.


**Example Request**

```bash
curl $OPENAI_BASE_URL/embeddings \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "input": "The food was delicious and the waiter...",
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }'
```

**Example Response**

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.02279663,
                -0.024612427,
                0.012863159,
                ...
                0.01612854,
                0.0038928986
            ],
            "index": 0
        }
    ],
    "model": "cohere.embed-multilingual-v3",
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
}
```

Alternatively, you can use the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

text = "hello"
# will output like [0.003578186, 0.028717041, 0.031021118, -0.0014066696,...]
print(get_embedding(text))
```

Or LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:5])
doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])
```

## Multimodal API

> [!NOTE]
> When you pass a remote `image_url`, the gateway fetches it server-side. Only `http(s)` URLs that resolve
> to a publicly routable address are fetched; URLs pointing at the instance metadata service, `localhost`,
> or private ranges are rejected with a `400`. You can restrict this further with `IMAGE_URL_ALLOWED_HOSTS`
> or turn it off with `ENABLE_IMAGE_URL_FETCH=false` — see the [Security Guide](./Security.md#2-image-url-fetching-ssrf).

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in these images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/aws-samples/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

If you need to use this API with non-public images, you can do base64 the image first and pass the encoded string. 
Replace `image/jpeg` with the actual content type. Currently, only 'image/jpeg', 'image/png', 'image/gif' or 'image/webp' is supported.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in this images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,<your image data>"
                    }
                }
            ]
        }
    ]
}'
```

**Example Response**

```json
{
    "id": "msg_01BY3wcz41x7XrKhxY3VzWke",
    "created": 1712543069,
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The image contains the following objects:\n\n1. A peach-colored short-sleeve button-up shirt\n2. An olive green plaid long coat/jacket\n3. A pair of white sneakers or canvas shoes\n4. A brown shoulder bag or purse\n5. A makeup brush or cosmetic applicator\n6. A tube or container (possibly lipstick or lip balm)\n7. A pair of sunglasses\n8. A thought bubble icon\n9. A footprint icon\n10. A leaf or plant icon\n11. A flower icon\n12. A cloud icon\n\nIn total, there are 12 distinct objects depicted in the illustrated scene."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 197,
        "completion_tokens": 147,
        "total_tokens": 344
    }
}
```


## Tool Call

**Important Notice**: Please carefully review the following points before using this Tool Call for Chat completion API.

1. Function Call is now deprecated in favor of Tool Call by OpenAI, hence it's not supported here, you should use Tool Call instead.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Shanghai today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or state which is required."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "Use this tool to get the current location if user does not provide a location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ],
    "tool_choice": "auto"
}'
```

**Example Response**

```json
{
    "id": "msg_01PjrKDWhYGsrTNdeqzWd6D9",
    "created": 1712543689,
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "0",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"Shanghai\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "total_tokens": 320
    }
}
```

You can try it with different questions, such as:
1. Hello, who are you?  (No tools are needed)
2. What is the weather like today?  (Should use get_current_location tool first)


## Reasoning

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.
- Only Claude 3.7 Sonnet (extended thinking) and DeepSeek R1 support Reasoning so far. Please make sure the model supports reasoning before use.
- For Claude 3.7 Sonnet, the reasoning mode (or thinking mode) is not enabled by default, you must pass additional `reasoning_effort` parameter in your request. Please also provide the right max_tokens (or max_completion_tokens) in your request. The budget_tokens is based on reasoning_effort (low: 30%, medium: 60%, high: 100% of max tokens), ensuring minimum budget_tokens of 1,024 with Anthropic recommending at least 4,000 tokens for comprehensive reasoning. Check [Bedrock Document](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-37.html) for more details.
- For DeepSeek R1, you don't need additional reasoning_effort parameter, otherwise, you may get an error.
- The reasoning response (CoT, thoughts) is added in an additional tag 'reasoning_content' which is not officially supported by OpenAI. This is to follow [Deepseek Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model#api-example). This may be changed in the future.

> **Note**: Omitting `max_tokens` (or `max_completion_tokens`) now allows Bedrock to use the model's native maximum output (e.g., 64K for Claude Sonnet 4) rather than the previous 2048 implicit cap. Please set `max_tokens` explicitly if you need to control costs or latency.

**Example Request**

- Claude 3.7 Sonnet

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "messages": [
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "stream": false
}'
```

- DeepSeek R1

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.deepseek.r1-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "stream": false
}'
```

**Example Response**

```json
{
    "id": "chatcmpl-83fb7a88",
    "created": 1740545278,
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "3.9 is bigger than 3.11.\n\nWhen comparing decimal numbers, we need to understand what these numbers actually represent:...",
                "reasoning_content": "I need to compare the decimal numbers 3.9 and 3.11.\n\nFor decimal numbers, we first compare the whole number parts, and if they're equal, we compare the decimal parts. \n\nBoth numbers ..."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 51,
        "completion_tokens": 565,
        "total_tokens": 616
    }
}
```

You can also use OpenAI SDK (run `pip3 install -U openai` first )

- Non-Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "which one is bigger, 3.9 or 3.11?"}]
response = client.chat.completions.create(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

- Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
    stream=True,
)

reasoning_content = ""
content = ""

for chunk in response:
    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    elif chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
```

## Interleaved thinking (beta)

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.

Extended thinking with tool use in Claude 4 models supports [interleaved thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-tool-use-interleaved) enables Claude 4 models to think between tool calls and run more sophisticated reasoning after receiving tool results. which is helpful for more complex agentic interactions.
With interleaved thinking, the `budget_tokens` can exceed the `max_tokens` parameter because it represents the total budget across all thinking blocks within one assistant turn.

**Supported Models**: Claude Sonnet 4, Claude Sonnet 4.5

**Example Request**

- Non-Streaming (Claude Sonnet 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Non-Streaming (Claude Sonnet 4)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Sonnet 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Sonnet 4)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```
