# Bedrock Access Gateway

OpenAI-compatible RESTful APIs for Amazon Bedrock

> [!IMPORTANT]
> **This project is deprecated.** Amazon Bedrock now serves OpenAI-compatible and
> Anthropic-compatible APIs natively, which is the reason this proxy existed. Call Amazon Bedrock
> directly instead of deploying this gateway.

## Migrating to native Amazon Bedrock APIs

Point your SDK at the [`bedrock-mantle`](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html) endpoint:

- **Claude models** — native [Anthropic Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html), so the Anthropic SDKs work unchanged.
- **GPT and other first- and third-party models** — OpenAI-compatible [Chat Completions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html) and [Responses](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html) APIs, so the OpenAI SDKs work unchanged.

Beyond replacing this proxy, `bedrock-mantle` adds per-application [cost attribution](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-projects.html), server-side tools, background inference and higher throughput limits. See [Endpoints supported by Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html) to compare it with `bedrock-runtime`.

## Overview

Amazon Bedrock offers a wide range of foundation models (such as Claude 3 Opus/Sonnet/Haiku, Llama 2/3, Mistral/Mixtral,
etc.) and a broad set of capabilities for you to build generative AI applications. Check the [Amazon Bedrock](https://aws.amazon.com/bedrock) landing page for additional information.

Sometimes, you might have applications developed using OpenAI APIs or SDKs, and you want to experiment with Amazon Bedrock without modifying your codebase. Or you may simply wish to evaluate the capabilities of these foundation models in tools like AutoGen etc. Well, this repository allows you to access Amazon Bedrock models seamlessly through OpenAI APIs and SDKs, enabling you to test these models without code changes.

If you find this GitHub repository useful, please consider giving it a free star ⭐ to show your appreciation and support for the project.

**Features:**

- [x] Support streaming response via server-sent events (SSE)
- [x] Support Model APIs
- [x] Support Chat Completion APIs
- [x] Support Responses API, so the Codex CLI works against Bedrock (**new**)
- [x] Support Anthropic Messages API, so Claude Code works against Bedrock (**new**)
- [x] Support Tool Call
- [x] Support Embedding API
- [x] Support Multimodal API
- [x] Support Cross-Region Inference
- [x] Support Application Inference Profiles (**new**)
- [x] Support Reasoning (**new**)
- [x] Support Interleaved thinking (**new**)
- [x] Support Prompt Caching (**new**)

Please check [Usage Guide](./docs/Usage.md) for more details about how to use the new APIs.


## Get Started

### Prerequisites

Please make sure you have met below prerequisites:

- Access to Amazon Bedrock foundation models.

> For more information on how to request model access, please refer to the [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) (Set Up > Model access)

### Architecture

The following diagram illustrates the reference architecture. It uses [Amazon API Gateway response streaming](https://aws.amazon.com/blogs/compute/building-responsive-apis-with-amazon-api-gateway-response-streaming/) with Lambda for SSE support.

![Architecture](assets/arch.png)

### Deployment Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **API Gateway + Lambda** | No VPC required, pay-per-request, native streaming support, lower operational overhead | Potential cold starts | Most use cases, cost-sensitive deployments |
| **ALB + Fargate** | Lowest streaming latency, no cold starts | Higher cost, requires VPC | High-throughput, latency-sensitive workloads |

You can also use Lambda Function URL as an alternative, see [example](https://github.com/awslabs/aws-lambda-web-adapter/tree/main/examples/fastapi-response-streaming)

### Deployment

Please follow the steps below to deploy the Bedrock Proxy APIs into your AWS account. Only supports regions where Amazon Bedrock is available (such as `us-west-2`). The deployment will take approximately **10-15 minutes** 🕒.

**Step 1: Create your own API key in Secrets Manager (MUST)**

> **Note:** This step is to use any string (without spaces) you like to create a custom API Key (credential) that will be used to access the proxy API later. This key does not have to match your actual OpenAI key, and you don't need to have an OpenAI API key. please keep the key safe and private.

1. Open the AWS Management Console and navigate to the AWS Secrets Manager service.
2. Click on "Store a new secret" button.
3. In the "Choose secret type" page, select:

   Secret type: Other type of secret
   Key/value pairs:
   - Key: api_key
   - Value: Enter your API key value

   Click "Next"
4. In the "Configure secret" page:
   Secret name: Enter a name (e.g., "BedrockProxyAPIKey")
   Description: (Optional) Add a description of your secret
5. Click "Next" and review all your settings and click "Store"

After creation, you'll see your secret in the Secrets Manager console. Make note of the secret ARN.

**Step 2: Build and push container images to ECR**

1. Clone this repository:
   ```bash
   git clone https://github.com/aws-samples/bedrock-access-gateway.git
   cd bedrock-access-gateway
   ```

2. Run the build and push script:
   ```bash
   cd scripts
   bash ./push-to-ecr.sh
   ```

3. Follow the prompts to configure:
   - ECR repository names (or use defaults)
   - Image tag (or use default: `latest`)
   - AWS region (or use default: `us-east-1`)

4. The script will build and push both Lambda and ECS/Fargate images to your ECR repositories.

5. **Important**: Copy the image URIs displayed at the end of the script output. You'll need these in the next step.

**Step 3: Deploy the CloudFormation stack**

1. Download the CloudFormation template you want to use:
   - For API Gateway + Lambda: [`deployment/BedrockProxy.template`](deployment/BedrockProxy.template)
   - For ALB + Fargate: [`deployment/BedrockProxyFargate.template`](deployment/BedrockProxyFargate.template)

2. Sign in to AWS Management Console and navigate to the CloudFormation service in your target region.

3. Click "Create stack" → "With new resources (standard)".

4. Upload the template file you downloaded.

5. On the "Specify stack details" page, provide the following information:
   - **Stack name**: Enter a stack name (e.g., "BedrockProxyAPI")
   - **ApiKeySecretArn**: Enter the secret ARN from Step 1
   - **ContainerImageUri**: Enter the ECR image URI from Step 2 output
   - **DefaultModelId**: (Optional) Change the default model if needed

   Click "Next".

6. On the "Configure stack options" page, you can leave the default settings or customize them according to your needs. Click "Next".

7. On the "Review" page, review all details. Check the "I acknowledge that AWS CloudFormation might create IAM resources" checkbox at the bottom. Click "Submit".

That is it! 🎉 Once deployed, click the CloudFormation stack and go to **Outputs** tab, you can find the API Base URL from `APIBaseUrl`, the value should look like `http://xxxx.xxx.elb.amazonaws.com/api/v1`.

### Troubleshooting

If you encounter any issues, please check the [Troubleshooting Guide](./docs/Troubleshooting.md) for more details.

### SDK/API Usage

All you need is the API Key and the API Base URL. If you didn't set up your own key following Step 1, the application will fail to start with an error message indicating that the API Key is not configured.

Now, you can try out the proxy APIs. Let's say you want to test Claude 3 Sonnet model (model ID: `anthropic.claude-3-sonnet-20240229-v1:0`)...

**Example API Usage**

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
# For older versions
# https://github.com/openai/openai-python/issues/624
export OPENAI_API_BASE=<API base url>
```

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

**Example SDK Usage**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

Please check [Usage Guide](./docs/Usage.md) for more details about how to use embedding API, multimodal API and tool call.

### Responses API and the Codex CLI

`POST /responses` is served alongside `/chat/completions`, which is what clients built on the
newer OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses) need —
the Codex CLI among them, since it does not speak Chat Completions.

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    instructions="You are a helpful assistant.",
    input="Hello!",
)

print(response.output_text)
```

To point the Codex CLI at the gateway, add a provider to `~/.codex/config.toml`:

```toml
model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
model_provider = "bedrock-gateway"

[model_providers.bedrock-gateway]
name = "Bedrock Access Gateway"
base_url = "<API base url>"   # e.g. http://localhost:8000/api/v1
env_key = "BEDROCK_GATEWAY_API_KEY"
wire_api = "responses"
```

Then `export BEDROCK_GATEWAY_API_KEY=<API key>` and run `codex`. Shell commands, file edits and
reasoning summaries all work; Codex will warn that it has no metadata for the model, which is
harmless.

OpenAI GPT models on Bedrock work as well — set `model` to their inference-profile ID, e.g.
`global.openai.gpt-6-astra` or `global.openai.gpt-5.6-sol`, and optionally add
`model_reasoning_effort = "low" | "medium" | "high"`:

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

**Native passthrough for OpenAI GPT models.** Bedrock serves these models through its own
[Responses API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html)
on `bedrock-runtime`, so the gateway does not translate their requests. Any `/responses` request
whose model id (after the `gpt-*` → `DEFAULT_MODEL` alias) matches `*openai.gpt-*`, other than
`*gpt-oss*`, is forwarded as is to
`https://bedrock-runtime.<AWS_REGION>.amazonaws.com/openai/v1/responses`, signed with SigV4
using the gateway's own AWS credentials. Namespaces, `tool_search`, custom tools and reasoning
items therefore reach the model untouched. Streaming events are relayed byte for byte, and
upstream errors keep their status and body. The gateway API key is still checked and is never
forwarded. A few points to know:

- `store` defaults to `false` when the client leaves it unset, so Bedrock keeps no conversation
  data. An explicit `store: true` is honoured, and `previous_response_id` then works against the
  stored response. Only `POST /responses` is proxied, so `GET /responses/{id}` is not available.
- The IAM role needs `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream` on the
  account's default project (`arn:aws:bedrock:*:<account>:project/default`) as well as on the
  inference profile. Both CloudFormation templates include it; add it yourself if you run the
  gateway under your own role.
- Bedrock's endpoint has its own limits: the model must be a `us.` / `global.` (or `us-gov.`)
  cross-Region profile, not a foundation-model id or an application inference profile.
  `background: true` is rejected, and Guardrails do not apply.
- The endpoint runs no hosted tools and rejects a whole request that declares one, so the gateway
  drops them on this path too, with a log warning. Only `function`, `namespace`, `custom` and
  `tool_search` with `execution: "client"` tools are forwarded, unchanged. Everything else
  (`web_search`, which Codex sends by default, `file_search`, `mcp`, `code_interpreter`, hosted
  `tool_search`, ...) is removed. A `tool_choice` that names a removed tool falls back to `"auto"`. GPT OSS models do not support Responses on `bedrock-runtime`, which is
  why they are excluded and stay on the Converse path.

| Setting | Default | Purpose |
| --- | --- | --- |
| `RESPONSES_NATIVE_MODEL_PATTERNS` | `*openai.gpt-*` | Comma-separated, case-insensitive glob patterns of model ids to pass through. Set it to an empty string to send every model through Converse. |
| `RESPONSES_NATIVE_EXCLUDE_PATTERNS` | `*gpt-oss*` | Model ids that match these patterns stay on Converse even if they match the include patterns. |
| `BEDROCK_RUNTIME_RESPONSES_URL` | `https://bedrock-runtime.<AWS_REGION>.amazonaws.com/openai/v1/responses` | Overrides the upstream URL. |

If you turn the passthrough off, GPT-6 / GPT-5.x go through Converse like every other model. They
reject the `temperature` field and return their reasoning as encrypted `redactedContent`, so the
gateway drops `temperature` for them and skips the encrypted block.

All other models are translated to Bedrock Converse. What the translation covers and what it does not:

| Responses feature | Behaviour |
| --- | --- |
| Text, image and function-call input items, streaming, function tools, reasoning | Translated to Bedrock Converse. |
| `namespace` tool groups (how Codex sends MCP tools) | Members become Bedrock tools named `<namespace>__<name>`. If that name is invalid, longer than 64 characters or already taken, it is sanitised, truncated and given a short hash. Calls come back as `function_call` items with the original `namespace` and `name`. Replayed calls and outputs are mapped through the same names. `defer_loading` is ignored, so deferred tools are offered up front. |
| `tool_choice` | `"required"` becomes Bedrock `any`. A named function (with an optional `namespace`) or custom tool becomes Bedrock `tool`. If no supported tool is left, or the named tool is not in `tools`, the gateway returns 400 `invalid_request_error` (`param: "tool_choice"`) before calling Bedrock. `"none"` falls back to `"auto"`, because Bedrock's `toolChoice` has no equivalent. `allowed_tools` is ignored with a warning. |
| `tool_search` with `execution: "client"` | Exposed to the model as a `tool_search` function. The model's call is returned as a `tool_search_call` item, and the tools listed in a replayed `tool_search_output` are added to that request. Hosted (server) tool search is dropped with a warning. |
| `additional_tools` input items | Their tools are added to the request's tool list. |
| `custom` tools (e.g. Codex's grammar-based `exec`) | Sent as a function that takes a single string `input`. Calls come back as `custom_tool_call` items. The grammar is written into the tool description, because Bedrock cannot enforce it. |
| Hosted tools (`web_search`, `file_search`, ...) | Dropped with a log warning, since they have no Bedrock counterpart. The response's `tools` echoes only the declarations from the request's `tools` that took effect, and a namespace keeps only its accepted members. |
| `reasoning` | Emitted as reasoning items with summary text. Set `DEFAULT_MAX_TOKENS` if you need a budget other than 32,768 when a request omits `max_output_tokens`. |
| `store`, `previous_response_id` | Ignored. The gateway is stateless, so send the full conversation in `input` (which is what the Codex CLI does). |
| Input `reasoning` items | Dropped. Bedrock only accepts a reasoning block back with the signature it issued, and that signature has no place in the Responses wire format. |

Errors the gateway raises on `/responses` use the OpenAI shape
`{"error": {"message", "type", "param", "code"}}`, so the OpenAI SDKs and Codex can show them.
On the native passthrough path, Bedrock's own error status and body are returned unchanged.

To check that Codex really calls MCP tools through the gateway, an HTTP 200 or a tools list is not
enough: a dropped tool still gives a 200. Instead, run a prompt that needs one MCP tool. Then look
in the session rollout (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`) for an `item_completed`
event whose item has `"type":"McpToolCall"`, the expected `server` and `tool`, and
`"status":"completed"`. App Server clients receive the same item as `mcpToolCall`.

### Anthropic Messages API and Claude Code

`POST /messages` (plus `POST /messages/count_tokens`) speaks the
[Anthropic Messages API](https://docs.anthropic.com/en/api/messages), so the Anthropic SDKs and
Claude Code can use the gateway. It accepts the API key as `x-api-key` or as a bearer token.

Point Claude Code at the gateway root — the part before `/v1`, since Claude Code appends
`/v1/messages` itself:

```bash
export ANTHROPIC_BASE_URL=<API base url without /v1>   # e.g. http://localhost:8000/api
export ANTHROPIC_API_KEY=<API key>
claude
```

First-party model names such as `claude-sonnet-4-5` or `claude-opus-4-6` are mapped onto the
matching Bedrock inference profile (global first, then regional), so Claude Code works with its
defaults. You can also pick any Bedrock model, including non-Claude ones:

```bash
export ANTHROPIC_MODEL=global.openai.gpt-6-luna            # or qwen.qwen3-coder-480b-a35b-v1:0, ...
export ANTHROPIC_SMALL_FAST_MODEL=global.anthropic.claude-haiku-4-5-20251001-v1:0
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16000                 # if the model's output limit is below 32k
```

Claude Code prints a warning that a non-Claude model id is not in its model catalog; that is
harmless, but set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the model's real context window so
auto-compact kicks in at the right point.

What the translation covers and what it does not:

| Messages feature | Behaviour |
| --- | --- |
| Text, image, PDF/text document, `tool_use` / `tool_result` blocks, streaming, custom tools | Translated to Bedrock Converse. |
| `thinking` / `redacted_thinking` blocks | Kept with their signatures, so extended and interleaved thinking survive tool-use turns. Unsigned thinking (e.g. from DeepSeek or Qwen) is not sent back. |
| `cache_control` | Becomes a Converse `cachePoint` (including `ttl`) on models that support prompt caching; ignored elsewhere. |
| `thinking`, `output_config`, `context_management`, `top_k` | Passed to Claude as is; dropped for other models. |
| `anthropic-beta` header | Only the flags Bedrock accepts are forwarded to Claude (see `ANTHROPIC_BETA_ALLOWLIST`); the rest are dropped, since Bedrock rejects a whole request over one unknown flag. |
| Images (vision) | Kept for models with image input. Claude receives images inside `tool_result` as is; for other vision models (GPT, Qwen-VL, ...), which Bedrock only lets take images outside a tool result, they are moved right after it, so Claude Code screenshots and image reads still work. Text-only models get a 400. |
| Mid-conversation `system` messages | Folded into the adjacent user turn as text. |
| Server tools (`web_search`, `web_fetch`, `code_execution`, ...) | Dropped with a log warning, so Claude Code's WebSearch returns no results. |
| `tool_choice: {"type": "none"}` | Falls back to `auto`; Bedrock's `toolChoice` has no equivalent. |
| `count_tokens` | Uses Bedrock CountTokens where the model supports it and a tiktoken estimate otherwise. |

### Application Inference Profiles

This proxy now supports **Application Inference Profiles**, which allow you to track usage and costs for your model invocations. You can use application inference profiles created in your AWS account for cost tracking and monitoring purposes.

**Using Application Inference Profiles:**

```bash
# Use an application inference profile ARN as the model ID
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/your-profile-id",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

**SDK Usage with Application Inference Profiles:**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/your-profile-id",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message.content)
```

**Benefits of Application Inference Profiles:**
- **Cost Tracking**: Track usage and costs for specific applications or use cases
- **Usage Monitoring**: Monitor model invocation metrics through CloudWatch
- **Tag-based Cost Allocation**: Use AWS cost allocation tags for detailed billing analysis

For more information about creating and managing application inference profiles, see the [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html).

### Prompt Caching

This proxy now supports **Prompt Caching** for Claude and Nova models, which can reduce costs by up to 90% and latency by up to 85% for workloads with repeated prompts.

**Supported Models:**
- Claude models (Claude 3.5 Haiku, Claude 4, Claude 4.5, etc.)
- Nova models (Nova Micro, Nova Lite, Nova Pro, Nova Premier)

**Enabling Prompt Caching:**

You can enable prompt caching in two ways:

1. **Globally via Environment Variable** (set in ECS Task Definition or Lambda):
```bash
ENABLE_PROMPT_CACHING=true
```

2. **Per-request via `extra_body`** :

**Python SDK:**
```python
from openai import OpenAI

client = OpenAI()

# Cache system prompts
response = client.chat.completions.create(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[
        {"role": "system", "content": "You are an expert assistant with knowledge of..."},
        {"role": "user", "content": "Help me with this task"}
    ],
    extra_body={
        "prompt_caching": {"system": True}
    }
)

# Check cache hit
if response.usage.prompt_tokens_details:
    cached_tokens = response.usage.prompt_tokens_details.cached_tokens
    print(f"Cached tokens: {cached_tokens}")
```

**cURL:**
```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "messages": [
      {"role": "system", "content": "Long system prompt..."},
      {"role": "user", "content": "Question"}
    ],
    "extra_body": {
      "prompt_caching": {"system": true}
    }
  }'
```

**Cache Options:**
- `"prompt_caching": {"system": true}` - Cache system prompts
- `"prompt_caching": {"messages": true}` - Cache user messages
- `"prompt_caching": {"system": true, "messages": true}` - Cache both

**Requirements:**
- Prompt must be ≥1,024 tokens to enable caching
- Cache TTL is 5 minutes (resets on each cache hit)
- Nova models have a 20,000 token caching limit

For more information, see the [Amazon Bedrock Prompt Caching Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).

## Other Examples

### LangChain

Make sure you use `ChatOpenAI(...)` instead of `OpenAI(...)`

```python
# pip install langchain-openai
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_api_base=os.environ['OPENAI_BASE_URL'],
)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=chat)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = llm_chain.invoke(question)
print(response)

```

## FAQs

### About Privacy

This application does not collect any of your data. Furthermore, it does not log any requests or responses by default.

### Why choose API Gateway vs ALB?

**API Gateway + Lambda** uses [API Gateway response streaming](https://aws.amazon.com/blogs/compute/building-responsive-apis-with-amazon-api-gateway-response-streaming/) with [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) to support SSE streaming without requiring a VPC. This is a cost-effective, serverless option with up to 10 minutes timeout.

**ALB + Fargate** provides the lowest streaming latency with no cold starts, ideal for high-throughput workloads.

### Which regions are supported?

Generally speaking, all regions that Amazon Bedrock supports will also be supported, if not, please raise an issue in Github.

Note that not all models are available in those regions.

### Which models are supported?

You can use the [Models API](./docs/Usage.md#models-api) to get/refresh a list of supported models in the current region.

### Can I run this locally

Yes, you can run this locally, e.g. run below command under `src` folder:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API base url should look like `http://localhost:8000/api/v1`.

### Any performance sacrifice or latency increase by using the proxy APIs

Compared with direct AWS SDK calls, the proxy architecture will add some latency. The default API Gateway + Lambda deployment provides good streaming performance with Lambda response streaming.

For lowest latency on streaming responses, consider the ALB + Fargate deployment option which eliminates cold starts and provides consistent performance.

### Any plan to support SageMaker models?

Currently, there is no plan to support SageMaker models. This may change provided there's a demand from customers.

### Any plan to support Bedrock custom models?

Fine-tuned models and models with Provisioned Throughput are currently not supported. You can clone the repo and make the customization if needed.

### How to upgrade?

To use the latest features, you need follow the deployment guide to redeploy the application. You can upgrade the existing CloudFormation stack to get the latest changes.

## Security

See the [Security Guide](./docs/Security.md) for how to enable HTTPS on the ALB and how to configure the
controls around remote image URL fetching in the multimodal API. If you deployed before those controls
were added, read [If you deployed an earlier version](./docs/Security.md#if-you-deployed-an-earlier-version).

To report a security issue, see [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
