---
title: "PRD: Responses namespace / tool_search support and native passthrough"
proposalUuid: 06fd6331-0217-4263-bf85-083682ad40f0
documentUuid: 6523e31c-be34-422c-b8f7-b37d6a389314
---

# PRD: Responses namespace / tool_search support and native passthrough

## Problem
Codex CLI 0.160.0 always sends MCP tools (for example the 83 Chorus tools) to `/v1/responses`
as `namespace` groups. The gateway's `_convert_tools()` (`src/api/models/responses.py`) keeps
only top-level `type: "function"` tools and drops whole namespace groups. As a result:

- Bedrock never sees the Chorus tools.
- With `tool_choice: "required"` and only namespace tools, no `toolConfig` is built, so the
  required constraint is lost too.
- The response echoes the original tools and returns 200/completed, so the client cannot tell
  anything went wrong.

The response schema also has no `namespace` field, so even a successful call could not be
routed back to the MCP tool: Codex resolves a call by `(namespace, name)` and treats a missing
namespace as `functions`. `additional_tools` input items (Codex responses_lite / code-mode),
`tool_search` and `custom` tools have no handling either.

Production evidence: the Tokyo gateway logged 692 namespace-drop warnings between 14:10 and
14:43 UTC on 2026-10-03, and Leo's native Codex sessions had no Chorus business tools.

## Users / scenarios
1. **Codex + OpenAI GPT model** (Leo: `global.openai.gpt-6-astra`, Tokyo). Must make native
   MCP calls, including the tool_search and responses_lite modes Codex enables for catalog slugs.
2. **Codex + Claude or another Converse model.** Must make native MCP calls through the
   translated path.
3. **Other OpenAI Responses SDK clients** using plain function tools. Must keep working unchanged.

## Decisions (from elaboration rounds 1–2, answered by the requester)
| # | Decision |
|---|----------|
| Route | Two tracks: native passthrough for OpenAI GPT models, translation fix for everything else (R2-Q1=a) |
| Passthrough selection | Model id glob rule `*openai.gpt-*` excluding `*gpt-oss*`, overridable by env (R2-Q2=a) |
| store | Honour an explicit client value; default to `false` when unset (R2-Q3=b) |
| Name mapping | `<namespace>__<name>`; truncate and add a short hash when invalid, too long or colliding; derived deterministically per request, no gateway state (Q1=a) |
| Output identity | `function_call` carries the OpenAI `namespace` field plus the original `name` (Q2=a) |
| Forced choice without a tool | HTTP 400 `invalid_request_error` before inference (Q3=a) |
| Hosted tools | Keep dropping with a warning; the response `tools` echo lists only effective tools (Q5=a) |
| tool_search / additional_tools | Full client-mode tool_search plus additional_tools expansion on the Converse path (R2-Q4=c) |
| Custom (grammar) tools | Map to a single-string-`input` function and round-trip as `custom_tool_call` (stated assumption in the confirmation comment; YOLO run requested afterwards without objection) |
| Deployment | The agent edits both CloudFormation templates (IAM for the native endpoint), builds and pushes the image, and updates the Tokyo stack (Q6=a, R2-Q5=a) |
| Acceptance | The agent verifies a native `mcpToolCall` from local Codex 0.160.0 on both tracks; a human confirms Leo's fresh session (Q6=a) |

## Functional requirements
1. **Native passthrough.**
   - Requests whose (post-alias) model matches the passthrough rule are SigV4-signed with the
     gateway's AWS credentials and forwarded to `https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1/responses`.
   - All client fields are preserved, except that `store` defaults to `false` when absent and
     the gateway-only `extra_body` is not forwarded.
   - Non-streaming: return the upstream status and JSON body.
   - Streaming: relay the SSE bytes as they arrive.
   - Upstream errors keep their HTTP status and body.
   - Gateway API-key auth still applies.
2. **Namespace translation (Converse path).** Flatten namespace members, resolve
   `tool_choice`, emit and replay namespaced calls, keep `call_id` correlation, and return 400
   for a forced choice that has nothing to force, using an OpenAI-style error body
   (`{"error":{"message","type":"invalid_request_error","param","code"}}`).
3. **Effective tools echo.** The response `tools` lists only the tools that were actually sent
   to the model. Hosted tools are dropped with a warning.
4. **tool_search (client mode).**
   - A `{"type":"tool_search","execution":"client",...}` tool becomes a Bedrock function.
   - A model call to it is emitted as `tool_search_call` with `execution: "client"`, an object
     `arguments`, and the original `call_id`.
   - Replayed `tool_search_call` / `tool_search_output` items become toolUse / toolResult, and
     the tools in `tool_search_output.tools` join the request's tool list.
5. **additional_tools.** `{"type":"additional_tools","tools":[...]}` input items add their tools
   (function, namespace, custom, tool_search) to the tool list. They are never silently dropped.
6. **Custom tools.** `{"type":"custom","name":...}` becomes a function with
   `{"input": string}`. Calls are emitted as `custom_tool_call` (`input`), and replayed
   `custom_tool_call` / `custom_tool_call_output` items are mapped back.
7. **Docs.** README.md, docs/Usage.md and docs/Usage_CN.md describe:
   - which models use passthrough and how to override the rule;
   - the protocol coverage on the Converse path;
   - the remaining limits;
   - the IAM permission the native endpoint needs;
   - how to verify with Codex.

## Non-functional
- Stateless: every mapping is rebuilt from the current request.
- No new runtime dependencies beyond what `src/requirements.txt` already pins (boto3/botocore, requests).
- Existing tests keep passing. New unit tests cover every requirement offline.

## Acceptance criteria
- Plain top-level function calls still pass on the Converse path.
- On the Converse path, a namespace-only request with `required` returns a function_call with
  the correct `namespace`, `name` and `call_id`.
- The name-mapping edge cases are covered: the same name in two namespaces, a name over 64
  characters, invalid characters, and mixed top-level and namespace tools. A named
  `tool_choice` selects the correct Bedrock tool.
- The full streaming event sequence and the non-streaming output both carry the namespace.
  Replaying a call with its output plus a follow-up response works end to end.
- `required` or a named choice with no effective tool returns 400. additional_tools and
  tool_search round-trip in regression tests.
- After the Tokyo deploy, local Codex 0.160.0 produces a successful native `mcpToolCall` for
  `chorus_checkin`:
  - via `global.openai.gpt-6-astra` (passthrough);
  - via a Claude model (Converse).

  HTTP 200, the checkin banner, tools/list, or a shell / CLI / HTTP substitute do not count as passing.
- Leo's fresh Codex session is confirmed by a human. This is tracked as a follow-up and is not
  blocking agent verification.

## Out of scope
Hosted tool_search and other hosted OpenAI tools on the Converse path, Bedrock Mantle, and
server-side state for the Converse path.
