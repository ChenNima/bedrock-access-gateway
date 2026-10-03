---
slug: responses-tool-protocol
title: Responses API tool protocol (namespace, tool_search, native passthrough)
status: active
created: 2026-10-03
---

## Intent
Make the gateway's OpenAI Responses endpoint (`/v1/responses`) carry the full tool protocol
that agent clients such as Codex CLI rely on. Namespace tool groups, tool_search with deferred
loading, additional_tools items, custom (grammar) tools, and forced tool_choice must reach the
model and come back intact. Nothing should be dropped silently.

## Requirements
Requests for OpenAI GPT models on Bedrock (model id matching `*openai.gpt-*`, excluding gpt-oss)
go to Bedrock's native Responses endpoint (`bedrock-runtime.<region>.amazonaws.com/openai/v1/responses`)
unchanged and signed with SigV4. The patterns can be overridden through environment variables.
The gateway adds `store=false` when the client did not set `store`. Streaming responses are
relayed byte-for-byte. Upstream error statuses and bodies are returned unchanged.

All other models keep the Responses → Converse translation, which now:
- flattens namespace members into Bedrock tool names `<namespace>__<name>`, using a sanitised,
  truncated name with a short hash when the plain form is invalid, longer than 64 characters,
  or collides with another tool;
- returns `function_call` items with the original `namespace` and `name`, both streaming and
  non-streaming, and maps replayed `function_call` / `function_call_output` items back through
  the same mapping;
- maps `tool_choice: "required"` to Bedrock `any` and a named function (optionally with a
  namespace) to Bedrock `tool`; returns 400 `invalid_request_error` when no usable tool is left
  or the named tool does not exist;
- keeps dropping hosted tools (web_search, …) with a warning, and echoes only the tools that
  actually took effect in the response `tools`;
- supports client-executed `tool_search`: it is exposed to Bedrock as a function, the model's
  call is emitted as `tool_search_call` (`execution: "client"`), and later `tool_search_output`
  items become tool results whose loaded tools join the tool config;
- expands `additional_tools` input items into the tool list;
- maps `custom` tools (for example Codex's lark-grammar `exec`) to a function with a single
  string `input`, emitted as `custom_tool_call` and replayed from `custom_tool_call(_output)`.

- [ ] Codex 0.160.0, connected through the deployed Tokyo gateway with `global.openai.gpt-6-astra`, gets a successful native `mcpToolCall` on a Chorus MCP tool
- [ ] Codex 0.160.0, connected through the gateway with a Claude model (Converse path), gets a successful native `mcpToolCall` on a Chorus MCP tool
- [ ] Namespace-only requests with `tool_choice: "required"` produce a namespaced function_call, or a clear 400, never a silent plain answer
- [ ] README / Usage / Usage_CN document the supported protocol surface and its limits

## Non-goals
- Hosted (server-executed) tool_search, web_search, file_search or other OpenAI hosted tools on the Converse path
- Bedrock Mantle as an upstream (no GPT-5/6 in Tokyo, no `global.*` profiles)
- Server-side state: `previous_response_id` keeps working only on the native passthrough path
