---
slug: converse-claude-thinking
title: Claude reasoning mapping on the Converse translation path
status: active
created: 2026-10-05
---

## Intent
Clients that speak OpenAI Chat Completions (`reasoning_effort`) or the Responses API
(`reasoning.effort`) can turn on Claude reasoning through the gateway. The gateway translates the
effort into whatever thinking configuration the target Claude model accepts on Bedrock Converse,
so a client never has to know which generation of Claude it is talking to.

## Requirements
Claude models default to adaptive thinking: `thinking: {type: "adaptive", display: "summarized"}`
plus `output_config: {effort}`. Only a known list of legacy models (Claude 3.x, 3.7 Sonnet,
Opus 4 / 4.1 / 4.5, Sonnet 4 / 4.5, Haiku 4.5) keep `reasoning_config: {type: "enabled",
budget_tokens}`, with the budget derived from max_tokens. The legacy list has a built-in
default and can be replaced through `BUDGET_THINKING_MODEL_PATTERNS`. Matching runs on the
foundation model id that inference profiles resolve to.

Effort maps one-to-one for low / medium / high. Responses `minimal` maps to low, and `none` sends
no thinking configuration at all. Adaptive models do not need max_tokens.

Claude models outside the legacy list and outside Opus 4.6 / Sonnet 4.6 reject sampling
parameters, so the gateway drops `temperature` and `topP` for them whether or not reasoning is on.

- [ ] claude-opus-5 / claude-sonnet-5-5 accept reasoning effort on Chat Completions and Responses and return reasoning text
- [ ] legacy Claude models keep enabled + budget_tokens unchanged
- [ ] temperature / topP never reach a Claude model that rejects them
- [ ] Codex 0.160.0 with default reasoning completes a native mcp_tool_call through the Tokyo gateway on claude-opus-5

## Non-goals
- Exposing `xhigh` / `max` effort levels.
- Explicitly disabling thinking (`disabled` / `between_tools`) for `none`.
- The Anthropic Messages route, which already forwards `thinking` / `output_config` unchanged.
