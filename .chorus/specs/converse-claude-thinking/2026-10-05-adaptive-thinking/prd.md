---
title: "PRD: Adaptive thinking for Claude on the Converse path"
proposalUuid: 28ee490b-2df2-4b49-a77c-8a29f742951a
documentUuid: 45db91fc-9abc-4cf5-bb0a-9b6fff204106
---

# PRD: Adaptive thinking for Claude on the Converse path

## Problem
`BedrockModel._parse_request` (`src/api/models/bedrock.py`) sends every `anthropic.claude` model
`additionalModelRequestFields.reasoning_config = {"type": "enabled", "budget_tokens": N}` whenever
a request sets a reasoning effort. Claude Opus 4.7 / 4.8, Opus 5 / 5.5, Sonnet 5 / 5.5 and
Fable 5 / 5.1 reject that shape with a 400:

```
ValidationException ... ConverseStream: "thinking.type.enabled" is not supported for this model.
Use "thinking.type.adaptive" and "output_config.effort"
```

Chat Completions (`reasoning_effort`) and Responses (`reasoning.effort`, through
`REASONING_EFFORT_MAP`) share this code, so both endpoints fail. Codex sets a reasoning effort by
default, and `global.anthropic.claude-opus-5` is the Tokyo stack's `DefaultModelId` (also the
target of the `gpt-*` alias), so the default configuration fails. Streaming failures are logged
as HTTP 200, and the error only appears in the SSE `response.failed` event.

Field evidence:
- Tokyo acceptance of task bcaf31dc (2026-10-03): Codex 0.160.0 + claude-opus-5 failed until
  `model_reasoning_effort = "none"` was set.
- Tokyo gateway log, 2026-10-04 02:07 UTC: the pi agent on `global.anthropic.claude-sonnet-5-5`
  failed the same way, while `/v1/messages` succeeded because the Messages route forwards
  `thinking` / `output_config` unchanged.

Two follow-on failures would appear once the thinking shape is fixed:
1. The new models default `thinking.display` to `omitted`, which leaves the thinking text empty.
   Chat `reasoning_content` and the Responses reasoning summary would always be empty.
2. Opus 4.7+, Sonnet 5 and Fable return 400 on `temperature` / `top_p`, and Sonnet 5.5 returns
   400 on non-default values. Clients such as Codex may send them.

## Users / scenarios
1. Codex 0.160.0 → `/v1/responses` → claude-opus-5 (Tokyo default model) with the default
   reasoning effort. It must make native `mcp_tool_call`s.
2. The pi agent and other Responses clients → claude-sonnet-5-5 with a reasoning effort.
3. OpenAI SDK Chat Completions clients setting `reasoning_effort` on any current Claude model.
4. Existing users of Claude 3.7 / 4.x / 4.5 with `reasoning_effort` keep today's behaviour.

## Decisions (from elaboration rounds 1–2, answered by the requester)
| # | Decision |
|---|----------|
| q1 | Reverse detection: every Claude model defaults to adaptive. Only a known legacy list keeps enabled + budget. |
| r2q1 | The legacy list has a built-in default and can be replaced by the env var `BUDGET_THINKING_MODEL_PATTERNS` (fnmatch globs, same style as `RESPONSES_NATIVE_MODEL_PATTERNS`). |
| q2 | Opus 4.6 / Sonnet 4.6 switch to adaptive as well (they are not on the legacy list). |
| q3 | Effort low/medium/high maps 1:1; Responses `minimal` → low. No public schema change. |
| q4 | Responses `none` keeps today's behaviour: no thinking or effort is sent, and the model uses its default. |
| q5 | Always send `display: "summarized"` so reasoning text is returned. |
| q6 / r2q2 | Silently drop `temperature` and `topP` for every Claude model outside the legacy list and outside Opus 4.6 / Sonnet 4.6, whether or not reasoning is on. |
| q7 | Adaptive models no longer require max_tokens when reasoning is on. |

## Acceptance criteria
- Chat Completions `reasoning_effort=low|medium|high` and Responses `reasoning.effort` on
  claude-opus-5 / claude-sonnet-5-5 no longer error, and the response contains reasoning content.
- Legacy models (e.g. Claude Sonnet 4.5, Haiku 4.5, 3.7 Sonnet) still receive
  `reasoning_config: enabled + budget_tokens`, and every existing test passes.
- Codex 0.160.0, **without** `model_reasoning_effort = "none"`, completes a native `mcp_tool_call`
  on claude-opus-5 through the Tokyo gateway.

## Out of scope
- `xhigh` / `max` effort, explicit thinking disable, and the Anthropic Messages route.
- Prefill handling for the new models (`NO_ASSISTANT_PREFILL_MODELS` only lists Opus 4.6). This
  is a separate follow-up if it shows up.
