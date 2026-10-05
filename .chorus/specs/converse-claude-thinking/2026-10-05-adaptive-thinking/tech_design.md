---
title: "Tech Design: Adaptive thinking for Claude on the Converse path"
proposalUuid: 28ee490b-2df2-4b49-a77c-8a29f742951a
documentUuid: 6408b4bc-017c-480d-aefc-e38f35a7124a
---

# Tech Design: Adaptive thinking for Claude on the Converse path

## Model classification
All checks run on `model_lower = self._resolve_to_foundation_model(chat_request.model).lower()`,
which `_parse_request` already computes. A profile id that cannot be resolved (e.g.
`global.anthropic.claude-opus-5` without metadata) still contains `anthropic.claude`, and the
globs below carry a leading `*` so they match through region prefixes.

`src/api/setting.py`:

```python
# Claude models that predate adaptive thinking and still take reasoning_config with a
# budget_tokens. Every other Claude model gets adaptive thinking plus output_config.effort.
BUDGET_THINKING_MODEL_PATTERNS = parse_patterns(os.environ.get(
    "BUDGET_THINKING_MODEL_PATTERNS",
    "*anthropic.claude-v2*,*anthropic.claude-instant*,*anthropic.claude-3-*,"
    "*anthropic.claude-opus-4-2025*,*anthropic.claude-opus-4-1*,*anthropic.claude-opus-4-5*,"
    "*anthropic.claude-sonnet-4-2025*,*anthropic.claude-sonnet-4-5*,*anthropic.claude-haiku-4-5*",
))
```

Notes on the globs:
- `claude-opus-4-2025*` / `claude-sonnet-4-2025*` match the dated Claude 4.0 ids
  (`anthropic.claude-opus-4-20250514-v1:0`) without also matching `claude-opus-4-6` / `4-7` / `4-8`.
- `claude-3-*` covers 3 Haiku/Sonnet/Opus, 3.5 and 3.7 (`claude-3-5-…`, `claude-3-7-…`).
- Setting the variable replaces the whole default list. An empty value means no legacy models.

`src/api/models/bedrock.py` gets two module-level helpers (or private methods):

```python
def _uses_budget_thinking(model_lower) -> bool:
    return any(fnmatch(model_lower, p.lower()) for p in BUDGET_THINKING_MODEL_PATTERNS)

# Adaptive Claude models that still accept temperature / top_p.
SAMPLING_SUPPORTED_ADAPTIVE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

def _rejects_sampling_params(model_lower) -> bool:
    return (
        "anthropic.claude" in model_lower
        and not _uses_budget_thinking(model_lower)
        and not any(m in model_lower for m in SAMPLING_SUPPORTED_ADAPTIVE_MODELS)
    )
```

Use the matching helper that already exists for `RESPONSES_NATIVE_MODEL_PATTERNS` in
`responses.py` / `setting.py` if one does; otherwise use `fnmatch.fnmatchcase` on lowercased
values.

## Request building (`_parse_request`)
1. **Sampling.** Next to the existing `TEMPERATURE_UNSUPPORTED_MODELS` block: if
   `_rejects_sampling_params(model_lower)`, pop both `temperature` and `topP` from
   `inference_config` (DEBUG log, as today). This runs regardless of `reasoning_effort`.
2. **Reasoning, Claude branch:**
   - `_uses_budget_thinking` → existing behaviour unchanged (max_tokens required, topP popped,
     `reasoning_config` enabled + `_calc_budget_tokens`).
   - otherwise →
     ```python
     args["additionalModelRequestFields"] = {
         "thinking": {"type": "adaptive", "display": "summarized"},
         "output_config": {"effort": chat_request.reasoning_effort},
     }
     ```
     There is no max_tokens check and no budget. `maxTokens` is still sent when the client or
     the Responses fallback (`DEFAULT_MAX_TOKENS`) provides it.
     The branch also pops `topP` from `inference_config`, as the legacy branch and the
     `extra_body` thinking path already do: thinking on Claude does not take `top_p`. This
     matters for Opus 4.6 / Sonnet 4.6, which are exempt from step 1 and would otherwise send
     `topP` alongside adaptive thinking. `temperature` is left as is for those models,
     matching today's legacy behaviour.
3. The `extra_body` merge below stays as is. A client-supplied `thinking` / `output_config` in
   `extra_body` overrides the derived one, which is the existing precedence.
4. DeepSeek and other branches are untouched.

## Responses path
`REASONING_EFFORT_MAP` (minimal→low, low, medium, high) and `none` → no effort stay unchanged.
The `DEFAULT_MAX_TOKENS` fallback for reasoning stays: it is harmless for adaptive models and is
still required for legacy ones. Update the comment so it no longer claims Claude always needs it.

## Response / streaming
With `display: "summarized"`, Converse returns `reasoningContent.reasoningText.text` (and
`delta.reasoningContent.text` when streaming) in the same shape as today, so
`_create_response`, the streaming chunk builder and the Responses reasoning-item emitter need no
change. The tests should assert this by feeding a Converse-shaped response.

## Tests (`test/test_reasoning_config.py`, new)
Build `BedrockModel()._parse_request(ChatRequest(...))` for model ids, with
`_resolve_to_foundation_model` either left as pass-through or patched via `profile_metadata`:
- `global.anthropic.claude-opus-5`, `anthropic.claude-sonnet-5-5`, `anthropic.claude-opus-4-6-v1`
  → adaptive `thinking` + `output_config.effort` equal to the requested effort, no
  `reasoning_config`, and no 400 without max_tokens.
- `anthropic.claude-sonnet-4-5-20250929-v1:0`, `anthropic.claude-haiku-4-5-20251001-v1:0`,
  `anthropic.claude-3-7-sonnet-20250219-v1:0`, `anthropic.claude-opus-4-20250514-v1:0`
  → `reasoning_config` enabled + budget, and still 400 without max_tokens.
- Sampling: temperature/top_p are dropped for opus-5 / sonnet-5-5 (with and without reasoning),
  kept for opus-4-6 / sonnet-4-5 when reasoning is off, and the topP conflict rule still applies
  to 4.5. With reasoning on, opus-4-6 / sonnet-4-6 keep temperature but lose topP.
- Response shape: a Converse response (non-streaming `reasoningContent.reasoningText.text`
  and streaming `delta.reasoningContent.text`) still yields Chat `reasoning_content` and a
  Responses reasoning item with summary text.
- `BUDGET_THINKING_MODEL_PATTERNS` override (monkeypatch the module attribute) moves a model
  between modes.
- A profile id resolved through `profile_metadata` to a 5.x foundation id gets adaptive.
- Responses: `reasoning.effort="minimal"` → `output_config.effort == "low"` for opus-5, and
  `"none"` → no `additionalModelRequestFields` thinking.
- Existing tests (`test_reasoning_budget.py`, `test_responses*.py`, ...) keep passing.

## Docs
- README Responses section and `docs/Usage.md` / `docs/Usage_CN.md` reasoning section: describe
  adaptive vs budget behaviour and document `BUDGET_THINKING_MODEL_PATTERNS`.
- Where the docs describe Codex with Claude models, state that the default reasoning effort
  works. The docs do not currently advise `model_reasoning_effort = "none"`, so there is
  nothing to remove.

## Deployment acceptance (Tokyo)
Build and push the image, update the Tokyo `BedrockProxyAPI` stack/Lambda the same way the
parent idea's acceptance (task bcaf31dc) did, then run Codex 0.160.0 with the default reasoning
settings against claude-opus-5 and complete a native `mcp_tool_call`. Send live Chat Completions
and Responses reasoning requests to both claude-opus-5 and claude-sonnet-5-5. This also confirms
on Bedrock that `display: "summarized"` returns reasoning text. Check the Lambda logs for
the absence of the `thinking.type.enabled` ValidationException.
