"""Forward Responses API traffic for OpenAI models to Bedrock's native Responses endpoint.

Bedrock serves OpenAI GPT models behind an OpenAI-compatible ``/openai/v1/responses`` route
on ``bedrock-runtime``. That route understands the whole Responses protocol (namespaces,
tool_search, custom tools, reasoning items), so for those models the gateway forwards the
client's request as-is instead of translating it onto Converse and losing what Converse
cannot express. The request is signed with the gateway's own AWS credentials; the client's
gateway API key is checked by the router and never forwarded.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-responses-api.html
"""

import fnmatch
import json
import logging
from typing import AsyncIterable

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.models.responses import _aiter
from api.schema import ResponsesRequest
from api.setting import (
    AWS_REGION,
    BEDROCK_RUNTIME_RESPONSES_URL,
    DEBUG,
    RESPONSES_NATIVE_EXCLUDE_PATTERNS,
    RESPONSES_NATIVE_MODEL_PATTERNS,
)

logger = logging.getLogger(__name__)

# Tool types Bedrock's native endpoint accepts. It runs no tools server-side, and a single
# hosted tool (web_search, file_search, code_interpreter, mcp, ...) makes it reject the whole
# request with a 400. Codex sends web_search by default, so hosted tools are dropped here, as
# on the Converse path. This is an allowlist so that hosted types added later are dropped too.
# tool_search is only kept with execution "client"; a hosted search is dropped.
NATIVE_TOOL_TYPES = frozenset({"function", "namespace", "custom", "tool_search"})

# Connect quickly, but leave room for a long reasoning turn before the first byte.
TIMEOUT = (10, 900)


def is_native_responses_model(
    model: str,
    patterns: tuple[str, ...] | None = None,
    exclude_patterns: tuple[str, ...] | None = None,
) -> bool:
    """Whether a model id is served by the native Responses endpoint. Matching ignores case.

    An empty pattern list disables the passthrough, sending every model through Converse.
    """
    patterns = RESPONSES_NATIVE_MODEL_PATTERNS if patterns is None else patterns
    exclude_patterns = RESPONSES_NATIVE_EXCLUDE_PATTERNS if exclude_patterns is None else exclude_patterns
    model = model.lower()
    if not any(fnmatch.fnmatchcase(model, p.lower()) for p in patterns):
        return False
    return not any(fnmatch.fnmatchcase(model, p.lower()) for p in exclude_patterns)


def build_payload(request: ResponsesRequest) -> dict:
    """The client's request as sent, minus gateway-only fields.

    Only fields the client set are forwarded, so the upstream defaults apply to everything
    else. ``store`` is the exception: the gateway is stateless and cannot serve a stored
    response back, so it defaults to False unless the client asked for something explicitly.
    """
    payload = request.model_dump(exclude_unset=True)
    payload.pop("extra_body", None)
    payload["model"] = request.model
    if "store" not in request.model_fields_set:
        payload["store"] = False
    _drop_hosted_tools(payload)
    return payload


def _is_native_tool(tool) -> bool:
    if not isinstance(tool, dict):
        return False
    tool_type = tool.get("type")
    if tool_type not in NATIVE_TOOL_TYPES:
        return False
    if tool_type == "tool_search":
        return tool.get("execution") == "client"
    return True


def _drop_hosted_tools(payload: dict) -> None:
    """Remove tools the native endpoint cannot run, leaving the supported ones untouched."""
    tools = payload.get("tools")
    if tools:
        kept, dropped = [], []
        for tool in tools:
            (kept if _is_native_tool(tool) else dropped).append(tool)
        for tool in dropped:
            tool_type = tool.get("type") if isinstance(tool, dict) else type(tool).__name__
            if tool_type == "tool_search":
                logger.warning(
                    "Dropping tool_search with execution %s, only client is supported", tool.get("execution")
                )
            else:
                logger.warning("Dropping hosted Responses tool of type %s, not supported by Bedrock", tool_type)
        if dropped:
            if kept:
                payload["tools"] = kept
            else:
                payload.pop("tools")

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and not _is_supported_tool_choice(tool_choice, payload.get("tools") or []):
        logger.warning("tool_choice %s names a dropped tool, falling back to auto", json.dumps(tool_choice))
        payload["tool_choice"] = "auto"


def _is_supported_tool_choice(tool_choice: dict, tools: list) -> bool:
    choice_type = tool_choice.get("type")
    if choice_type in ("function", "custom"):
        return True
    if choice_type == "allowed_tools":
        allowed = tool_choice.get("tools") or []
        return all(not isinstance(t, dict) or _is_supported_tool_choice(t, tools) for t in allowed)
    if choice_type == "tool_search":
        return any(t.get("type") == "tool_search" for t in tools)
    # Hosted tool choices ({"type": "web_search"}, {"type": "mcp", ...}, ...) name a tool
    # that is never forwarded.
    return False


class NativeResponsesProxy:
    """Signs and forwards a Responses request, relaying the upstream reply unchanged."""

    def __init__(self, url: str | None = None, region: str = AWS_REGION):
        self.url = (
            url
            or BEDROCK_RUNTIME_RESPONSES_URL
            or f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
        )
        self.region = region

    def _signed_headers(self, body: bytes, stream: bool) -> dict:
        headers = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        aws_request = AWSRequest(method="POST", url=self.url, data=body, headers=headers)
        # Read on every request so rotated credentials (Lambda, ECS task roles) are picked up.
        credentials = boto3.Session().get_credentials().get_frozen_credentials()
        SigV4Auth(credentials, "bedrock", self.region).add_auth(aws_request)
        return dict(aws_request.headers.items())

    def _post(self, body: bytes, stream: bool) -> requests.Response:
        headers = self._signed_headers(body, stream)
        return requests.post(self.url, data=body, headers=headers, stream=stream, timeout=TIMEOUT)

    async def forward(self, request: ResponsesRequest) -> Response:
        payload = build_payload(request)
        if DEBUG:
            logger.info("Native Responses request: " + json.dumps(payload))
        stream = bool(payload.get("stream"))
        body = json.dumps(payload).encode("utf-8")

        try:
            upstream = await run_in_threadpool(self._post, body, stream)
        except requests.RequestException as e:
            logger.error("Native Responses request for model %s failed: %s", request.model, str(e))
            raise HTTPException(status_code=502, detail=str(e))

        if not 200 <= upstream.status_code < 300:
            try:
                content = upstream.content
            finally:
                upstream.close()
            if DEBUG:
                logger.info("Native Responses error %s: %s", upstream.status_code, content[:2000])
            return Response(
                content=content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("Content-Type", "application/json"),
            )

        if stream:
            return StreamingResponse(content=self._relay(upstream), media_type="text/event-stream")

        try:
            content = upstream.content
        finally:
            upstream.close()
        try:
            return JSONResponse(content=json.loads(content), status_code=upstream.status_code)
        except ValueError:
            return Response(
                content=content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("Content-Type"),
            )

    @staticmethod
    async def _relay(upstream: requests.Response) -> AsyncIterable[bytes]:
        """Yield the upstream SSE bytes exactly as they arrive."""
        try:
            async for chunk in _aiter(upstream.iter_content(chunk_size=None)):
                if chunk:
                    yield chunk
        finally:
            upstream.close()
