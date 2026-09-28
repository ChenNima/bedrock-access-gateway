from typing import Annotated

from fastapi import APIRouter, Body, Depends, Header
from fastapi.responses import StreamingResponse

from api.auth import anthropic_api_key_auth
from api.models.messages import BedrockMessagesModel, parse_betas
from api.schema import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)

# Errors on these routes are rendered in the Anthropic format, see app.py.
router = APIRouter(
    prefix="/messages",
    dependencies=[Depends(anthropic_api_key_auth)],
)


@router.post("", response_model=AnthropicMessagesResponse)
async def messages(
    messages_request: Annotated[
        AnthropicMessagesRequest,
        Body(
            examples=[
                {
                    "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "max_tokens": 1024,
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello!"}],
                }
            ],
        ),
    ],
    anthropic_beta: Annotated[str | None, Header()] = None,
):
    # Convert and open the Bedrock call up front, so a bad request or a throttled call
    # still gets an HTTP error instead of a stream that opens only to fail.
    model = BedrockMessagesModel()
    args = model.build_converse_args(messages_request, parse_betas(anthropic_beta))
    if messages_request.stream:
        raw = await model.invoke(args, stream=True)
        return StreamingResponse(
            content=model.respond_stream(messages_request, raw),
            media_type="text/event-stream",
        )
    return await model.respond(messages_request, args)


@router.post("/count_tokens", response_model=AnthropicCountTokensResponse)
async def count_tokens(
    count_request: AnthropicCountTokensRequest,
    anthropic_beta: Annotated[str | None, Header()] = None,
):
    model = BedrockMessagesModel()
    return AnthropicCountTokensResponse(
        input_tokens=await model.count_tokens(count_request, parse_betas(anthropic_beta))
    )
