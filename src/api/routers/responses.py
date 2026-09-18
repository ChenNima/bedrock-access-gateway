from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.responses import BedrockResponsesModel
from api.schema import Error, ResponsesRequest, ResponsesResponse
from api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/responses",
    dependencies=[Depends(api_key_auth)],
)


# Unlike the Chat Completions route this does not exclude unset fields: the Responses API
# spells out nulls (error, incomplete_details, usage) and clients read them.
@router.post("", response_model=ResponsesResponse | Error)
async def responses(
    responses_request: Annotated[
        ResponsesRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "instructions": "You are a helpful assistant.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hello!"}],
                        }
                    ],
                }
            ],
        ),
    ],
):
    if responses_request.model.lower().startswith("gpt-"):
        responses_request.model = DEFAULT_MODEL

    # Convert up front so a bad request still gets an HTTP error instead of a stream that
    # opens only to fail. Exception will be raised if model not supported.
    model = BedrockResponsesModel()
    chat_request = model.build_chat_request(responses_request)
    if responses_request.stream:
        return StreamingResponse(
            content=model.respond_stream(responses_request, chat_request),
            media_type="text/event-stream",
        )
    return await model.respond(responses_request, chat_request)
