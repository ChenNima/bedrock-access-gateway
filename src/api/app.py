import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from mangum import Mangum
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers import chat, embeddings, messages, model, responses
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, TITLE, VERSION

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
app = FastAPI(**config)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
origins_list = [origin.strip() for origin in allowed_origins.split(",")] if allowed_origins != "*" else ["*"]

# Warn if CORS allows all origins
if origins_list == ["*"]:
    logging.warning("CORS is configured to allow all origins (*). Set ALLOWED_ORIGINS environment variable to restrict access.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,  # nosec - configurable via ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(model.router, prefix=API_ROUTE_PREFIX)
app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)
app.include_router(responses.router, prefix=API_ROUTE_PREFIX)
app.include_router(messages.router, prefix=API_ROUTE_PREFIX)

# Anthropic SDKs (and so Claude Code) read errors from {"type": "error", "error": {...}}
# and pick their retry behaviour from the status code and error type.
ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def is_anthropic_route(request) -> bool:
    return request.url.path.startswith(f"{API_ROUTE_PREFIX}/messages")


def anthropic_error(status_code: int, message: str) -> JSONResponse:
    # Bedrock words a context overflow differently; Claude Code looks for this phrase to
    # compact the conversation and retry.
    if status_code == 400 and "input is too long" in message.lower():
        message = f"prompt is too long: {message}"
    error_type = ANTHROPIC_ERROR_TYPES.get(status_code, "api_error")
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}


@app.exception_handler(StarletteHTTPException)
async def anthropic_http_exception_handler(request, exc):
    if is_anthropic_route(request):
        return anthropic_error(exc.status_code, str(exc.detail))
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger = logging.getLogger(__name__)
    
    # Log essential info only - avoid sensitive data and performance overhead
    logger.warning(
        "Request validation failed: %s %s - %s", 
        request.method, 
        request.url.path,
        str(exc).split('\n')[0]  # First line only
    )
    
    if is_anthropic_route(request):
        return anthropic_error(400, str(exc))
    return PlainTextResponse(str(exc), status_code=400)


handler = Mangum(app)

if __name__ == "__main__":
    # Bind to 0.0.0.0 for container environments, network is handled by network policies and load balancers
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)  # nosec B104
