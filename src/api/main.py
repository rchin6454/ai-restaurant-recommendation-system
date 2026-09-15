"""FastAPI app: lifespan, error envelope, request logging (architecture §7; plan 4.1, 4.4).

    uvicorn src.api.main:app --reload

The catalog loads in the lifespan hook, before the server accepts traffic (A-02). A missing or
outdated Parquet stops startup with the ingest command in the message (A-01) rather than failing
every request. Each worker process loads its own copy of the catalog, so keep `--workers` low (A-10).

Every non-2xx response has one shape (the §7 error envelope), and no stack trace ever reaches a client:

    {"error": {"code": "invalid_request", "message": "...", "details": [{"field": "min_rating", "message": "..."}],
               "request_id": "3f9c..."}}
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import CatalogInfo, load_catalog_info, router
from src.config import get_logger
from src.data.catalog import CatalogError

logger = get_logger(__name__)

# A maximal valid body (500-char free text, 20 cuisines) is a few KB (A-12). Enforced on the declared
# Content-Length; a chunked body without one is not capped here.
MAX_BODY_BYTES = 16 * 1024
_CLIENT_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "invalid_request",
    500: "internal_error",
    503: "unavailable",
}


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


def error_response(
    status: int, message: str, request_id: str | None, *, details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(code=_CODES.get(status, "error"), message=message, details=details or [], request_id=request_id)
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def create_app(catalog_loader: Callable[[], CatalogInfo] = load_catalog_info) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.catalog_info = catalog_loader()
        except CatalogError as exc:  # A-01: fail fast, naming the path and the ingest command
            logger.critical("startup aborted: catalog unavailable", extra={"reason": str(exc)})
            raise
        logger.info("api ready", extra={"rows": app.state.catalog_info.rows})
        yield
        app.state.catalog_info = None

    app = FastAPI(
        title="Bengaluru Restaurant Recommender",
        version="0.1.0",
        lifespan=lifespan,
        responses={422: {"model": ErrorEnvelope}, 500: {"model": ErrorEnvelope}, 503: {"model": ErrorEnvelope}},
    )
    app.state.catalog_info = None
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _CLIENT_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()

        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            response = error_response(413, f"Request body is larger than {MAX_BODY_BYTES:,} bytes.", request_id)
        else:
            try:
                response = await call_next(request)
            except Exception:  # A-04: full detail to the logs, a generic envelope to the client
                logger.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
                response = error_response(500, "Something went wrong on our side. Please try again.", request_id)

        response.headers["X-Request-ID"] = request_id
        # Method, path and status only: bodies carry user free text and aren't logged.
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:  # A-03
        details = [
            ErrorDetail(field=".".join(str(p) for p in err["loc"] if p != "body") or "body", message=err["msg"])
            for err in exc.errors()
        ]
        return error_response(422, "The request is invalid. See `details`.", _request_id(request), details=details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
        return error_response(exc.status_code, message, _request_id(request), headers=exc.headers)

    @app.exception_handler(CatalogError)
    async def catalog_error(request: Request, exc: CatalogError) -> JSONResponse:
        logger.error("catalog error during request", extra={"request_id": _request_id(request), "reason": str(exc)})
        return error_response(503, "The restaurant catalog is unavailable.", _request_id(request))

    return app


app = create_app()
