"""FastAPI app: lifespan, error envelope, request logging, rate limiting (architecture §7; plan 4.1, 4.4, 5.2).

    uvicorn src.api.main:app --reload

The catalog loads in the lifespan hook, before the server accepts traffic (A-02). A missing or
outdated Parquet stops startup with the ingest command in the message (A-01) rather than failing
every request. Each worker process loads its own copy of the catalog, so keep `--workers` low (A-10).

Every non-2xx response has one shape (the §7 error envelope), and no stack trace ever reaches a client:

    {"error": {"code": "invalid_request", "message": "...", "details": [{"field": "min_rating", "message": "..."}],
               "request_id": "3f9c...", "retry_after_s": null}}
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import CatalogInfo, load_catalog_info, router
from src.config import get_logger, settings
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
    429: "rate_limited",
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
    retry_after_s: int | None = None  # set on 429 (A-07)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


def error_response(
    status: int, message: str, request_id: str | None, *, details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None, retry_after_s: int | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(code=_CODES.get(status, "error"), message=message, details=details or [],
                        request_id=request_id, retry_after_s=retry_after_s)
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


class ClientRateLimiter:
    """Sliding one-minute window of `POST /recommend` calls per client address (plan 5.2; A-06, A-07, S-04).

    Any call can spend Groq tokens, and the account fits only about one LLM call a minute, so one
    client in a loop must not starve everyone else. Keyed on the socket peer address: behind a
    reverse proxy every user shares that address, so size the limit for that, or key on a trusted
    forwarded header instead. Repeat queries are cheap anyway (5.1 cache) but still count here.
    """

    WINDOW_S = 60.0

    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic, max_clients: int = 10_000) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._calls: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, client: str) -> float:
        """Record a call and return 0, or return the seconds until `client` may call again (nothing recorded)."""
        with self._lock:
            now = self._clock()
            calls = self._calls.setdefault(client, deque())
            while calls and calls[0] <= now - self.WINDOW_S:
                calls.popleft()
            self._calls.move_to_end(client)
            if len(calls) >= self.per_minute:
                return calls[0] + self.WINDOW_S - now
            calls.append(now)
            while len(self._calls) > self._max_clients:  # memory bound: forget the least recently seen client
                self._calls.popitem(last=False)
            return 0.0


def create_app(
    catalog_loader: Callable[[], CatalogInfo] = load_catalog_info,
    *,
    requests_per_minute: int | None = None,
) -> FastAPI:
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
        responses={
            422: {"model": ErrorEnvelope},
            429: {"model": ErrorEnvelope},
            500: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )
    app.state.catalog_info = None
    app.state.rate_limiter = ClientRateLimiter(requests_per_minute or settings.api_requests_per_minute)
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _CLIENT_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()

        declared = request.headers.get("content-length", "")
        wait = 0.0
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            response = error_response(413, f"Request body is larger than {MAX_BODY_BYTES:,} bytes.", request_id)
        elif request.method == "POST" and request.url.path == "/recommend" and (
            wait := app.state.rate_limiter.check(request.client.host if request.client else "unknown")
        ) > 0:
            retry = math.ceil(wait)
            response = error_response(
                429, f"Too many requests. Try again in {retry} s.", request_id,
                headers={"Retry-After": str(retry)}, retry_after_s=retry,
            )
        else:
            try:
                response = await call_next(request)
            except Exception:  # A-04: full detail to the logs, a generic envelope to the client
                logger.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
                response = error_response(500, "Something went wrong on our side. Please try again.", request_id)

        response.headers["X-Request-ID"] = request_id
        # Method, path and status only: bodies carry user free text and aren't logged (O-07).
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
