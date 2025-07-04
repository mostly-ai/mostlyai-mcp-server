# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from prometheus_client import REGISTRY, Counter, Gauge, Histogram
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

INFO = Gauge("app_info", "Application information.", ["application"])
REQUESTS = Counter("requests_total", "Total count of requests by method and path.", ["method", "path", "application"])
RESPONSES = Counter(
    "responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "application"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path", "application"],
)
EXCEPTIONS = Counter(
    "exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type", "application"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path", "application"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, application: str) -> None:
        super().__init__(app)
        self.application = application
        INFO.labels(application=self.application).inc()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        path, is_handled_path = self.get_path(request)

        if not is_handled_path:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(method=method, path=path, application=self.application).inc()
        REQUESTS.labels(method=method, path=path, application=self.application).inc()
        before_time = time.perf_counter()
        try:
            response = await call_next(request)
        except BaseException as e:
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(
                method=method, path=path, exception_type=type(e).__name__, application=self.application
            ).inc()
            raise e from None
        else:
            status_code = response.status_code
            after_time = time.perf_counter()

            REQUESTS_PROCESSING_TIME.labels(method=method, path=path, application=self.application).observe(
                after_time - before_time
            )
        finally:
            RESPONSES.labels(method=method, path=path, status_code=status_code, application=self.application).inc()
            REQUESTS_IN_PROGRESS.labels(method=method, path=path, application=self.application).dec()

        return response

    @staticmethod
    def get_path(request: Request) -> tuple[str, bool]:
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True

        return request.url.path, False


async def metrics(request: Request) -> Response:
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})
