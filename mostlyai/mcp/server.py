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

import logging
import os
from typing import Any, Literal
import asyncio
import contextlib
import weakref

import click
from fastmcp import FastMCP
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.shared._httpx_utils import create_mcp_http_client
from mostlyai.mcp.keycloak import KeycloakOAuthProvider
from mostlyai.mcp.logger import init_logging

init_logging()

logger = logging.getLogger(__name__)

DEFAULT_AUTH_CODE_EXPIRY_SECONDS = 5 * 60  # 5 minutes
DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS = 1 * 60  # 5 minutes
DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS = 3 * 60  # 15 minutes


def create_keycloak_mcp_server(host: str, port: int) -> FastMCP:
    logger.info(f"creating keycloak MCP server on {host}:{port}")
    auth = KeycloakOAuthProvider(host=host, port=port)
    mcp = FastMCP(name="Mostly AI MCP Server", auth=auth)

    # track active requests for debugging
    active_requests = set()
    # track running tasks for debugging
    running_tasks = weakref.WeakSet()

    async def _monitor_tasks():
        """Monitor running tasks and log periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # check every 30 seconds
                current_tasks = [t for t in asyncio.all_tasks() if not t.done()]
                logger.debug(f"monitoring: {len(current_tasks)} active tasks, {len(running_tasks)} tracked tasks")
                
                # log details of long-running tasks
                for task in current_tasks:
                    if hasattr(task, '_start_time'):
                        runtime = asyncio.get_event_loop().time() - task._start_time
                        if runtime > 60:  # tasks running for more than 1 minute
                            logger.warning(f"long-running task: {task.get_name()}, runtime: {runtime:.1f}s")
            except Exception as e:
                logger.debug(f"task monitor error: {e}")

    @mcp.on_startup
    async def startup_handler():
        logger.info("mcp server startup handler called")
        # start background task monitor
        monitor_task = asyncio.create_task(_monitor_tasks(), name="task_monitor")
        running_tasks.add(monitor_task)
        logger.info("startup handler completed")

    @mcp.on_shutdown
    async def shutdown_handler():
        logger.info("mcp server shutdown handler called")
        logger.info(f"active requests during shutdown: {len(active_requests)}")
        if active_requests:
            logger.warning(f"found {len(active_requests)} active requests during shutdown")
            for req_id in list(active_requests):
                logger.warning(f"active request: {req_id}")
        
        # cleanup auth provider resources
        if hasattr(auth, 'cleanup'):
            logger.info("cleaning up auth provider")
            await auth.cleanup()
        
        # cancel all tracked tasks
        logger.info(f"cancelling {len(running_tasks)} tracked tasks")
        for task in list(running_tasks):
            if not task.done():
                logger.debug(f"cancelling task: {task.get_name()}")
                task.cancel()
        
        # wait briefly for tasks to finish
        if running_tasks:
            await asyncio.sleep(1)
        
        logger.info("mcp server shutdown handler completed")

    @mcp.custom_route("/oauth/callback", methods=["GET"])
    async def oauth_callback_handler(request: Request) -> Response:
        request_id = id(request)
        active_requests.add(request_id)
        logger.info(f"oauth callback handler called, request_id: {request_id}")
        
        try:
            code = request.query_params.get("code")
            state = request.query_params.get("state")
            error = request.query_params.get("error")

            logger.debug(f"oauth callback params - code: {'present' if code else 'missing'}, state: {'present' if state else 'missing'}, error: {error}")

            if error:
                error_description = request.query_params.get("error_description", error)
                logger.error(f"oauth callback error: {error}, description: {error_description}")
                return JSONResponse({"error": error, "error_description": error_description}, status_code=400)

            if not code or not state:
                logger.error("oauth callback missing code or state parameter")
                return JSONResponse({"error": "Missing code or state parameter"}, status_code=400)

            try:
                logger.info("handling oauth callback")
                redirect_url = await auth.handle_callback(code, state)
                logger.info(f"oauth callback successful, redirecting to: {redirect_url}")
                return RedirectResponse(url=redirect_url, status_code=302)
            except HTTPException as e:
                logger.error(f"oauth callback HTTP exception: {e.status_code} - {e.detail}")
                return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)
            except Exception as e:
                logger.exception(f"oauth callback unexpected error: {str(e)}")
                return JSONResponse({"error": "Internal server error"}, status_code=500)
        finally:
            active_requests.discard(request_id)
            logger.debug(f"oauth callback handler finished, request_id: {request_id}")

    def _get_keycloak_token() -> str:
        """Get the Keycloak token for the current request."""
        logger.debug("getting keycloak token from MCP token")
        mcp_token_obj = get_access_token()
        if not mcp_token_obj:
            logger.error("no MCP token found in request")
            raise ValueError("Unauthorized: no MCP token found")

        keycloak_token = auth._mcp_access_to_keycloak_access_map.get(mcp_token_obj.token)
        if not keycloak_token:
            logger.error("invalid or expired MCP token")
            raise ValueError("Unauthorized: invalid or expired MCP token")

        logger.debug("keycloak token retrieved successfully")
        return keycloak_token

    async def _request(endpoint: str) -> dict[str, Any]:
        logger.info(f"making request to endpoint: {endpoint}")
        
        try:
            keycloak_token = _get_keycloak_token()
            base_url = os.environ.get('MOSTLY_BASE_URL')
            if not base_url:
                logger.error("MOSTLY_BASE_URL environment variable not set")
                raise ValueError("MOSTLY_BASE_URL environment variable not set")
            
            full_url = f"{base_url}{endpoint}"
            logger.debug(f"requesting URL: {full_url}")

            async with create_mcp_http_client() as client:
                logger.debug("created HTTP client, sending request")
                response = await client.get(
                    full_url,
                    headers={"Authorization": f"Bearer {keycloak_token}"},
                )

                logger.info(f"request completed with status: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Failed to send request to {endpoint}: {response.text}"
                    logger.error(error_msg)
                    raise HTTPException(400, error_msg)

                result = response.json()
                logger.debug(f"request successful, response size: {len(str(result))}")
                return result
                
        except Exception as e:
            logger.exception(f"error in _request for endpoint {endpoint}: {str(e)}")
            raise

    # ================ TOOLS ================

    @mcp.tool(description="Get the service info from Mostly AI.")
    async def get_service_info() -> dict[str, Any]:
        logger.info("get_service_info tool called")
        try:
            result = await _request("/api/v2/about")
            logger.info("get_service_info completed successfully")
            return result
        except Exception as e:
            logger.error(f"get_service_info failed: {str(e)}")
            raise

    @mcp.tool(description="Get the user info from Mostly AI.")
    async def get_user_info() -> dict[str, Any]:
        logger.info("get_user_info tool called")
        try:
            result = await _request("/api/v2/users/me")
            logger.info("get_user_info completed successfully")
            return result
        except Exception as e:
            logger.error(f"get_user_info failed: {str(e)}")
            raise

    logger.info("keycloak MCP server created successfully")
    return mcp


@click.command()
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option(
    "--transport",
    default="sse",
    type=click.Choice(["sse", "streamable-http"]),
    help="Transport protocol to use ('sse' or 'streamable-http')",
)
def main(port: int, host: str, transport: Literal["sse", "streamable-http"]) -> None:
    mcp = create_keycloak_mcp_server(host=host, port=port)
    mcp.run(transport=transport, log_level="debug")


if __name__ == "__main__":
    main()
