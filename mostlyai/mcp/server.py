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

    @mcp.custom_route("/oauth/callback", methods=["GET"])
    async def oauth_callback_handler(request: Request) -> Response:
        logger.info("oauth callback handler called")
        
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")

        if error:
            error_description = request.query_params.get("error_description", error)
            logger.error(f"oauth callback error: {error}")
            return JSONResponse({"error": error, "error_description": error_description}, status_code=400)

        if not code or not state:
            logger.error("oauth callback missing code or state parameter")
            return JSONResponse({"error": "Missing code or state parameter"}, status_code=400)

        try:
            logger.info("handling oauth callback")
            redirect_url = await auth.handle_callback(code, state)
            logger.info("oauth callback successful")
            return RedirectResponse(url=redirect_url, status_code=302)
        except HTTPException as e:
            logger.error(f"oauth callback HTTP exception: {e.status_code}")
            return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)
        except Exception as e:
            logger.exception(f"oauth callback error: {str(e)}")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    def _get_keycloak_token() -> str:
        """Get the Keycloak token for the current request."""
        mcp_token_obj = get_access_token()
        if not mcp_token_obj:
            logger.error("no MCP token found")
            raise ValueError("Unauthorized: no MCP token found")

        keycloak_token = auth._mcp_access_to_keycloak_access_map.get(mcp_token_obj.token)
        if not keycloak_token:
            logger.error("invalid or expired MCP token")
            raise ValueError("Unauthorized: invalid or expired MCP token")

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

            async with create_mcp_http_client() as client:
                response = await client.get(
                    full_url,
                    headers={"Authorization": f"Bearer {keycloak_token}"},
                )

                if response.status_code != 200:
                    error_msg = f"Failed to send request to {endpoint}: {response.text}"
                    logger.error(error_msg)
                    raise HTTPException(400, error_msg)

                return response.json()
                
        except Exception as e:
            logger.exception(f"error in _request for endpoint {endpoint}")
            raise

    # ================ TOOLS ================

    @mcp.tool(description="Get the service info from Mostly AI.")
    async def get_service_info() -> dict[str, Any]:
        logger.info("get_service_info tool called")
        return await _request("/api/v2/about")

    @mcp.tool(description="Get the user info from Mostly AI.")
    async def get_user_info() -> dict[str, Any]:
        logger.info("get_user_info tool called")
        return await _request("/api/v2/users/me")

    logger.info("keycloak MCP server created successfully")
    return mcp


@click.command()
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
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
