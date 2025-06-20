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
    auth = KeycloakOAuthProvider(host=host, port=port)
    mcp = FastMCP(name="Mostly AI MCP Server", auth=auth)

    @mcp.custom_route("/oauth/callback", methods=["GET"])
    async def oauth_callback_handler(request: Request) -> Response:
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")

        if error:
            error_description = request.query_params.get("error_description", error)
            return JSONResponse({"error": error, "error_description": error_description}, status_code=400)

        if not code or not state:
            return JSONResponse({"error": "Missing code or state parameter"}, status_code=400)

        try:
            redirect_url = await auth.handle_callback(code, state)
            return RedirectResponse(url=redirect_url, status_code=302)
        except HTTPException as e:
            return JSONResponse({"error": str(e.detail)}, status_code=e.status_code)
        except Exception:
            logger.exception("Error in OAuth callback")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    def _get_keycloak_token() -> str:
        """Get the Keycloak token for the current request."""
        mcp_token_obj = get_access_token()
        if not mcp_token_obj:
            raise ValueError("Unauthorized: no MCP token found")

        keycloak_token = auth._mcp_access_to_keycloak_access_map.get(mcp_token_obj.token)
        if not keycloak_token:
            raise ValueError("Unauthorized: invalid or expired MCP token")

        return keycloak_token

    async def _request(endpoint: str) -> dict[str, Any]:
        keycloak_token = _get_keycloak_token()

        async with create_mcp_http_client() as client:
            response = await client.get(
                f"{os.environ['MOSTLY_BASE_URL']}{endpoint}",
                headers={"Authorization": f"Bearer {keycloak_token}"},
            )

            if response.status_code != 200:
                raise HTTPException(400, f"Failed to send request to {endpoint}: {response.text}")

            return response.json()

    # ================ TOOLS ================

    @mcp.tool(description="Get the service info from Mostly AI.")
    async def get_service_info() -> dict[str, Any]:
        return await _request("/api/v2/about")

    @mcp.tool(description="Get the user info from Mostly AI.")
    async def get_user_info() -> dict[str, Any]:
        return await _request("/api/v2/users/me")

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
