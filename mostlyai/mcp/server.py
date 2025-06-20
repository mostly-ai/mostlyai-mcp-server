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
from mostlyai.sdk import MostlyAI

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
        
    async def _mostly() -> MostlyAI:
        keycloak_token = _get_keycloak_token()
        mostly = MostlyAI(base_url=os.environ["MOSTLY_BASE_URL"], bearer_token=keycloak_token)
        return mostly

    # ================ TOOLS ================

    @mcp.tool(description="Get the service info from Mostly AI.")
    async def get_service_info() -> dict[str, Any]:
        mostly = await _mostly()
        return mostly.about()

    @mcp.tool(description="Get the user info from Mostly AI.")
    async def get_user_info() -> dict[str, Any]:
        mostly = await _mostly()
        return mostly.me()

    @mcp.tool(description="List all connectors available to the user.")
    async def list_connectors() -> dict[str, Any]:
        mostly = await _mostly()
        connectors = list(mostly.connectors.list())
        return {"connectors": connectors}

    @mcp.tool(description="List all generators available to the user.")
    async def list_generators() -> dict[str, Any]:
        mostly = await _mostly()
        generators = list(mostly.generators.list())
        return {"generators": generators}

    @mcp.tool(description="List all synthetic datasets available to the user.")
    async def list_synthetic_datasets() -> dict[str, Any]:
        mostly = await _mostly()
        synthetics = list(mostly.synthetics.list())
        return {"synthetic_datasets": synthetics}

    @mcp.tool(description="Train a new generator on provided data. Returns the generator details.")
    async def train_generator(
        name: str,
        data_url: str,
    ) -> dict[str, Any]:
        """Train a generator on data from a URL (CSV file)."""
        mostly = await _mostly()
        
        # train a generator using the provided data URL
        generator = await mostly.train(
            name=name,
            data=data_url,
        )
        
        return {
            "generator_id": generator.id,
            "name": generator.name,
            "status": generator.status,
            "description": generator.description
        }

    @mcp.tool(description="Probe a generator to get synthetic samples. Returns the generated data.")
    async def probe_generator(
        generator_id: str,
        size: int = 100
    ) -> dict[str, Any]:
        """Live probe a generator to get synthetic samples on demand."""
        mostly = await _mostly()
        
        # get the generator by ID
        generator = await mostly.generators.get(generator_id)
        
        # probe for synthetic samples
        df = await mostly.probe(generator, size=size)
        
        # convert dataframe to dict for JSON serialization
        return {
            "generator_id": generator_id,
            "sample_size": size,
            "data": df.to_dict(orient="records") if hasattr(df, 'to_dict') else df
        }

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
