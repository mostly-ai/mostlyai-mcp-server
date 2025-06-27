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
from contextlib import asynccontextmanager
from typing import Any, Literal

import click
import pandas as pd
from fastmcp import Context, FastMCP
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from mostlyai.mcp.keycloak import KeycloakOAuthProvider
from mostlyai.mcp.logger import init_logging
from mostlyai.mcp.utils import df_as_dict, doc_section
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import (
    AboutService,
    Connector,
    ConnectorListItem,
    CurrentUser,
    Generator,
    GeneratorListItem,
    SyntheticDataset,
    SyntheticDatasetListItem,
)

init_logging()

logger = logging.getLogger(__name__)

DEFAULT_AUTH_CODE_EXPIRY_SECONDS = 5 * 60  # 5 minutes
DEFAULT_ACCESS_TOKEN_EXPIRY_SECONDS = 1 * 60  # 1 minutes
DEFAULT_REFRESH_TOKEN_EXPIRY_SECONDS = 15 * 60  # 15 minutes


def create_keycloak_mcp_server(host: str, port: int) -> FastMCP:
    auth = KeycloakOAuthProvider(host=host, port=port)
    mcp = FastMCP(name="Mostly AI MCP Server", auth=auth)

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> Response:
        return Response(status_code=200)

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

    def _get_keycloak_token(ctx: Context) -> str:
        # FIXME: for some reason, the token in the auth context is not up to date during CallToolRequest
        # therefore mcp.server.auth.middleware.auth_context.get_access_token may not work all the time
        # this is a workaround to get the token from the request context, which seems to be more reliable
        try:
            request_context = ctx.request_context
            mcp_token = request_context.request.headers["Authorization"].split(" ")[1]
        except Exception:
            raise ValueError("Unauthorized: no MCP token found")

        keycloak_token = auth._mcp_access_to_keycloak_access_map.get(mcp_token)
        if not keycloak_token:
            raise ValueError("Unauthorized: invalid or expired MCP token")

        return keycloak_token

    @asynccontextmanager
    async def _mostly(ctx: Context):
        try:
            keycloak_token = _get_keycloak_token(ctx)
            mostly = MostlyAI(base_url=os.environ["MOSTLY_BASE_URL"], bearer_token=keycloak_token, quiet=True)
            yield mostly
        except Exception as e:
            raise e

    # ================ TOOLS ================

    # --- Top-level API ---

    @mcp.tool(description="Retrieve information about the platform.")
    async def get_platform_info(ctx: Context) -> AboutService | dict:
        async with _mostly(ctx) as mostly:
            return mostly.about()

    @mcp.tool(description="Retrieve information about the current user.")
    async def get_user_info(ctx: Context) -> CurrentUser | dict:
        async with _mostly(ctx) as mostly:
            return mostly.me()

    @mcp.tool(description="Retrieve a list of available models of a specific type.")
    async def get_models(ctx: Context) -> dict[str, list[str]] | dict:
        async with _mostly(ctx) as mostly:
            return mostly.models()

    @mcp.tool(description="Retrieve a list of available compute resources, that can be used for executing tasks.")
    async def get_computes(ctx: Context) -> list[dict[str, any]] | dict:
        async with _mostly(ctx) as mostly:
            return mostly.computes()

    # --- Connectors ---

    @mcp.tool(description=doc_section(r"### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.list"))
    async def list_connectors(
        ctx: Context,
        offset: int = 0,
        limit: int | None = None,
        search_term: str | None = None,
        access_type: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> list[ConnectorListItem] | dict:
        async with _mostly(ctx) as mostly:
            return list(
                mostly.connectors.list(
                    offset=offset,
                    limit=limit,
                    search_term=search_term,
                    access_type=access_type,
                    owner_id=owner_id,
                    visibility=visibility,
                    created_from=created_from,
                    created_to=created_to,
                    sort_by=sort_by,
                )
            )

    @mcp.tool(description=doc_section(r"### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.get"))
    async def get_connector(
        ctx: Context,
        connector_id: str,
    ) -> Connector | dict:
        async with _mostly(ctx) as mostly:
            return mostly.connectors.get(connector_id)

    @mcp.tool(description=doc_section(r"### mostlyai.sdk.client.connectors.\_MostlyConnectorsClient.create"))
    async def create_connector(
        ctx: Context,
        config: dict,
        test_connection: bool | None = True,
    ) -> Connector | dict:
        async with _mostly(ctx) as mostly:
            return mostly.connect(config=config, test_connection=test_connection)

    @mcp.tool(description=doc_section("### mostlyai.sdk.domain.Connector.locations"))
    async def get_connector_locations(
        ctx: Context,
        connector_id: str,
        prefix: str = "",
    ) -> list[str] | dict:
        async with _mostly(ctx) as mostly:
            return mostly.connectors._locations(connector_id, prefix)

    @mcp.tool(description=doc_section("### mostlyai.sdk.domain.Connector.schema"))
    async def get_connector_schema(ctx: Context, connector_id: str, location: str) -> list[dict[str, Any]] | dict:
        async with _mostly(ctx) as mostly:
            return mostly.connectors._schema(connector_id, location)

    @mcp.tool(description=doc_section("### mostlyai.sdk.domain.Connector.read_data"))
    async def read_connector_data(
        ctx: Context,
        connector_id: str,
        location: str,
        limit: int | None = None,
        shuffle: bool = False,
    ) -> dict:
        async with _mostly(ctx) as mostly:
            result = mostly.connectors._read_data(connector_id, location, limit, shuffle)
            return df_as_dict(result)

    @mcp.tool(
        description=doc_section("### mostlyai.sdk.domain.Connector.write_data")
        + "The data must be provided as a dictionary of lists, where the keys are the column names and the values are the column values. Example: {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}",
        enabled=False,  # FIXME: this is not working as expected
    )
    async def write_connector_data(
        ctx: Context,
        connector_id: str,
        location: str,
        data: dict[str, list[Any]],
        if_exists: str = "fail",
    ) -> None | dict:
        async with _mostly(ctx) as mostly:
            return mostly.connectors._write_data(connector_id, location, pd.DataFrame.from_dict(data), if_exists)

    @mcp.tool(description=doc_section("##### query"))
    async def query_connector(
        ctx: Context,
        connector_id: str,
        sql: str,
    ) -> dict:
        async with _mostly(ctx) as mostly:
            result = mostly.connectors._query(connector_id, sql)
            return df_as_dict(result)

    # --- Generators ---

    @mcp.tool(description=doc_section(r"### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.list"))
    async def list_generators(
        ctx: Context,
        offset: int = 0,
        limit: int | None = None,
        status: str | list[str] | None = None,
        search_term: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> list[GeneratorListItem] | dict:
        async with _mostly(ctx) as mostly:
            return list(
                mostly.generators.list(
                    offset=offset,
                    limit=limit,
                    status=status,
                    search_term=search_term,
                    owner_id=owner_id,
                    visibility=visibility,
                    created_from=created_from,
                    created_to=created_to,
                    sort_by=sort_by,
                )
            )

    @mcp.tool(description=doc_section(r"### mostlyai.sdk.client.generators.\_MostlyGeneratorsClient.get"))
    async def get_generator(
        ctx: Context,
        generator_id: str,
    ) -> Generator | dict:
        async with _mostly(ctx) as mostly:
            return mostly.generators.get(generator_id)

    @mcp.tool(description=doc_section("### mostlyai.sdk.client.api.MostlyAI.train"))
    async def train_generator(
        ctx: Context,
        config: dict | None = None,
        data: str | None = None,
        name: str | None = None,
        start: bool = True,
        wait: bool = False,  # no effect yet
        progress_bar: bool = False,  # no effect yet
    ) -> Generator | dict:
        async with _mostly(ctx) as mostly:
            g = mostly.train(
                config=config,
                data=data,
                name=name,
                start=start,
                wait=False,
                progress_bar=False,
            )
            # if wait:
            #    await job_wait(ctx=ctx, get_progress_fn=g.training.progress, progress_bar=progress_bar)
            #     g.reload()
            return g

    @mcp.tool(description=doc_section("##### clone"))
    async def clone_generator(
        ctx: Context,
        generator_id: str,
        training_status: str = "pending",
    ) -> Generator | dict:
        async with _mostly(ctx) as mostly:
            return mostly.generators._clone(generator_id, training_status)

    # --- Synthetic Datasets ---

    @mcp.tool(description=doc_section("### mostlyai.sdk.client.api.MostlyAI.generate"))
    async def generate_synthetic_dataset(
        ctx: Context,
        generator: str,
        config: dict | None = None,
        size: int | dict | None = None,
        seed: dict | None = None,
        name: str | None = None,
        start: bool = True,
        wait: bool = False,  # no effect yet
        progress_bar: bool = False,  # no effect yet
    ) -> SyntheticDataset | dict:
        async with _mostly(ctx) as mostly:
            sd = mostly.generate(
                generator=generator,
                config=config,
                size=size,
                seed=seed,
                name=name,
                start=start,
                wait=False,
                progress_bar=False,
            )
            # if wait:
            #    await job_wait(ctx=ctx, get_progress_fn=sd.generation.progress, progress_bar=progress_bar)
            #    sd.reload()
            return sd

    @mcp.tool(
        description=doc_section(r"### mostlyai.sdk.client.synthetic_datasets.\_MostlySyntheticDatasetsClient.list")
    )
    async def list_synthetic_datasets(
        ctx: Context,
        offset: int = 0,
        limit: int | None = None,
        status: str | list[str] | None = None,
        search_term: str | None = None,
        owner_id: str | list[str] | None = None,
        visibility: str | list[str] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        sort_by: str | list[str] | None = None,
    ) -> list[SyntheticDatasetListItem] | dict:
        async with _mostly(ctx) as mostly:
            return list(
                mostly.synthetic_datasets.list(
                    offset=offset,
                    limit=limit,
                    status=status,
                    search_term=search_term,
                    owner_id=owner_id,
                    visibility=visibility,
                    created_from=created_from,
                    created_to=created_to,
                    sort_by=sort_by,
                )
            )

    @mcp.tool(description=doc_section("### mostlyai.sdk.client.api.MostlyAI.probe"))
    async def probe_generator(
        ctx: Context,
        generator: str,
        size: int | dict | None = None,
        seed: dict | None = None,
        config: dict | None = None,
        return_type: str = "auto",
    ) -> dict:
        async with _mostly(ctx) as mostly:
            result = mostly.probe(
                generator=generator,
                size=size,
                seed=seed,
                config=config,
                return_type=return_type,
            )
            return df_as_dict(result)

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
