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

import json
from unittest.mock import patch

import pytest
from fastmcp import Client

import mostlyai.sdk.client.api as mostlyai_api
import mostlyai.sdk.client.connectors as connectors
import mostlyai.sdk.client.generators as generators
from mostlyai.mcp import server as mcp_server_mod
from mostlyai.mcp.server import create_keycloak_mcp_server

ZERO_PARAMS_TOOLS = [
    "get_platform_info",
    "get_user_info",
    "get_models",
    "get_computes",
]

TOOLS = [
    *ZERO_PARAMS_TOOLS,
    "list_connectors",
    "get_connector",
    "create_connector",
    "get_connector_locations",
    "get_connector_schema",
    "read_connector_data",
    "query_connector",
    "list_generators",
    "get_generator",
    "poll_generator_progress",
    "train_generator",
    "clone_generator",
    "get_synthetic_dataset",
    "poll_synthetic_dataset_progress",
    "generate_synthetic_dataset",
    "list_synthetic_datasets",
    "probe_generator",
]


@pytest.fixture
def mcp_server(monkeypatch):
    monkeypatch.setenv("MCP_KEYCLOAK_REALM", "test-realm")
    monkeypatch.setenv("MCP_KEYCLOAK_CLIENT_ID", "test-client")
    monkeypatch.setenv("MCP_KEYCLOAK_AUTH_URL", "http://localhost:8080/auth")
    monkeypatch.setenv("MOSTLY_BASE_URL", "https://mostly.ai")
    monkeypatch.setattr(mcp_server_mod, "_get_keycloak_token", lambda ctx, auth: "test-keycloak-token")
    server = create_keycloak_mcp_server(host="localhost", port=8000)
    return server


@pytest.fixture
def zero_params_api_mock(monkeypatch):
    tool_to_method_and_value = {
        "get_platform_info": ("about", {"version": "dummy-version"}),
        "get_user_info": ("me", {"id": "dummy-user-uuid"}),
        "get_models": ("models", {"TABULAR": [], "LANGUAGE": []}),
        "get_computes": ("computes", [{"id": "dummy-compute-uuid"}]),
    }
    for tool, (method, value) in tool_to_method_and_value.items():
        monkeypatch.setattr(mostlyai_api.MostlyAI, method, lambda self, v=value: v)
    return {tool: value for tool, (method, value) in tool_to_method_and_value.items()}


@pytest.mark.asyncio
async def test_list_tools(mcp_server):
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        assert sorted([tool.name for tool in tools]) == sorted(TOOLS)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ZERO_PARAMS_TOOLS)
async def test_zero_params_tools(mcp_server, zero_params_api_mock, tool):
    async with Client(mcp_server) as client:
        result = await client.call_tool(tool)
        actual = json.loads(result[0].text)
        expected = zero_params_api_mock[tool]
        if isinstance(actual, dict) and isinstance(expected, list) and len(expected) == 1:
            actual = [actual]
        elif isinstance(actual, list) and isinstance(expected, dict):
            expected = [expected]
        assert actual == expected


@pytest.mark.asyncio
async def test_list_connectors(mcp_server):
    with patch.object(connectors._MostlyConnectorsClient, "list", return_value=[]) as mock_list:
        async with Client(mcp_server) as client:
            await client.call_tool(
                "list_connectors", {"limit": 100, "access_type": "WRITE_DATA", "created_from": "2024-06-01"}
            )
        mock_list.assert_called_once_with(
            offset=0,
            limit=100,
            search_term=None,
            access_type="WRITE_DATA",
            owner_id=None,
            visibility=None,
            created_from="2024-06-01",
            created_to=None,
            sort_by=None,
        )


@pytest.mark.asyncio
async def test_get_generator(mcp_server):
    with patch.object(generators._MostlyGeneratorsClient, "get", return_value=[]) as mock_get:
        async with Client(mcp_server) as client:
            await client.call_tool("get_generator", {"generator_id": "dummy-generator-uuid"})
        mock_get.assert_called_once_with("dummy-generator-uuid")


@pytest.mark.asyncio
async def test_generate_synthetic_dataset(mcp_server):
    with patch.object(mostlyai_api.MostlyAI, "generate", return_value=[]) as mock_generate:
        async with Client(mcp_server) as client:
            await client.call_tool("generate_synthetic_dataset", {"generator": "dummy-generator-uuid", "size": 100})
        mock_generate.assert_called_once_with(
            generator="dummy-generator-uuid",
            config=None,
            size=100,
            seed=None,
            name=None,
            start=True,
            wait=False,
        )
