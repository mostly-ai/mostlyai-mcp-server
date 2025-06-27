# MOSTLY AI MCP Server

![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mcp-server)
[![GitHub stars](https://img.shields.io/github/stars/mostly-ai/mostly-mcp-server?style=social)](https://github.com/mostly-ai/mostlyai-mcp-server/stargazers)

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server with OAuth 2.1 support for LLM agents to interact with the [MOSTLY AI Platform](https://app.mostly.ai).

## Available Tools

| Tool | Description |
|------|-------------|
| `get_platform_info` | retrieve information about the platform |
| `get_user_info` | retrieve information about the current user |
| `get_models` | retrieve a list of available models of a specific type |
| `get_computes` | retrieve a list of available compute resources for tasks |
| `list_connectors` | list all available connectors |
| `get_connector` | get details of a specific connector |
| `create_connector` | create a new connector and optionally test the connection |
| `get_connector_locations` | list available locations (schemas, tables, buckets, etc.) for a connector |
| `get_connector_schema` | get the schema of a table or file at a connector location |
| `read_connector_data` | read data from a connector location |
| `write_connector_data` | write data to a connector location (currently disabled) |
| `query_connector` | execute a read-only SQL query against a connector's data source |
| `list_generators` | list all available generators |
| `get_generator` | get details of a specific generator |
| `train_generator` | train a new generator with provided data or configuration |
| `clone_generator` | clone an existing generator |
| `generate_synthetic_dataset` | generate a synthetic dataset using a generator |
| `list_synthetic_datasets` | list all available synthetic datasets |
| `probe_generator` | probe a generator for a new synthetic dataset (quick sample) |

## Usage of Remote Server

### Prerequisites

An MCP host that supports the latest MCP specification and remote servers, such as [Claude](https://claude.ai/), [VS Code](https://code.visualstudio.com/) and [Cursor](https://www.cursor.com/).

### Installation
<details>

  **<summary>Claude Desktop or Web UI</summary>**

  `Search and tools` -> `Add integrations` -> `Add integration` -> Set Integration name to the name of your choice (e.g., `mostlyai`) and Integration URL to `https://mcp.mostly.ai/sse`.

</details>

<details>

  **<summary>VS Code (>= v1.101)</summary>**

  [![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0098FF)](https://insiders.vscode.dev/redirect/mcp/install?name=mostlyai&config=%7B%22url%22%3A%20%22https%3A%2F%2Fmcp.mostly.ai%2Fsse%22%7D)

For quick installation, please use the one-click button above.
Alternatively, one can add the following JSON to `.vscode/mcp.json` (for the workspace) or the global `settings.json` (for the user).

```json
{
  "servers": {
    "mostlyai": {
      "url": "https://mcp.mostly.ai/sse"
    }
  }
}
```

</details>

<details>

  **<summary>Cursor</summary>**
    [![Install in Cursor](https://img.shields.io/badge/Cursor-Install_Server-111111)](https://cursor.com/install-mcp?name=mostlyai&config=eyJ1cmwiOiJodHRwczovL21jcC5tb3N0bHkuYWkvc3NlIn0%3D)

For quick installation, please use the one-click button above.
Alternatively, one can add the following JSON to `~/.cursor/mcp.json`.

```json
{
    "mcpServers": {
      "mostlyai": {
        "url": "https://mcp.mostly.ai/sse"
      }
    }
}
```

</details>

## Development

### Setup virtual environment

`uv sync --frozen`

### Start the server locally

1. Set the necessary environment variables (see `.env.example`)
2. `uv run mostlyai-mcp-server --port 8000`
3. Connect to the server with the configuration described in the above section, but change the url to `http://localhost:8000/sse`.

### Debug

We recommend to use the Anthropic's [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) for testing and debugging by running `npx @modelcontextprotocol/inspector`.
