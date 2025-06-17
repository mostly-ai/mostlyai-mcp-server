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

from typing import Literal

import click
from fastmcp import FastMCP, Context
import requests
import asyncio
from fastmcp.server.dependencies import get_context

mcp = FastMCP(name="MOSTLY AI MCP Server")


@mcp.tool(description="Get general information about the MOSTLY AI Service")
def get_service_info() -> dict[str, str]:
    return {
        "version": "4.7.2",
        "assistant": True,
    }


@mcp.resource(name="mostlyai-docs", description="MOSTLY AI SDK Documentation", uri="https://mostly-ai.github.io/mostlyai/llms-full.txt")
def get_mostlyai_docs() -> str:
    response = requests.get("https://mostly-ai.github.io/mostlyai/llms-full.txt")
    response.raise_for_status()  # raise an exception for bad status codes
    return response.text


@mcp.tool(description="Simulate a training process with progress updates")
async def train(model_name: str) -> dict[str, str]:
    # get the context for progress updates
    ctx = get_context()
    
    # simulate 20 seconds of training with progress updates every second
    for i in range(20):
        # calculate progress percentage
        progress = (i + 1) * 5  # 5% increments
        
        # report progress
        await ctx.info(f"Training {model_name}: {progress}% complete")
        
        # simulate work
        await asyncio.sleep(1)
    
    # report completion
    await ctx.info(f"Training of {model_name} completed successfully!")
    
    return {
        "status": "completed",
        "model": model_name,
        "message": "Training completed successfully"
    }


@click.command()
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--host", default="localhost", help="Host to bind to")
@click.option(
    "--transport",
    default="sse",
    type=click.Choice(["sse", "streamable-http"]),
    help="Transport protocol to use ('sse' or 'streamable-http')",
)
def main(port: int, host: str, transport: Literal["sse", "streamable-http"]) -> int:
    mcp.run(transport=transport, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
