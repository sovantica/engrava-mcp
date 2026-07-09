"""End-to-end smoke test over a real stdio subprocess.

Unlike the in-memory transport the other tests use, this spawns the server
as a real ``python -m engrava_mcp`` subprocess and talks to it over the MCP
SDK's stdio transport. It exercises the console entry point, the package's
module-run wiring, and the JSON-RPC-over-stdio serialisation path that the
in-process tests cannot.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

if TYPE_CHECKING:
    from pathlib import Path

#: Every tool the server advertises over stdio (6 read + 5 write).
EXPECTED_TOOL_NAMES = frozenset(
    {
        "get_thought",
        "search_memory",
        "search_keywords",
        "list_memory",
        "query_memory",
        "memory_stats",
        "store_thought",
        "update_thought",
        "link_thoughts",
        "delete_thought",
        "delete_edge",
    }
)


def _server_params(db_path: Path) -> StdioServerParameters:
    """Build the stdio launch parameters for a fresh server subprocess.

    Args:
        db_path: Temp SQLite path the subprocess resolves its store from.

    Returns:
        Parameters that launch ``python -m engrava_mcp`` against ``db_path``
        in a read-write (non read-only) configuration.

    """
    env = dict(os.environ)
    env["ENGRAVA_DB_PATH"] = str(db_path)
    env.pop("ENGRAVA_MCP_READ_ONLY", None)
    env.pop("ENGRAVA_MCP_CONFIG", None)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "engrava_mcp"],
        env=env,
    )


async def test_stdio_subprocess_serves_tools(tmp_path: Path) -> None:
    """Spawn the real server and round-trip a tool call over stdio.

    Asserts the client initialises, the full 11-tool surface is advertised,
    and a ``memory_stats`` call returns a valid result for the empty store.

    Args:
        tmp_path: Pytest temp directory holding the throwaway database file.

    """
    params = _server_params(tmp_path / "smoke.sqlite")

    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            init_result = await session.initialize()
            assert init_result.serverInfo.name

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == EXPECTED_TOOL_NAMES

            result = await session.call_tool("memory_stats", {})
            assert result.isError is False
            assert result.structuredContent is not None
            # A freshly created store has no thoughts.
            assert result.structuredContent["thought_count"] == 0
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - sandbox guard
        pytest.skip(f"stdio subprocess could not be spawned in this environment: {exc}")
