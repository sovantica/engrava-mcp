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
        "get_edges",
        "list_edges",
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

    Asserts the client initialises, the full 13-tool surface is advertised,
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
            assert init_result.server_info.name

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == EXPECTED_TOOL_NAMES

            result = await session.call_tool("memory_stats", {})
            assert result.is_error is False
            assert result.structured_content is not None
            # A freshly created store has no thoughts.
            assert result.structured_content["thought_count"] == 0
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - sandbox guard
        pytest.skip(f"stdio subprocess could not be spawned in this environment: {exc}")


async def test_stdio_subprocess_logs_opening_line_to_errlog(tmp_path: Path) -> None:
    """The startup progress line reaches a real ``errlog`` file, not stdout.

    Passes the SDK's ``stdio_client`` a real temporary file as ``errlog`` --
    the parameter it forwards to the subprocess's own stderr -- and asserts
    that ``resolve_store``'s "opening the store from" line lands there, and
    that the session still initialises and lists its tools exactly as the
    other test in this module already proves. This test does not inspect raw
    stdout at all: keeping the log lines off it rests on the MCP SDK's own
    ``configure_logging`` call (``mcp.server.mcpserver.utilities.logging``),
    not on anything asserted here.

    Args:
        tmp_path: Pytest temp directory holding the throwaway database file
            and the errlog file.

    """
    params = _server_params(tmp_path / "smoke.sqlite")
    errlog_path = tmp_path / "server-errlog.txt"

    try:
        with errlog_path.open("w", encoding="utf-8") as errlog:
            async with (
                stdio_client(params, errlog=errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                init_result = await session.initialize()
                assert init_result.server_info.name

                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                assert names == EXPECTED_TOOL_NAMES
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - sandbox guard
        pytest.skip(f"stdio subprocess could not be spawned in this environment: {exc}")

    errlog_text = errlog_path.read_text(encoding="utf-8")
    assert "opening the store from ENGRAVA_DB_PATH" in errlog_text
