"""An in-process MCP client connection, matching the shape the 1.x test harness used.

``mcp`` 1.x exposed ``mcp.shared.memory.create_connected_server_and_client_session``: an
async context manager wiring an in-memory client straight to a ``FastMCP``/``Server``
instance, with no transport in between.  That helper does not exist in 2.x — the module
now exports ``create_client_server_memory_streams`` and lower-level stream primitives,
nothing that serves the same purpose.

2.x's replacement is :class:`mcp.Client`: passed an :class:`~mcp.server.mcpserver.MCPServer`
instance directly, it drives the connection in-process (no JSON-RPC framing in the default
``"auto"`` mode) and exposes the same call surface the tests already drove —
``call_tool``, ``list_tools``, ``get_prompt``, ``list_resources``, and so on.

This module is the one place that names the seam, so every test file imports
:func:`connect_client` from here rather than aliasing the SDK class locally. A bare
``from mcp import Client as connect_client`` in each of the nine call sites would read as
importing a helper *function* (matching the old helper's name and call shape), but it is
a class: ruff's ``N813`` flags a CamelCase name imported under a lowercase alias, and every
call site that annotated its return type as ``ClientSession`` (the 1.x helper's yield type)
would be typed wrong under mypy, since a connected :class:`~mcp.Client` is not a
``ClientSession``. A thin wrapper function keeps the import ruff-clean, gives callers a
correctly-typed annotation to reach for (``Client``), and means a further harness change
touches one file instead of nine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import Client

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


def connect_client(server: MCPServer) -> Client:
    """Connect an in-process MCP client to ``server``.

    Args:
        server: The server instance to connect to, in-process — no transport,
            no subprocess.

    Returns:
        A :class:`~mcp.Client`, used as an async context manager:
        ``async with connect_client(server) as client: ...``. Exposes the same
        call surface the tests drove through the 1.x ``ClientSession`` —
        ``call_tool``, ``list_tools``, ``get_prompt``, ``list_prompts``,
        ``list_resources``, ``list_resource_templates``, ``read_resource``.

    """
    return Client(server)
