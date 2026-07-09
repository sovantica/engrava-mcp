"""``python -m engrava_mcp`` entry point.

Delegates to :func:`engrava_mcp.server.main`, which builds the MCP server
and serves it over stdio.
"""

from __future__ import annotations

from engrava_mcp.server import main

if __name__ == "__main__":
    main()
