"""Model Context Protocol server for engrava.

Exposes engrava's public read API as MCP tools over stdio, so MCP-aware
agents can fetch thoughts, search memory, and run structured queries.
The server is a standalone API consumer — it registers no engrava hooks,
manifests, or extensions.

Run it with ``python -m engrava_mcp`` or the ``engrava-mcp`` console
script.
"""

from __future__ import annotations

from engrava_mcp._compat import incompatible_engrava_message

try:
    from engrava_mcp.server import build_server, main
except ImportError as exc:  # pragma: no cover - raw trigger is un-triggerable in-process
    # The server consumes engrava's PUBLIC API. A missing symbol here means an
    # incompatible `engrava` is installed. We check API *presence*, not a
    # version number, on purpose: engrava's source version lags its released
    # version (a dev/editable install reports the pre-release number), so a
    # numeric `>=0.5` assert would false-positive. Re-raise with an actionable
    # message and preserve the original error. The message logic itself is
    # covered by tests in ``tests/test_compat.py``.
    raise ImportError(incompatible_engrava_message(exc)) from exc

__all__ = ["build_server", "main"]
