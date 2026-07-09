"""Store resolution for the engrava MCP server.

The MCP server is a standalone process that wraps engrava's public async
API.  It resolves a :class:`~engrava.SqliteEngravaCore` from environment
variables so the same server entry point can target either a fully
configured deployment (``engrava.yaml``) or a bare database file.

Two environment variables are recognised, in priority order:

``ENGRAVA_MCP_CONFIG``
    Path to an ``engrava.yaml`` file.  When set, the store is built with
    :meth:`SqliteEngravaCore.from_config`, which applies the configured
    embedding provider, vector backend, journal, and TTL settings.

``ENGRAVA_DB_PATH``
    Path to a SQLite database file.  When set (and ``ENGRAVA_MCP_CONFIG``
    is not), a connection is opened directly and the core schema is
    ensured.  No embedding provider or vector backend is configured, so
    hybrid search degrades to its lexical backend.

:func:`resolve_store` returns a :class:`ResolvedStore` that bundles the
store with an :meth:`~ResolvedStore.aclose` coroutine.  Closing the
``ResolvedStore`` always releases the underlying connection regardless of
which resolution path produced it, so callers never depend on store
connection-ownership internals.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
from engrava import SqliteEngravaCore, load_config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

#: Module logger.  Startup diagnostics (e.g. the degraded-configuration
#: warning) are emitted through this so an operator sees why semantic search
#: is inert without it being fatal.
logger = logging.getLogger("engrava_mcp")

#: Environment variable naming an ``engrava.yaml`` config file.
CONFIG_ENV_VAR = "ENGRAVA_MCP_CONFIG"

#: Environment variable naming a bare SQLite database file.
DB_PATH_ENV_VAR = "ENGRAVA_DB_PATH"

#: SQLite ``busy_timeout`` (milliseconds) applied to the bare-``ENGRAVA_DB_PATH``
#: connection so a contended write waits for the lock instead of failing
#: immediately with "database is locked".  Mirrors the value
#: :meth:`SqliteEngravaCore.from_config` applies.
BUSY_TIMEOUT_MS = 5000

#: Message logged at startup when no embedding provider is resolved, so the
#: operator understands that semantic (vector) search is inert and how to
#: enable it.  Lexical FTS, the graph, MindQL, and the audit trail are
#: unaffected.
_NO_PROVIDER_WARNING = (
    "No embedding provider configured: semantic (vector) search is inert; "
    "queries fall back to lexical full-text search. Full-text search, the "
    "graph, MindQL, and the audit trail are unaffected. To enable semantic "
    "search, declare an embedding provider in an engrava.yaml and point "
    f"{CONFIG_ENV_VAR} at it."
)


class StoreResolutionError(RuntimeError):
    """Raised when no store can be resolved from the environment.

    Args:
        message: Human-readable description of the resolution failure.

    """


@dataclass(frozen=True)
class ResolvedStore:
    """A resolved store paired with its connection-cleanup coroutine.

    Attributes:
        store: The schema-ready ``SqliteEngravaCore`` to serve queries.
        _closer: Async callback that releases the underlying connection.

    """

    store: SqliteEngravaCore
    _closer: Callable[[], Awaitable[None]]

    async def aclose(self) -> None:
        """Release the underlying database connection."""
        await self._closer()


async def resolve_store() -> ResolvedStore:
    """Resolve a store from the environment.

    Resolution honours :data:`CONFIG_ENV_VAR` first, then
    :data:`DB_PATH_ENV_VAR`.

    Returns:
        A :class:`ResolvedStore` whose connection is released by
        :meth:`ResolvedStore.aclose`.

    Raises:
        StoreResolutionError: If neither environment variable is set.
        ConfigError: If the configured ``engrava.yaml`` is invalid.

    """
    config_path = os.environ.get(CONFIG_ENV_VAR)
    if config_path:
        config = load_config(config_path)
        if config.embeddings is None or config.embeddings.provider is None:
            logger.warning(_NO_PROVIDER_WARNING)
        store = await SqliteEngravaCore.from_config(config_path)
        return ResolvedStore(store=store, _closer=store.close)

    db_path = os.environ.get(DB_PATH_ENV_VAR)
    if db_path:
        return await _resolve_from_db_path(db_path)

    msg = (
        "No engrava store configured. Set "
        f"{CONFIG_ENV_VAR} to an engrava.yaml path or "
        f"{DB_PATH_ENV_VAR} to a SQLite database path."
    )
    raise StoreResolutionError(msg)


async def _resolve_from_db_path(db_path: str) -> ResolvedStore:
    """Open a database file and build a store over it.

    The connection is brought to parity with
    :meth:`SqliteEngravaCore.from_config`'s concurrency pragmas: WAL journal,
    enforced foreign keys, ``busy_timeout`` so a contended write waits for the
    lock instead of failing immediately, and ``synchronous=NORMAL`` (the safe,
    faster WAL setting).  The bare-database path configures no embedding
    provider, so semantic search is inert and a warning is logged.

    Args:
        db_path: Filesystem path to a SQLite database file.

    Returns:
        A :class:`ResolvedStore` whose cleanup closes the opened
        connection.

    """
    connection = await aiosqlite.connect(str(Path(db_path)))
    try:
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        await connection.execute("PRAGMA synchronous=NORMAL")
        connection.row_factory = aiosqlite.Row
        store = SqliteEngravaCore(connection)
        await store.ensure_schema()
    except Exception:
        await connection.close()
        raise
    logger.warning(_NO_PROVIDER_WARNING)
    return ResolvedStore(store=store, _closer=connection.close)
