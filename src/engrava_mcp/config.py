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
    hybrid search runs without its vector arm.  Search itself runs under
    engrava's default search policy, and no ``hooks_class`` is configured,
    so on-store extension hooks are not attached on this launch (see
    :func:`_resolve_from_db_path`).

:func:`resolve_store` returns a :class:`ResolvedStore` that bundles the
store with an :meth:`~ResolvedStore.aclose` coroutine.  Closing the
``ResolvedStore`` always releases the underlying connection regardless of
which resolution path produced it, so callers never depend on store
connection-ownership internals.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import aiosqlite
from engrava import SearchConfig, SqliteEngravaCore, load_config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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

#: Entry-point group through which engrava extensions advertise themselves.
#: An extension that hooks the store is wired through the engrava config's
#: ``hooks_class``, so it can only be attached on the :data:`CONFIG_ENV_VAR`
#: launch.  Detection is generic over this group: engrava-mcp never imports,
#: names, or depends on any particular extension.
EXTENSIONS_ENTRY_POINT_GROUP = "engrava.extensions"

#: Logged when extension discovery itself fails.  Discovery is a diagnostic, so
#: it degrades to "nothing detected" rather than failing an otherwise working
#: launch; this line keeps that degradation from being silent in its turn.
_EXTENSION_DISCOVERY_FAILED = (
    "Could not read installed-package metadata, so this launch cannot report "
    "which engrava extensions are advertised: %s"
)


def _warn_softly(message: str, *args: object) -> None:
    """Emit a startup warning without letting the diagnostic break the launch.

    Used by the two extension diagnostics only, deliberately.  They are pure
    advice — they explain a working configuration, never a broken one — so
    neither is worth failing a resolution that has otherwise succeeded, and an
    operator who cannot be told about an unwired extension is better off than
    one whose server refuses to start over it.  Handlers and filters are
    supplied by the embedding application and can raise, so emission is
    attempted rather than assumed.

    :data:`_NO_PROVIDER_WARNING` deliberately does **not** go through here: it
    predates these diagnostics and an embedding application may be relying on
    its emission failing loudly.  Changing that is not this function's business.

    Delivery is therefore **best effort**.  What this guarantees is narrow and
    worth stating exactly: an :class:`Exception` raised while emitting cannot
    fail a resolution that has otherwise succeeded.  It does not guarantee that
    a line reaches anyone — that depends on a logging configuration this process
    does not own.

    Args:
        message: Warning message, possibly carrying ``%``-style placeholders.
        args: Values for those placeholders, formatted lazily by ``logging``.

    """
    # Suppressed rather than logged, because the logging channel is what
    # failed: there is nowhere left to report it to.
    with contextlib.suppress(Exception):
        logger.warning(message, *args)


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


class _ExtensionScan(NamedTuple):
    """The outcome of reading the extensions entry-point group.

    Attributes:
        names: Advertised extension names, sorted.  Empty when none is
            advertised or the read failed.
        read_failure: The error that stopped the read, or ``None``.

    """

    names: list[str]
    read_failure: Exception | None


def _scan_advertised_extensions() -> _ExtensionScan:
    """Name the engrava extensions installed packages advertise.

    Only the entry point's *presence* is read: taking ``name`` off the metadata
    is the whole detection, so engrava-mcp itself never calls ``load()``, reads
    the entry point's target, or imports it.  That keeps this server uncoupled
    from any extension's shape.

    The promise is bounded to what this code does, not to what happens while it
    reads: enumerating metadata runs whatever distribution finders are
    installed, and a finder-supplied ``name`` is free to execute anything it
    likes.  What is guaranteed is that engrava-mcp never reaches for the entry
    point's target.

    It also bounds what can be *concluded*: this reports what distributions
    advertise, which is not the same as an extension that is importable, or
    that hooks the store at all.

    Every :class:`Exception` from the read is captured and returned rather than
    propagated.  Reading distribution metadata can fail on a broken install or
    an unusual distribution finder, and a diagnostic must not take down a launch
    that is otherwise working.  The guard covers the whole read — the lookup,
    iterating what it returns, taking each name, and coercing it to text —
    because a custom finder can raise at any of those points, not only at the
    call.  The coercion is what lets callers treat the result as plain strings:
    a finder is free to hand back a name that is not one, and a diagnostic must
    not be the thing that fails on it.

    Nothing is logged here.  The caller decides when to report, which is what
    keeps a failed launch as quiet as it was before this diagnostic existed.

    Returns:
        The scan outcome.  Names are sorted, so a report does not depend on
        metadata iteration order.

    """
    try:
        names = sorted(
            str(entry_point.name)
            for entry_point in importlib.metadata.entry_points(group=EXTENSIONS_ENTRY_POINT_GROUP)
        )
    except Exception as exc:  # noqa: BLE001 - see above; BaseException is deliberately not caught
        return _ExtensionScan(names=[], read_failure=exc)
    return _ExtensionScan(names=names, read_failure=None)


def _report_extensions(scan: _ExtensionScan) -> None:
    """Report a scan, if there is anything to say about it.

    Args:
        scan: The outcome of :func:`_scan_advertised_extensions`.

    """
    if scan.read_failure is not None:
        _warn_softly(_EXTENSION_DISCOVERY_FAILED, scan.read_failure)
    elif scan.names:
        _warn_softly(_unwired_extensions_warning(scan.names))


def _unwired_extensions_warning(names: Sequence[str]) -> str:
    """Build the warning for extensions this launch cannot wire.

    Deliberately about the *launch*, not about the extensions: entry-point
    metadata says a distribution advertises an extension, not what it does.  So
    the message says "advertise", says this mode is intentional rather than
    broken, and tells an operator who wanted nothing from extensions that there
    is nothing to do.

    Args:
        names: Advertised names of the installed extensions.

    Returns:
        A message naming them, the launch that will not wire them, and the
        launch that can.

    """
    return (
        f"Installed packages advertise Engrava extension(s): {', '.join(names)}. "
        f"{DB_PATH_ENV_VAR} intentionally does not configure store hooks; to "
        "enable an extension's store hooks, launch with "
        f"{CONFIG_ENV_VAR} and an engrava.yaml setting hooks.class. "
        "Otherwise no action is needed."
    )


async def _configure_connection(connection: aiosqlite.Connection) -> None:
    """Bring a freshly opened connection to the state store resolution needs.

    Applies the concurrency pragmas that put the connection at parity with
    :meth:`SqliteEngravaCore.from_config` — WAL journal, enforced foreign keys,
    ``busy_timeout`` so a contended write waits for the lock instead of failing
    immediately, and ``synchronous=NORMAL`` (the safe, faster WAL setting) — and
    installs the row factory the store expects.

    Factored out of :func:`_resolve_from_db_path` so a caller that needs a
    connection prepared *identically* to the resolved one gets it from here
    rather than re-listing the pragmas, which would drift.

    Args:
        connection: The connection to configure, in place.

    """
    await connection.execute("PRAGMA journal_mode=WAL")
    await connection.execute("PRAGMA foreign_keys=ON")
    await connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    await connection.execute("PRAGMA synchronous=NORMAL")
    connection.row_factory = aiosqlite.Row


async def _resolve_from_db_path(db_path: str) -> ResolvedStore:
    """Open a database file and build a store over it.

    The connection is configured by :func:`_configure_connection`.  The
    bare-database path configures no embedding provider, so semantic search is
    inert and a warning is logged.

    It also carries no ``hooks_class``, so an installed engrava extension that
    hooks the store cannot be attached here and would otherwise sit inert with
    no signal at all.  When extension metadata can be read and advertises any
    extension, a second warning names it and points at the launch where a store
    hook can be configured; when reading it fails, the failure is reported
    instead — :func:`_scan_advertised_extensions` states exactly which failures
    that covers and :func:`_warn_softly` what "reported" is worth.  The scan runs
    before the connection is opened and is reported only once the store is
    built, so a launch that fails on the way to a store says nothing about
    extensions — exactly as it said nothing before this diagnostic existed.

    This path deliberately does **not** wire hooks itself: it has no
    configuration channel, and giving it one is a different change.

    **"Zero-config" here means engrava's default search policy**, not "a store
    handed no search configuration".  The two are not the same: a store built
    without a :class:`~engrava.SearchConfig` resolves its recency fusion weight
    to ``0.0``, which silently disables the transaction-time recency signal that
    ``search_memory``'s ``recency_now`` argument selects — the argument would be
    accepted and have no effect.  Passing a default ``SearchConfig`` gives this
    launch the search policy an ``engrava.yaml`` declaring no ``search`` section
    resolves to.  It equalises the *policy* only: a yaml that also configures an
    embedding provider or a vector backend can rank differently, because this
    launch configures neither.

    Args:
        db_path: Filesystem path to a SQLite database file.

    Returns:
        A :class:`ResolvedStore` whose cleanup closes the opened
        connection.

    """
    # Scanned before the connection is opened, not after. The scan needs nothing
    # from the store, and doing it here keeps it outside the window where a
    # connection exists that no caller can yet close — so the cleanup below is
    # exactly the one this path always had.
    scan = _scan_advertised_extensions()
    connection = await aiosqlite.connect(str(Path(db_path)))
    try:
        await _configure_connection(connection)
        store = SqliteEngravaCore(connection, search_config=SearchConfig())
        await store.ensure_schema()
    except Exception:
        await connection.close()
        raise
    logger.warning(_NO_PROVIDER_WARNING)
    # Guarded, and guarded here rather than by widening the block above. Between
    # a successful build and the return, the connection exists and no caller can
    # close it; anything unwinding through this line would strand it. That is a
    # pre-existing property of this window — the warning above has always sat in
    # it — and changing that is not this function's business. Not reproducing it
    # on the line this change adds is.
    try:
        _report_extensions(scan)
    except BaseException:
        # An ordinary failure to close is suppressed so it cannot replace the
        # exception the caller is actually being told about — a cancellation is
        # more useful to them than "the close failed too".
        with contextlib.suppress(Exception):
            await connection.close()
        raise
    return ResolvedStore(store=store, _closer=connection.close)
