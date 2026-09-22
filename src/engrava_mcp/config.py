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
``ResolvedStore`` closes the store and, on the ``ENGRAVA_DB_PATH`` route, then
the connection this server opened for it (on the other route the store owns its
connection), so callers need not depend on store connection-ownership
internals.  What happens when that close does not go cleanly is described on
:meth:`ResolvedStore.aclose`.
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
import anyio
from engrava import ConnectionQuarantinedError, SearchConfig, SqliteEngravaCore, load_config

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
#: enable it.  Lexical FTS, the graph, and MindQL are unaffected.  The audit
#: trail is a separate knob (``journal: enabled: true`` in an
#: ``engrava.yaml``) this message deliberately does not claim either way:
#: it fires on the bare :data:`DB_PATH_ENV_VAR` route too, and that route
#: never has a journal at all.
_NO_PROVIDER_WARNING = (
    "No embedding provider configured: semantic (vector) search is inert; "
    "queries fall back to lexical full-text search. Full-text search, the "
    "graph, and MindQL are unaffected. To enable semantic search, declare "
    "an embedding provider in an engrava.yaml and point "
    f"{CONFIG_ENV_VAR} at it."
)

#: How long the connection close is waited on, in seconds.  Equal to the
#: library's default close bound.
_CONNECTION_CLOSE_TIMEOUT_SECONDS = 30.0

#: Logged when the store's close raised
#: :class:`~engrava.ConnectionQuarantinedError`.  The reason is the library's own
#: description of why the connection was quarantined, and is logged as given.
_STORE_QUARANTINED_AT_SHUTDOWN = "Closing the store reported a quarantined connection: %s"

#: Logged when the connection close did not finish within the bound.
_CONNECTION_CLOSE_TIMED_OUT = (
    "The database connection did not close within %s seconds; abandoning it."
)

#: Logged when closing the connection raised while the store's own close was failing.
_CONNECTION_CLOSE_FAILED = "Database connection cleanup also raised after the store close failed."

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


def _warn_softly(message: str, *args: object, exc_info: bool = False) -> None:
    """Emit a warning without letting the diagnostic change the outcome.

    Used by the two extension diagnostics and by the shutdown warnings,
    deliberately.  The extension diagnostics are non-fatal: they report
    extension wiring or discovery without invalidating a store resolution that
    has otherwise succeeded.  Neither is worth failing a resolution, and an
    operator who cannot be told about an unwired extension is better off than
    one whose server refuses to start over it.  The shutdown warnings are used
    for the same reason: a diagnostic is not worth changing an outcome.
    Handlers and filters are supplied by the embedding application and can
    raise, so emission is attempted rather than assumed.

    :data:`_NO_PROVIDER_WARNING` deliberately does **not** go through here: it
    predates these diagnostics and an embedding application may be relying on
    its emission failing loudly.  Changing that is not this function's business.

    Delivery is therefore **best effort**.  What this guarantees is narrow and
    worth stating exactly: an :class:`Exception` raised while emitting cannot
    change the outcome of the operation that emitted the warning.  It does not
    guarantee that a line reaches anyone — that depends on a logging
    configuration this process does not own.

    Args:
        message: Warning message, possibly carrying ``%``-style placeholders.
        args: Values for those placeholders, formatted lazily by ``logging``.
        exc_info: Attach the exception being handled to the record, as
            ``logger.warning(..., exc_info=True)`` does.

    """
    # Suppressed rather than logged, because the logging channel is what
    # failed: there is nowhere left to report it to.
    with contextlib.suppress(Exception):
        logger.warning(message, *args, exc_info=exc_info)


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
        _closer: Async callback that closes the store and, on the
            :data:`DB_PATH_ENV_VAR` launch, then the connection this server
            opened for it.  Reports whether that connection close abandoned a
            wedged worker rather than closing cleanly (see :meth:`aclose`).

    """

    store: SqliteEngravaCore
    _closer: Callable[[], Awaitable[bool]]

    async def aclose(self) -> bool:
        """Close the store, then the underlying database connection.

        On the :data:`DB_PATH_ENV_VAR` launch the store is built around a
        connection this server opened and the store does not own, so the
        connection is closed here, after the store.  Its close is attempted
        whichever way the store's close ends, and that attempt is bounded: when
        the bound expires the connection is abandoned, a warning is logged, and
        this method stops waiting for it.  On the :data:`CONFIG_ENV_VAR` launch
        the store owns its connection and closing the store is all this method
        does.

        If closing the store raises
        :class:`~engrava.ConnectionQuarantinedError`, a best-effort warning
        carrying the library's reason is emitted and this method returns
        ``False`` instead of raising it -- but only when that close does not
        leave a wedged worker thread behind.  On the :data:`CONFIG_ENV_VAR`
        launch it can: the store owns its connection there, so its own close
        applies its own internal bound on the very same worker, and the
        library's own :class:`~engrava.ConnectionQuarantinedError` raised
        *from that close* is verified (at the library's own source) to mean
        exactly that bound expiring with the physical close still not done --
        never a benign quarantine for some unrelated reason.  That case is
        caught and reported ``True`` before it ever reaches this method (see
        :func:`resolve_store`'s own closer for the :data:`CONFIG_ENV_VAR`
        launch); what reaches this ``except`` clause is therefore only ever
        the :data:`DB_PATH_ENV_VAR` launch's own residual case, where the
        store never owned the connection this server is asking about at all,
        so a quarantine reported here carries no relationship to it.

        Warnings are best effort: an :class:`Exception` raised while emitting
        one is dropped and changes nothing else about the outcome.

        Returns:
            Whether the connection close abandoned a worker thread that never
            answered within its bound, rather than closing cleanly -- either
            :data:`_CONNECTION_CLOSE_TIMEOUT_SECONDS` on the
            :data:`DB_PATH_ENV_VAR` launch, or the store's own
            ``close_timeout_seconds`` on the :data:`CONFIG_ENV_VAR` launch.
            ``False`` on a clean close, and on the
            :class:`~engrava.ConnectionQuarantinedError` path above (the
            :data:`DB_PATH_ENV_VAR` launch's residual case only -- see it for
            why).  This is the fact ``server.py``'s ``lifespan`` uses to
            decide whether the worker is genuinely wedged and the process
            must force its own exit — callers that do not need that decision
            (every test in ``tests/test_shutdown.py`` that calls this method
            directly) are free to ignore it.

        Raises:
            Exception: Anything the store's close raises other than
                :class:`~engrava.ConnectionQuarantinedError`; and, when the
                store closed cleanly, anything the connection's own close
                raises.  If the connection's close also raises an
                :class:`Exception` while the store's close is failing, that
                exception does not replace the store's; anything else it
                raises does.

        """
        try:
            return await self._closer()
        except ConnectionQuarantinedError as exc:
            _warn_softly(_STORE_QUARANTINED_AT_SHUTDOWN, exc.reason)
            return False


async def resolve_store() -> ResolvedStore:
    """Resolve a store from the environment.

    Resolution honours :data:`CONFIG_ENV_VAR` first, then
    :data:`DB_PATH_ENV_VAR`.

    Returns:
        A :class:`ResolvedStore` whose :meth:`~ResolvedStore.aclose` closes the
        store and, on the :data:`DB_PATH_ENV_VAR` launch, then the connection
        this server opened for it.

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

        async def _closer() -> bool:
            """Close the store, reporting whether its own bound abandoned a wedged worker.

            The store owns its connection here, so ``store.close()`` performs
            the whole close, including its own internal bound on that same
            non-daemon aiosqlite worker thread.  Verified at the library's own
            source (``SqliteEngravaCore.close`` / ``_finish_close_wait``): the
            *only* place ``close()`` raises
            :class:`~engrava.ConnectionQuarantinedError` from within itself is
            when its own wait for the physical close exceeds
            ``close_timeout_seconds`` with that task still not done -- the
            task is never cancelled, only abandoned, so it keeps running (or
            not) in the background exactly like :func:`_close_connection`
            leaves the :data:`DB_PATH_ENV_VAR` route's own connection.  There
            is therefore no other, benign reason for ``close()`` itself to
            raise this here, and it is caught and reported as a wedge rather
            than left to :meth:`ResolvedStore.aclose`'s own catch, which is
            the correct default only for a caller that cannot make this same
            guarantee (see the :data:`DB_PATH_ENV_VAR` route, where
            ``store.close()`` never owns the connection at all).

            Returns:
                ``True`` when ``store.close()`` raised
                :class:`~engrava.ConnectionQuarantinedError` -- always a report
                of its own bound expiring with the worker not answering.
                ``False`` on a clean close.

            """
            try:
                await store.close()
            except ConnectionQuarantinedError as exc:
                _warn_softly(_STORE_QUARANTINED_AT_SHUTDOWN, exc.reason)
                return True
            return False

        return ResolvedStore(store=store, _closer=_closer)

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


async def _close_connection(connection: aiosqlite.Connection) -> bool:
    """Close a connection, giving up the wait after the bound.

    The bound is :data:`_CONNECTION_CLOSE_TIMEOUT_SECONDS`.  When it expires a
    warning is logged and the connection is abandoned.  Anything the close
    itself raises propagates.

    Args:
        connection: The connection to close.

    Returns:
        Whether the bound expired before the close finished -- an abandoned
        wait on a worker that never answered, not a clean close.  The caller
        decides what abandoning it means; this function only reports whether
        it happened.

    """
    with anyio.move_on_after(_CONNECTION_CLOSE_TIMEOUT_SECONDS) as scope:
        await connection.close()
    if scope.cancelled_caught:
        _warn_softly(_CONNECTION_CLOSE_TIMED_OUT, _CONNECTION_CLOSE_TIMEOUT_SECONDS)
    return scope.cancelled_caught


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

    async def _closer() -> bool:
        """Close the store, then the connection this server opened for it.

        The connection goes second so a pending flush still has an open
        connection to write through. An attempt to close it is made whichever
        way the store's close ends, because the store's close will not: the
        constructor above never marks the store as owning it. With engrava 0.7
        the store's close does nothing on this route (it neither owns the
        connection nor tracks accesses), so the cleanup is written not to depend
        on that. The attempt is bounded; when the bound expires the connection
        is abandoned with a warning. If the store's close raised and the
        connection close then raises an :class:`Exception`, that is reported in
        a warning and the store's exception is what ``_closer`` raises; anything
        else the connection close raises propagates instead.

        Returns:
            Whether the connection close abandoned the connection at its bound
            (see :func:`_close_connection`).  Not reported when the store's own
            close raises -- this function re-raises that instead, and
            :meth:`ResolvedStore.aclose` decides what to report for that case.

        """
        try:
            await store.close()
        except BaseException:
            # The store's exception must not be replaced by an ordinary cleanup failure.
            try:
                await _close_connection(connection)
            except Exception:  # noqa: BLE001 - reported below; the store's error is re-raised
                _warn_softly(_CONNECTION_CLOSE_FAILED, exc_info=True)
            raise
        return await _close_connection(connection)

    return ResolvedStore(store=store, _closer=_closer)
