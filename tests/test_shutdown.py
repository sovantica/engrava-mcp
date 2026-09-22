"""Shutdown when the store's own close does not end cleanly.

The tests make ``SqliteEngravaCore.close()`` raise
:class:`~engrava.ConnectionQuarantinedError` (the error the library raises when
the database worker does not answer) and make the connection's close hang or
raise.  A worker that does not answer also holds up anything else queued behind
it.  The server's shutdown path has to handle each: it must still attempt to
close the connection it opened; a quarantined store is logged, not raised; and it
must not wait on an unresponsive worker without a bound.

Observation is done on the real objects rather than on private state.  A spy
wraps ``aiosqlite.Connection.close`` and ``SqliteEngravaCore.close`` — recording
when each runs and delegating to the genuine implementation unless a test
substitutes a behaviour — so where a test leaves the connection's close alone,
"the connection was closed" means the real close ran to completion.

**Guarding the guards.**  A test that exercises a bound must fail, not hang, when
the bound is gone.  Those tests run under an outer :func:`anyio.fail_after`, and
it is anyio's rather than ``asyncio.wait_for`` on purpose: in aiosqlite 0.22
``close`` awaits a second future in a ``finally`` block, which a single asyncio
cancellation does not interrupt, whereas anyio re-delivers its cancellation until
the scope exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import TYPE_CHECKING

import aiosqlite
import anyio
import pytest
from engrava import ConnectionQuarantinedError, SqliteEngravaCore

from engrava_mcp import build_server, config
from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR, resolve_store
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
    from pathlib import Path

#: Outer bound on the tests that exercise the connection-close bound, so a
#: regression that removes it fails the test instead of hanging the suite.
GUARD_SECONDS = 3.0

#: The bound the tests substitute for the production one, short to keep the
#: suite quick.
SHORT_BOUND_SECONDS = 0.05

#: The reason the tests give a quarantined store, and look for in the log.
QUARANTINE_REASON = "the database worker did not answer the close"

CONNECTION_CLOSE = "connection.close"
STORE_CLOSE = "store.close"


class ShutdownRecorder:
    """Records the closes a shutdown performs, and lets a test change how they end.

    Attributes:
        events: ``"store.close"`` when the store's close finished (however it
            ended) and ``"connection.close"`` when a connection's close began,
            in the order they happened.
        completed_connection_closes: How many connection closes ran to completion.
        connections: Every connection opened through ``aiosqlite.connect`` while
            the recorder was installed.

    """

    def __init__(
        self,
        real_store_close: Callable[[SqliteEngravaCore], Awaitable[None]],
        real_connection_close: Callable[[aiosqlite.Connection], Awaitable[None]],
    ) -> None:
        self.events: list[str] = []
        self.completed_connection_closes = 0
        self.connections: list[aiosqlite.Connection] = []
        self._store_close = real_store_close
        self._connection_close = real_connection_close

    def store_close_raises(self, error: BaseException) -> None:
        """Make every store close raise ``error`` instead of closing.

        Args:
            error: The exception the store's close raises.

        """

        async def _raise(_store: SqliteEngravaCore) -> None:
            raise error

        self._store_close = _raise

    def connection_close_never_completes(self) -> None:
        """Make every connection close wait forever instead of closing."""

        async def _never(_connection: aiosqlite.Connection) -> None:
            await asyncio.Event().wait()

        self._connection_close = _never

    def connection_close_raises(self, error: BaseException) -> None:
        """Make every connection close raise ``error`` instead of closing.

        Args:
            error: The exception a connection's close raises.

        """

        async def _raise(_connection: aiosqlite.Connection) -> None:
            raise error

        self._connection_close = _raise

    async def run_store_close(self, store: SqliteEngravaCore) -> None:
        """Run the store's close as currently configured, recording its end.

        Args:
            store: The store being closed.

        """
        try:
            await self._store_close(store)
        finally:
            self.events.append(STORE_CLOSE)

    async def run_connection_close(self, connection: aiosqlite.Connection) -> None:
        """Run a connection's close as currently configured, recording it.

        Args:
            connection: The connection being closed.

        """
        self.events.append(CONNECTION_CLOSE)
        await self._connection_close(connection)
        self.completed_connection_closes += 1


@pytest.fixture
async def recorder(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ShutdownRecorder]:
    """Install the close spies, then close whatever a test leaves open.

    A test that makes a close raise or hang leaves the real connection open, and
    an open aiosqlite connection keeps a non-daemon worker thread alive, so the
    teardown closes every connection this fixture saw, for real.

    Args:
        monkeypatch: Fixture used to install the spies.

    Yields:
        The recorder observing this test's closes.

    """
    real_connect = aiosqlite.connect
    real_connection_close = aiosqlite.Connection.close
    real_store_close = SqliteEngravaCore.close
    spy = ShutdownRecorder(real_store_close, real_connection_close)

    def _tracking_connect(database: str) -> aiosqlite.Connection:
        connection = real_connect(database)
        spy.connections.append(connection)
        return connection

    # Class attributes, so each is a plain function that receives the instance;
    # a bound method of the recorder would not.
    async def _spied_connection_close(connection: aiosqlite.Connection) -> None:
        await spy.run_connection_close(connection)

    async def _spied_store_close(store: SqliteEngravaCore) -> None:
        await spy.run_store_close(store)

    monkeypatch.setattr(aiosqlite, "connect", _tracking_connect)
    monkeypatch.setattr(aiosqlite.Connection, "close", _spied_connection_close)
    monkeypatch.setattr(SqliteEngravaCore, "close", _spied_store_close)
    try:
        yield spy
    finally:
        for connection in spy.connections:
            await real_connection_close(connection)


def select_db_path_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point store resolution at a bare database file (``ENGRAVA_DB_PATH``).

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Directory the database file is created in.

    """
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "shutdown.sqlite"))


def select_yaml_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point store resolution at an ``engrava.yaml`` (``ENGRAVA_MCP_CONFIG``).

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Directory the config and database files are created in.

    """
    config_file = tmp_path / "engrava.yaml"
    config_file.write_text(
        f"database:\n  path: {tmp_path / 'shutdown.sqlite'}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))


@pytest.fixture(params=["bare-database", "engrava-yaml"])
def launch_route(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> str:
    """Point store resolution at one of the server's two launch routes.

    Args:
        request: Supplies the route name being exercised.
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the database and config files.

    Returns:
        The name of the route the environment now selects.

    """
    route = str(request.param)
    if route == "bare-database":
        select_db_path_launch(monkeypatch, tmp_path)
    else:
        select_yaml_launch(monkeypatch, tmp_path)
    return route


async def serve_then_leave() -> None:
    """Build the server, connect a client, and leave the client's context.

    Leaving the context ends the server's lifespan, so an exception raised by the
    lifespan's teardown is seen from here.
    """
    server = build_server()
    async with connect_client(server) as client:
        await client.list_tools()


def shutdown_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Pick out what this package logged at WARNING or above since the last clear.

    Args:
        caplog: The log capture fixture.

    Returns:
        The matching records, oldest first.

    """
    return [
        record
        for record in caplog.records
        if record.name == config.logger.name and record.levelno >= logging.WARNING
    ]


@contextlib.contextmanager
def warnings_that_raise() -> Iterator[None]:
    """Make logging a warning through this package's logger raise, inside the block.

    Handlers and filters belong to the embedding application and can raise, so a
    warning the shutdown path logs is an operation that can itself fail.  The
    failure is injected as a logger filter, which runs inside the ``logger.warning``
    call.  Every warning raises, so none of the shutdown warnings (a quarantined
    store, an abandoned close, a failed close) gets through.

    Yields:
        Nothing; the filter is removed when the block ends.

    """

    def _raise(record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            msg = "the logging channel is broken"
            raise RuntimeError(msg)
        return True

    config.logger.addFilter(_raise)
    try:
        yield
    finally:
        config.logger.removeFilter(_raise)


class TestQuarantinedStoreAtShutdown:
    """A :class:`~engrava.ConnectionQuarantinedError` reported by the store's close is logged
    on a best-effort basis, not raised.
    """

    async def test_bare_database_launch_leaves_cleanly(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))

        # Nothing to assert on: the test fails if leaving the client raises.
        await serve_then_leave()

    async def test_bare_database_launch_still_closes_its_connection(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))

        # Whether the exit is clean is the sibling test's question; this one is
        # only about the connection, so an error from leaving is set aside.
        with contextlib.suppress(Exception):
            await serve_then_leave()

        assert recorder.completed_connection_closes == 1
        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]

    async def test_yaml_launch_leaves_cleanly(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_yaml_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))

        # The store owns its connection on this launch, so this server has no
        # connection of its own to close; the test fails if leaving the client
        # raises.
        await serve_then_leave()

    async def test_the_quarantine_reason_is_logged_once(
        self, recorder: ShutdownRecorder, launch_route: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()  # the launch's own startup warnings are not under test

            await resolved.aclose()

        records = shutdown_warnings(caplog)
        assert len(records) == 1, launch_route
        assert records[0].levelno == logging.WARNING
        assert QUARANTINE_REASON in records[0].getMessage()
        assert "quarantined" in records[0].getMessage()


class TestOtherFailuresAreNotSwallowed:
    """An error from the store's close other than a quarantine is raised, not logged away."""

    async def test_another_error_from_the_store_propagates(
        self, recorder: ShutdownRecorder, launch_route: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder.store_close_raises(RuntimeError("the disk went away"))
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            with pytest.raises(RuntimeError, match="the disk went away"):
                await resolved.aclose()

        # It is not reported as a quarantine either.
        assert shutdown_warnings(caplog) == [], launch_route

    async def test_the_bare_database_connection_is_closed_anyway(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(RuntimeError("the disk went away"))
        resolved = await resolve_store()

        with pytest.raises(RuntimeError, match="the disk went away"):
            await resolved.aclose()

        assert recorder.completed_connection_closes == 1
        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]


class TestConnectionCloseFailsToo:
    """An ordinary connection-close exception does not replace the store's exception."""

    async def test_the_stores_exception_is_the_one_raised(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(RuntimeError("the disk went away"))
        recorder.connection_close_raises(OSError("the worker vanished"))
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            with pytest.raises(RuntimeError, match="the disk went away"):
                await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
        (record,) = shutdown_warnings(caplog)
        assert "cleanup also raised" in record.getMessage()
        # The failure that was set aside is attached to the warning.
        assert record.exc_info is not None
        assert record.exc_info[0] is OSError

    async def test_a_quarantined_store_still_leaves_cleanly(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))
        recorder.connection_close_raises(OSError("the worker vanished"))
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
        messages = [record.getMessage() for record in shutdown_warnings(caplog)]
        assert len(messages) == 2
        assert any(QUARANTINE_REASON in message for message in messages)
        assert any("cleanup also raised" in message for message in messages)

    async def test_it_is_not_hidden_when_the_store_closed_cleanly(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.connection_close_raises(OSError("the worker vanished"))
        resolved = await resolve_store()

        with pytest.raises(OSError, match="the worker vanished"):
            await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]

    async def test_a_failing_log_call_does_not_replace_the_stores_exception(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "_CONNECTION_CLOSE_TIMEOUT_SECONDS", SHORT_BOUND_SECONDS)
        recorder.store_close_raises(ValueError("the store could not flush"))
        recorder.connection_close_never_completes()
        resolved = await resolve_store()

        with (
            warnings_that_raise(),
            pytest.raises(ValueError, match="the store could not flush"),
            anyio.fail_after(GUARD_SECONDS),
        ):
            await resolved.aclose()

        assert recorder.completed_connection_closes == 0
        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]


class TestOrdinaryShutdown:
    """A store that closes normally is closed first, then the connection, with no warning logged."""

    async def test_store_closes_first_then_the_connection_and_no_warning_is_logged(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            await resolved.aclose()

        assert shutdown_warnings(caplog) == []
        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
        assert recorder.completed_connection_closes == 1


class TestConnectionCloseIsBounded:
    """The wait for the connection to close has a bound."""

    async def test_a_close_that_never_completes_is_abandoned_and_logged(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "_CONNECTION_CLOSE_TIMEOUT_SECONDS", SHORT_BOUND_SECONDS)
        recorder.connection_close_never_completes()
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            with anyio.fail_after(GUARD_SECONDS):
                await resolved.aclose()

        assert recorder.completed_connection_closes == 0
        records = shutdown_warnings(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert "did not close" in records[0].getMessage()

    async def test_aclose_returns_once_the_bound_expires_on_a_wedged_worker(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The stand-in above shows the bound being applied; this shows the same
        # on a real aiosqlite connection, whose close (in aiosqlite 0.22) waits
        # on a worker thread and then awaits again in a ``finally`` block.  The
        # worker is wedged with a SQL function that blocks until the test
        # releases it.
        select_db_path_launch(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "_CONNECTION_CLOSE_TIMEOUT_SECONDS", SHORT_BOUND_SECONDS)
        threads_before = set(threading.enumerate())
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()
            (connection,) = recorder.connections
            (worker,) = set(threading.enumerate()) - threads_before
            wedged = threading.Event()
            release = threading.Event()

            def _block() -> int:
                wedged.set()
                release.wait()
                return 1

            await connection.create_function("block_worker", 0, _block)
            blocker = asyncio.ensure_future(connection.execute("SELECT block_worker()"))
            try:
                assert await asyncio.to_thread(wedged.wait, GUARD_SECONDS)

                with anyio.fail_after(GUARD_SECONDS):
                    await resolved.aclose()
            finally:
                # Let the worker finish and drain its queue, and wait for it to
                # exit, while the loop is still alive to receive its callbacks;
                # a worker that outlives the loop can raise on its last one.
                release.set()
                await blocker
                await asyncio.to_thread(worker.join, GUARD_SECONDS)

        records = shutdown_warnings(caplog)
        assert len(records) == 1
        assert "did not close" in records[0].getMessage()


class TestATimeoutRaisedByTheCloseItself:
    """A ``TimeoutError`` the connection's own close raises is not taken for the bound expiring."""

    async def test_it_propagates_and_is_not_reported_as_the_bound(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        error = TimeoutError("the worker gave up on its own")
        recorder.connection_close_raises(error)
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            with pytest.raises(TimeoutError, match="the worker gave up on its own") as caught:
                await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
        assert shutdown_warnings(caplog) == []
        assert caught.value is error

    async def test_while_the_store_is_failing_it_is_reported_as_a_failed_close(
        self,
        recorder: ShutdownRecorder,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(RuntimeError("the disk went away"))
        recorder.connection_close_raises(TimeoutError("the worker gave up on its own"))
        with caplog.at_level(logging.WARNING, logger=config.logger.name):
            resolved = await resolve_store()
            caplog.clear()

            with pytest.raises(RuntimeError, match="the disk went away"):
                await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
        (record,) = shutdown_warnings(caplog)
        assert "did not close within" not in record.getMessage()
        assert "cleanup also raised" in record.getMessage()
        # The failure that was set aside is attached to the warning.
        assert record.exc_info is not None
        assert record.exc_info[0] is TimeoutError


class TestWarningsAreBestEffort:
    """A shutdown warning that raises while being emitted does not change how shutdown ends."""

    async def test_a_quarantined_store_still_leaves_cleanly_when_the_log_call_fails(
        self, recorder: ShutdownRecorder, launch_route: str
    ) -> None:
        recorder.store_close_raises(ConnectionQuarantinedError(QUARANTINE_REASON))
        resolved = await resolve_store()

        with warnings_that_raise():
            await resolved.aclose()

        assert recorder.events[0] == STORE_CLOSE, launch_route

    async def test_an_abandoned_close_still_returns_when_the_log_call_fails(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "_CONNECTION_CLOSE_TIMEOUT_SECONDS", SHORT_BOUND_SECONDS)
        recorder.connection_close_never_completes()
        resolved = await resolve_store()

        with warnings_that_raise(), anyio.fail_after(GUARD_SECONDS):
            await resolved.aclose()

        assert recorder.completed_connection_closes == 0
        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]

    async def test_the_stores_exception_is_still_raised_when_the_log_call_fails_too(
        self, recorder: ShutdownRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        select_db_path_launch(monkeypatch, tmp_path)
        recorder.store_close_raises(RuntimeError("the disk went away"))
        recorder.connection_close_raises(OSError("the worker vanished"))
        resolved = await resolve_store()

        with warnings_that_raise(), pytest.raises(RuntimeError, match="the disk went away"):
            await resolved.aclose()

        assert recorder.events == [STORE_CLOSE, CONNECTION_CLOSE]
