"""Tests for how a busy SQLite store is answered by ``link_thoughts`` and ``store_thought``.

A busy ``sqlite3.OperationalError`` -- SQLite reported database contention -- means
different things for the two write tools this module covers, because of *where* it can
come from (see ``engrava_mcp.server``'s own module-level docstrings for ``_is_busy_error``,
``_store_thought_busy_message`` and ``_residual_write_guard``):

* ``link_thoughts``'s ``store.create_edge`` is one write unit (a ``BEGIN IMMEDIATE``, its
  journal append, then one commit) with nothing written after it, so a busy error there --
  whether from acquiring the lock or from the commit itself -- always means nothing was
  written, and the message says so.
* ``store_thought``'s ``store.create_thought`` commits before several steps that can still
  raise afterwards (auto-embed, hygiene cleanup, a hooks class, derived-record dispatch), so
  a busy error establishes nothing about the outcome. The message instead reports what one
  fresh read of the attempted id found, and never what this call did.

Most cases here stage a real failure with a **second, independent connection** to the same
on-disk database file -- a raw, synchronous ``sqlite3`` connection, deliberately not a second
``aiosqlite``/asyncio connection, so it genuinely contends as a separate connection rather
than sharing this module's own event loop or aiosqlite's per-connection worker thread. The
store under test always runs with a short ``PRAGMA busy_timeout`` so a real contention
failure surfaces in milliseconds rather than seconds.

Held-write-lock cases (WAL journal mode, the store's own default) hold the second
connection's own ``BEGIN IMMEDIATE`` so the store's write can never even acquire the lock.
``link_thoughts``'s commit-time case instead runs the store in rollback-journal mode
(``PRAGMA journal_mode=DELETE``) with the second connection holding an open *read*
transaction: the store's own ``BEGIN IMMEDIATE`` succeeds (a SHARED and a RESERVED lock are
compatible), but its ``COMMIT`` -- which must upgrade to EXCLUSIVE -- is busy for as long as
the reader's SHARED lock stands.

One case (the ``deduplicate=True`` read-back-raises combination in ``TestStoreThoughtBusyGuard``)
injects the busy ``OperationalError`` directly instead: ``deduplicate=True``'s own
probe-and-insert lock window retries a busy ``BEGIN IMMEDIATE`` internally and, on exhausting
its own retries, raises the typed ``WriteContentionError`` -- not a raw ``OperationalError`` --
so a real held lock at that point never reaches the guard this module tests. See that test's own
comment.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, NoReturn

import aiosqlite
import pytest
from engrava import EdgeType, SqliteEngravaCore

from engrava_mcp.server import link_thoughts_impl
from tests.conftest import make_thought
from tests.test_errors import _assert_no_leak, _client_for, _error_text

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@asynccontextmanager
async def _file_store(
    db_path: Path,
    *,
    journal_mode: str = "WAL",
    busy_timeout_ms: int = 100,
    embedding_provider: object | None = None,
    auto_embed: bool = False,
) -> AsyncIterator[SqliteEngravaCore]:
    """Build a real, file-backed store with a short busy timeout, for contention tests.

    Unlike the shared ``store`` fixture (``:memory:``, so no second connection could ever
    contend with it), this opens a real on-disk database file, so a second, independent
    connection to the same path is a genuine second lock holder.

    Args:
        db_path: Path to the (not yet existing) database file.
        journal_mode: The store connection's own ``PRAGMA journal_mode`` -- ``"WAL"`` for the
            held-write-lock cases, ``"DELETE"`` (rollback journal) for a commit-time case.
        busy_timeout_ms: The store connection's own ``PRAGMA busy_timeout``. Short, so a real
            contention failure surfaces quickly and the suite stays fast.
        embedding_provider: Forwarded to :class:`~engrava.SqliteEngravaCore`, for the
            auto-embed contention case.
        auto_embed: Forwarded to :class:`~engrava.SqliteEngravaCore`.

    Yields:
        A schema-initialised store with no seeded thoughts, closed on exit.

    """
    connection = await aiosqlite.connect(str(db_path))
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys=ON")
    await connection.execute(f"PRAGMA journal_mode={journal_mode}")
    await connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    store = SqliteEngravaCore(
        connection, embedding_provider=embedding_provider, auto_embed=auto_embed
    )
    await store.ensure_schema()
    try:
        yield store
    finally:
        await connection.close()


def _grab_write_lock(db_path: Path) -> sqlite3.Connection:
    """Open a second, independent connection and take the write lock.

    A raw, synchronous ``sqlite3`` connection -- not the store's own ``aiosqlite`` one -- so
    it genuinely contends as a second connection to the same file, the same shape
    ``PRAGMA busy_timeout`` itself waits out. ``BEGIN IMMEDIATE`` takes the write lock before
    returning, so the lock is actually held by the time the caller's own guarded call runs.

    Args:
        db_path: Path to the shared, on-disk database file.

    Returns:
        The holding connection, still inside its open write transaction. Release it with
        :func:`_release`.

    """
    holder = sqlite3.connect(str(db_path))
    holder.execute("BEGIN IMMEDIATE")
    return holder


def _grab_read_lock(db_path: Path) -> sqlite3.Connection:
    """Open a second connection and hold an open read transaction.

    In rollback-journal mode a writer's own ``COMMIT`` must upgrade from a RESERVED to an
    EXCLUSIVE lock, and SQLite refuses that upgrade while another connection holds even a
    SHARED lock -- which an open read transaction keeps taken until it ends. Unlike
    :func:`_grab_write_lock`, this never blocks a writer's own ``BEGIN IMMEDIATE`` (a SHARED
    and a RESERVED lock are compatible), only its later commit.

    Args:
        db_path: Path to the shared, on-disk database file.

    Returns:
        The holding connection, still inside its open read transaction. Release it with
        :func:`_release`.

    """
    holder = sqlite3.connect(str(db_path))
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM thought").fetchall()
    return holder


def _release(holder: sqlite3.Connection) -> None:
    """Roll back and close a connection from :func:`_grab_write_lock` / :func:`_grab_read_lock`.

    Args:
        holder: The connection to release. Nothing was ever written through it, so a rollback
            is always correct.

    """
    try:
        holder.execute("ROLLBACK")
    finally:
        holder.close()


class _LockGrabbingEmbeddingProvider:
    """An embedding provider whose ``embed`` takes the write lock on a second connection.

    Reproduces ``store_embedding``'s own ``BEGIN IMMEDIATE`` hitting contention *after*
    ``create_thought``'s own commit: ``embed`` runs first -- outside ``store_embedding``'s own
    write unit -- and, as a side effect, takes the write lock through a second, independent
    connection to the same database file, so by the time ``store_embedding`` tries its own
    ``BEGIN IMMEDIATE`` the lock is already held and it times out.
    """

    def __init__(
        self, *, db_path: Path, dimension: int = 3, model_name: str = "fake-embedder"
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name
        self._db_path = db_path
        self.holder: sqlite3.Connection | None = None

    async def embed(self, text: str) -> list[float]:
        """Take the write lock on a second connection, then return a fake vector."""
        self.holder = _grab_write_lock(self._db_path)
        return [0.1] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Delegate to :meth:`embed` per text."""
        return [await self.embed(text) for text in texts]

    def release(self) -> None:
        """Release the write lock this provider's ``embed`` took, if any."""
        if self.holder is not None:
            _release(self.holder)
            self.holder = None


def _expected_link_thoughts_busy_message() -> str:
    """The exact ``link_thoughts`` busy-store message, written out literally.

    Not imported from :mod:`engrava_mcp.server` -- comparing against the constant under test
    would make the exact-equality check below tautological.

    Returns:
        The exact message text, unprefixed by MCPServer's own
        ``"Error executing tool <name>: "`` wrapper.

    """
    return "SQLite reported database contention. Nothing was written, and retrying is safe."


def _expected_store_thought_not_found_message(thought_id: str) -> str:
    """The exact ``store_thought`` busy message when the read-back found nothing.

    See :func:`_expected_link_thoughts_busy_message` for why this is written out literally.

    Args:
        thought_id: The identifier the store attempted.

    Returns:
        The exact message text, unprefixed by MCPServer's own error-result wrapper.

    """
    return (
        f"store_thought for {thought_id!r}: SQLite reported database "
        "contention; whether the thought was stored could not be "
        "confirmed. A read just now found no thought with this id."
    )


def _expected_store_thought_found_message(thought_id: str) -> str:
    """The exact ``store_thought`` busy message when the read-back found a thought.

    See :func:`_expected_link_thoughts_busy_message` for why this is written out literally.

    Args:
        thought_id: The identifier the store attempted.

    Returns:
        The exact message text, unprefixed by MCPServer's own error-result wrapper.

    """
    return (
        f"store_thought for {thought_id!r}: SQLite reported database "
        "contention; whether the thought was stored could not be confirmed. "
        "A read just now found a thought with this id."
    )


def _expected_store_thought_readback_failed_message(thought_id: str) -> str:
    """The exact ``store_thought`` busy message when the read-back itself raised.

    See :func:`_expected_link_thoughts_busy_message` for why this is written out literally.

    Args:
        thought_id: The identifier the store attempted.

    Returns:
        The exact message text, unprefixed by MCPServer's own error-result wrapper.

    """
    return (
        f"store_thought for {thought_id!r}: SQLite reported database "
        "contention; whether the thought was stored could not be "
        "confirmed. Read it back with get_thought before retrying -- a "
        "blind retry can store a second copy."
    )


def _expected_store_thought_readback_failed_deduplicate_message(thought_id: str) -> str:
    """The exact ``store_thought`` (``deduplicate=True``) busy message when the read-back raised.

    See :func:`_expected_link_thoughts_busy_message` for why this is written out literally.

    Args:
        thought_id: The identifier the store attempted.

    Returns:
        The exact message text, unprefixed by MCPServer's own error-result wrapper.

    """
    return (
        f"store_thought for {thought_id!r} (deduplicate=True): SQLite "
        "reported database contention; whether the thought was stored "
        "could not be confirmed. With deduplicate=True the call may have "
        "matched an existing thought with identical content instead of "
        f"storing a new one, so reading {thought_id!r} back with "
        "get_thought cannot settle what happened: finding it shows a "
        "thought with that id exists, not that this call stored it."
    )


class TestBusyErrorClassifier:
    """``_is_busy_error`` recognises SQLite's busy result codes, structurally, and only those."""

    @pytest.mark.parametrize("code", [5, 261, 517, 773])
    def test_each_busy_code_is_busy(self, code: int) -> None:
        from engrava_mcp.server import _is_busy_error

        exc = sqlite3.OperationalError("database is locked")
        exc.sqlite_errorcode = code  # type: ignore[attr-defined]
        assert _is_busy_error(exc) is True

    def test_sqlite_locked_is_not_busy(self) -> None:
        # SQLITE_LOCKED (6): a different family of locking conflict, not busy.
        from engrava_mcp.server import _is_busy_error

        exc = sqlite3.OperationalError("database table is locked")
        exc.sqlite_errorcode = 6  # type: ignore[attr-defined]
        assert _is_busy_error(exc) is False

    def test_a_non_busy_code_is_not_busy(self) -> None:
        from engrava_mcp.server import _is_busy_error

        exc = sqlite3.OperationalError("disk I/O error")
        exc.sqlite_errorcode = 10  # SQLITE_IOERR  # type: ignore[attr-defined]
        assert _is_busy_error(exc) is False

    def test_a_missing_errorcode_is_not_busy(self) -> None:
        from engrava_mcp.server import _is_busy_error

        exc = sqlite3.OperationalError("some failure with no attached code")
        assert not hasattr(exc, "sqlite_errorcode")
        assert _is_busy_error(exc) is False


class TestLinkThoughtsBusyGuard:
    """A busy ``create_edge`` maps to a curated, retry-safe message; nothing else does."""

    async def test_non_busy_operational_error_propagates_unchanged(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = sqlite3.OperationalError("disk I/O error")
        original.sqlite_errorcode = 10  # type: ignore[attr-defined]

        async def _failing_create_edge(self: SqliteEngravaCore, record: object) -> NoReturn:
            raise original

        monkeypatch.setattr(type(store), "create_edge", _failing_create_edge)

        with pytest.raises(sqlite3.OperationalError) as excinfo:
            await link_thoughts_impl(store, "thought-alpha", "thought-beta", EdgeType.ASSOCIATED)
        assert excinfo.value is original

    async def test_held_write_lock_reports_the_busy_message_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "link-busy-write-lock.db"
        async with _file_store(db_path) as backend:
            await backend.create_thought(
                make_thought("thought-a", essence="edge source", content="a body")
            )
            await backend.create_thought(
                make_thought("thought-b", essence="edge target", content="b body")
            )

            holder = _grab_write_lock(db_path)
            try:
                async with _client_for(backend) as client:
                    result = await client.call_tool(
                        "link_thoughts",
                        {
                            "from_thought_id": "thought-a",
                            "to_thought_id": "thought-b",
                            "edge_type": "ASSOCIATED",
                        },
                    )
            finally:
                _release(holder)

            assert result.is_error is True
            text = _error_text(result.content)
            expected = _expected_link_thoughts_busy_message()
            assert text == f"Error executing tool link_thoughts: {expected}"
            _assert_no_leak(text)

            edges = await backend.list_edges(source="thought-a")
            assert len(edges) == 0

    async def test_commit_time_busy_reports_the_busy_message_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "link-busy-commit.db"
        async with _file_store(db_path, journal_mode="DELETE") as backend:
            await backend.create_thought(
                make_thought("thought-a", essence="edge source", content="a body")
            )
            await backend.create_thought(
                make_thought("thought-b", essence="edge target", content="b body")
            )

            holder = _grab_read_lock(db_path)
            try:
                async with _client_for(backend) as client:
                    result = await client.call_tool(
                        "link_thoughts",
                        {
                            "from_thought_id": "thought-a",
                            "to_thought_id": "thought-b",
                            "edge_type": "ASSOCIATED",
                        },
                    )
            finally:
                _release(holder)

            assert result.is_error is True
            text = _error_text(result.content)
            expected = _expected_link_thoughts_busy_message()
            assert text == f"Error executing tool link_thoughts: {expected}"
            _assert_no_leak(text)

            edges = await backend.list_edges(source="thought-a")
            assert len(edges) == 0


class TestStoreThoughtBusyGuard:
    """A busy ``create_thought`` reports only what a fresh read-back found."""

    async def test_held_write_lock_reports_no_thought_found(self, tmp_path: Path) -> None:
        db_path = tmp_path / "store-busy-write-lock.db"
        async with _file_store(db_path) as backend:
            holder = _grab_write_lock(db_path)
            try:
                async with _client_for(backend) as client:
                    result = await client.call_tool(
                        "store_thought",
                        {
                            "essence": "contended store",
                            "content": "never gets a chance to write",
                            "thought_id": "thought-busy-write-lock",
                        },
                    )
            finally:
                _release(holder)

            assert result.is_error is True
            text = _error_text(result.content)
            expected = _expected_store_thought_not_found_message("thought-busy-write-lock")
            assert text == f"Error executing tool store_thought: {expected}"
            _assert_no_leak(text)

            assert await backend.get_thought("thought-busy-write-lock") is None

    async def test_auto_embed_contention_after_commit_reports_thought_found(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "store-busy-auto-embed.db"
        provider = _LockGrabbingEmbeddingProvider(db_path=db_path)
        async with _file_store(db_path, embedding_provider=provider, auto_embed=True) as backend:
            try:
                async with _client_for(backend) as client:
                    result = await client.call_tool(
                        "store_thought",
                        {
                            "essence": "auto-embed contention",
                            "content": "commits, then the embed step hits contention",
                            "thought_id": "thought-busy-auto-embed",
                        },
                    )
            finally:
                provider.release()

            assert result.is_error is True
            text = _error_text(result.content)
            expected = _expected_store_thought_found_message("thought-busy-auto-embed")
            assert text == f"Error executing tool store_thought: {expected}"
            _assert_no_leak(text)

            stored = await backend.get_thought("thought-busy-auto-embed")
            assert stored is not None

    async def test_read_back_raising_reports_the_generic_advice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "store-busy-readback-fails.db"
        async with _file_store(db_path) as backend:

            async def _broken_get_thought(thought_id: str) -> NoReturn:
                msg = "the read-back itself failed"
                raise RuntimeError(msg)

            monkeypatch.setattr(backend, "get_thought", _broken_get_thought)

            holder = _grab_write_lock(db_path)
            try:
                async with _client_for(backend) as client:
                    result = await client.call_tool(
                        "store_thought",
                        {
                            "essence": "read-back also fails",
                            "content": "the confirming read itself raises",
                            "thought_id": "thought-busy-readback-fails",
                        },
                    )
            finally:
                _release(holder)

            assert result.is_error is True
            text = _error_text(result.content)
            expected = _expected_store_thought_readback_failed_message(
                "thought-busy-readback-fails"
            )
            assert text == f"Error executing tool store_thought: {expected}"
            _assert_no_leak(text)

    async def test_read_back_raising_with_deduplicate_reports_the_dedup_caveat(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # deduplicate=True's own probe-and-insert lock window retries a busy
        # BEGIN IMMEDIATE internally and, on exhausting its own retries,
        # raises the typed WriteContentionError -- not a raw
        # OperationalError -- so a real held lock never reaches this guard's
        # busy handling for this combination (confirmed: staging one here
        # instead produced the pre-existing "memory store is busy" message).
        # The busy OperationalError is injected directly instead, which
        # exercises exactly the same _store_thought_busy_message branch a
        # busy failure elsewhere in create_thought would.
        busy = sqlite3.OperationalError("database is locked")
        busy.sqlite_errorcode = 5  # type: ignore[attr-defined]

        async def _failing_create_thought(
            self: SqliteEngravaCore, thought: object, **kwargs: object
        ) -> NoReturn:
            raise busy

        async def _broken_get_thought(thought_id: str) -> NoReturn:
            msg = "the read-back itself failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(type(store), "create_thought", _failing_create_thought)
        monkeypatch.setattr(store, "get_thought", _broken_get_thought)

        async with _client_for(store) as client:
            result = await client.call_tool(
                "store_thought",
                {
                    "essence": "dedup + read-back fails",
                    "content": "both the write and the confirming read hit trouble",
                    "thought_id": "thought-busy-readback-fails-dedup",
                    "deduplicate": True,
                },
            )

        assert result.is_error is True
        text = _error_text(result.content)
        expected = _expected_store_thought_readback_failed_deduplicate_message(
            "thought-busy-readback-fails-dedup"
        )
        assert text == f"Error executing tool store_thought: {expected}"
        _assert_no_leak(text)

    async def test_non_busy_operational_error_on_create_thought_is_unaffected(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The classifier's False branch: a non-busy OperationalError from
        # create_thought must keep going through the pre-existing, generic
        # residual classification -- never the new busy-specific message.
        original = sqlite3.OperationalError("disk I/O error")
        original.sqlite_errorcode = 10  # type: ignore[attr-defined]

        async def _failing_create_thought(
            self: SqliteEngravaCore, thought: object, **kwargs: object
        ) -> NoReturn:
            raise original

        monkeypatch.setattr(type(store), "create_thought", _failing_create_thought)

        async with _client_for(store) as client:
            result = await client.call_tool(
                "store_thought",
                {"essence": "unaffected", "content": "a non-busy operational error"},
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "SQLite reported database contention" not in text
        assert "does not recognise" in text.lower()
        _assert_no_leak(text)

    async def test_busy_operational_error_on_update_thought_keeps_current_behaviour(
        self, store: SqliteEngravaCore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # update_thought is out of scope for the new busy handling: a raw busy
        # OperationalError there must still fall through to the pre-existing
        # generic residual message, exactly as before this change.
        busy = sqlite3.OperationalError("database is locked")
        busy.sqlite_errorcode = 5  # type: ignore[attr-defined]

        async def _failing_update(
            self: SqliteEngravaCore, thought_id: str, **changes: object
        ) -> NoReturn:
            raise busy

        monkeypatch.setattr(type(store), "update_thought", _failing_update)

        async with _client_for(store) as client:
            result = await client.call_tool(
                "update_thought", {"thought_id": "thought-alpha", "essence": "busy update"}
            )

        assert result.is_error is True
        text = _error_text(result.content)
        assert "SQLite reported database contention" not in text
        assert "does not recognise" in text.lower()
        _assert_no_leak(text)
