"""Tests for store resolution from the environment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from engrava import SqliteEngravaCore

from engrava_mcp import config
from engrava_mcp.config import (
    CONFIG_ENV_VAR,
    DB_PATH_ENV_VAR,
    StoreResolutionError,
    resolve_store,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Marker string used to prove a secret-bearing path or YAML value never
#: reaches a log record. Chosen to look nothing like real log text.
_SECRET_MARKER = "s3cr3t"  # noqa: S105 -- a marker string, not a real credential


class _SequentialMonotonic:
    """A fake clock returning a fixed sequence, then raising on any further call.

    Used to pin the elapsed-time arithmetic in :func:`resolve_store`'s startup
    logging without depending on real wall-clock timing. A call beyond the
    given sequence means ``resolve_store`` read the clock more times than this
    test expects, so it is treated as a test failure rather than papered over
    by repeating the last value.
    """

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            msg = "monotonic() called more times than the test expected"
            raise AssertionError(msg)
        return self._values.pop(0)


class TestResolveStore:
    """Tests for :func:`resolve_store` environment dispatch."""

    async def test_no_env_raises_resolution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)

        with pytest.raises(StoreResolutionError) as excinfo:
            await resolve_store()

        # Actionable: it names both documented configuration env vars.
        assert CONFIG_ENV_VAR in str(excinfo.value)
        assert DB_PATH_ENV_VAR in str(excinfo.value)

    async def test_db_path_resolves_a_usable_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "store.sqlite"))

        resolved = await resolve_store()
        try:
            assert await resolved.store.count_thoughts() == 0
        finally:
            await resolved.aclose()


class TestResolveFromDbPathCleanup:
    """The bare-database path closes its connection on a setup failure."""

    async def test_schema_failure_closes_connection_and_propagates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        opened: list[object] = []
        real_connect = config.aiosqlite.connect

        def _tracking_connect(*args: object, **kwargs: object) -> object:
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        class _SchemaError(RuntimeError):
            """Sentinel error raised in place of schema initialisation."""

        async def _failing_ensure_schema(self: SqliteEngravaCore) -> None:
            raise _SchemaError

        monkeypatch.setattr(config.aiosqlite, "connect", _tracking_connect)
        monkeypatch.setattr(SqliteEngravaCore, "ensure_schema", _failing_ensure_schema)
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "store.sqlite"))

        with pytest.raises(_SchemaError):
            await resolve_store()

        # The single opened connection must have been closed on the way out.
        assert len(opened) == 1
        connection = opened[0]
        # A closed aiosqlite connection rejects further work; confirm it is
        # no longer usable rather than depending on a private flag.
        with pytest.raises(ValueError, match="no active connection"):
            await connection.execute("SELECT 1")  # type: ignore[attr-defined]


class TestStartupLogging:
    """``resolve_store`` logs which route it took and how long opening took.

    Both records are read back from ``caplog`` rather than asserted to exist
    by side effect, and the elapsed-time arithmetic is pinned by patching
    ``config``'s own ``monotonic`` name (bound via ``from time import
    monotonic``), not an attribute on the shared ``time`` module. That name is
    a separate reference in ``config``'s namespace: reassigning it changes
    only what ``resolve_store`` sees when it calls the bare ``monotonic()``,
    and cannot affect ``time.monotonic`` itself -- the process-wide clock the
    event loop also reads. ``raising=False`` keeps the patch itself from
    failing on a tree that predates this feature, where ``config`` has no
    ``monotonic`` name at all: the point of that RED run is a failure on the
    missing records, not an ``AttributeError`` from the patch call.
    """

    def _info_records(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        """Records on the ``engrava_mcp`` logger at exactly ``INFO``.

        Filtering to ``INFO`` excludes the unrelated ``WARNING`` diagnostics
        (the no-provider warning, an unwired-extensions report) that can also
        fire on a resolution and would otherwise interleave with the two
        records under test here.

        Args:
            caplog: The active capture fixture.

        Returns:
            Matching records, in the order they were emitted.

        """
        return [
            record
            for record in caplog.records
            if record.name == "engrava_mcp" and record.levelno == logging.INFO
        ]

    async def test_config_route_logs_the_two_records_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config_file = tmp_path / "engrava.yaml"
        config_file.write_text(
            f"database:\n  path: {tmp_path / 'store.sqlite'}\n", encoding="utf-8"
        )
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
        monkeypatch.setattr(config, "monotonic", _SequentialMonotonic([10.0, 10.25]), raising=False)

        with caplog.at_level(logging.INFO, logger="engrava_mcp"):
            resolved = await resolve_store()
        try:
            records = self._info_records(caplog)
            assert [record.getMessage() for record in records] == [
                f"opening the store from {CONFIG_ENV_VAR}",
                "store ready in 0.25 s",
            ]
        finally:
            await resolved.aclose()

    async def test_db_path_route_logs_the_two_records_in_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "store.sqlite"))
        monkeypatch.setattr(config, "monotonic", _SequentialMonotonic([10.0, 10.25]), raising=False)

        with caplog.at_level(logging.INFO, logger="engrava_mcp"):
            resolved = await resolve_store()
        try:
            records = self._info_records(caplog)
            assert [record.getMessage() for record in records] == [
                f"opening the store from {DB_PATH_ENV_VAR}",
                "store ready in 0.25 s",
            ]
        finally:
            await resolved.aclose()

    async def test_config_route_logs_no_secret_from_path_or_yaml(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Both the yaml file's own path and a value read from inside it carry
        # the marker, so a leak from either source is caught.
        secret_dir = tmp_path / _SECRET_MARKER
        secret_dir.mkdir()
        config_file = secret_dir / "engrava.yaml"
        db_file = secret_dir / f"{_SECRET_MARKER}-store.sqlite"
        config_file.write_text(f"database:\n  path: {db_file}\n", encoding="utf-8")
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))

        with caplog.at_level(logging.INFO, logger="engrava_mcp"):
            resolved = await resolve_store()
        try:
            # The two records must actually be present, in order -- otherwise
            # the marker assertion below would pass vacuously on a resolution
            # that logged nothing at all.
            records = self._info_records(caplog)
            assert records[0].getMessage() == f"opening the store from {CONFIG_ENV_VAR}"
            assert records[1].getMessage().startswith("store ready in ")

            for record in caplog.records:
                assert _SECRET_MARKER not in record.getMessage()
        finally:
            await resolved.aclose()

    async def test_db_path_route_logs_no_secret_from_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret_dir = tmp_path / _SECRET_MARKER
        secret_dir.mkdir()
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(secret_dir / f"{_SECRET_MARKER}-store.sqlite"))

        with caplog.at_level(logging.INFO, logger="engrava_mcp"):
            resolved = await resolve_store()
        try:
            # Present, in order, before checking that neither carries the
            # marker -- otherwise the check below would pass vacuously on a
            # resolution that logged nothing at all.
            records = self._info_records(caplog)
            assert records[0].getMessage() == f"opening the store from {DB_PATH_ENV_VAR}"
            assert records[1].getMessage().startswith("store ready in ")

            for record in caplog.records:
                assert _SECRET_MARKER not in record.getMessage()
        finally:
            await resolved.aclose()

    async def test_resolution_failure_logs_neither_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)

        with (
            caplog.at_level(logging.INFO, logger="engrava_mcp"),
            pytest.raises(StoreResolutionError),
        ):
            await resolve_store()

        assert self._info_records(caplog) == []
