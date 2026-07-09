"""Tests for store resolution from the environment."""

from __future__ import annotations

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
    from pathlib import Path


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
