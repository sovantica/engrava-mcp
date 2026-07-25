"""Guards for the store-resolution seam between config and real behaviour.

``config._resolve_from_db_path`` applies four concurrency pragmas to the
connection it opens, and ``SqliteEngravaCore.from_config`` applies the same set
on the ``engrava.yaml`` path.  The feature tests build their own store by hand,
so nothing previously observed those pragmas: each could be deleted with the
whole suite still green.

These tests close that seam.  They drive :func:`resolve_store` — the function
the server actually runs — and read each pragma **back from the live
connection**, never from the source text.  One test goes further and exercises
real behaviour through the resolved store, so ``foreign_keys`` is proven to be
enforced on the path users run rather than merely set.

The connection is captured by patching ``aiosqlite.connect``, the same module
attribute both resolution paths call.  That keeps the assertions off engrava's
private attributes while still observing the genuine connection.

**What these tests can and cannot detect.**  Two of the four pragmas are set to
a value the connection would already hold, so deleting their line is not an
observable change and no black-box test can fail on it:

* ``busy_timeout`` — Python's ``sqlite3.connect`` defaults to ``timeout=5.0``,
  which is already ``5000`` ms.  Our explicit pragma pins that value rather than
  changing it, so these tests act as a **drift guard**: they fail if the
  intended value and the connection's effective value ever diverge.
* ``foreign_keys`` — ``SqliteEngravaCore.ensure_schema`` asserts
  ``PRAGMA foreign_keys=ON`` itself, and store resolution calls it, so our line
  is defence in depth.  The behavioural test below therefore guards the
  property that matters — enforcement being on at all, including if the
  upstream library ever stops setting it.

``journal_mode`` and ``synchronous`` have no such backstop: deleting either line
changes the connection and fails these tests directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import aiosqlite
import pytest
from engrava import EdgeType
from engrava.domain.exceptions import ReferentialIntegrityError

from engrava_mcp.config import (
    BUSY_TIMEOUT_MS,
    CONFIG_ENV_VAR,
    DB_PATH_ENV_VAR,
    ResolvedStore,
    resolve_store,
)
from engrava_mcp.server import link_thoughts_impl, store_thought_impl

if TYPE_CHECKING:
    from pathlib import Path

#: ``PRAGMA journal_mode`` reports the mode in lower case.
EXPECTED_JOURNAL_MODE = "wal"

#: ``PRAGMA foreign_keys`` reports 1 when enforcement is on.
FOREIGN_KEYS_ON = 1

#: ``PRAGMA synchronous`` reports 1 for ``NORMAL``.
SYNCHRONOUS_NORMAL = 1


class ResolvedWithConnection(NamedTuple):
    """A resolved store paired with the connection that was opened for it.

    Attributes:
        resolved: The store returned by :func:`resolve_store`.
        connection: The live connection the resolution path opened.

    """

    resolved: ResolvedStore
    connection: aiosqlite.Connection


async def _resolve_capturing_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> ResolvedWithConnection:
    """Run :func:`resolve_store` and capture the connection it opened.

    Both resolution paths call ``aiosqlite.connect`` on the shared module, so
    patching that attribute observes either one without reaching into engrava's
    internals.

    Args:
        monkeypatch: Fixture used to patch the shared ``aiosqlite`` module.

    Returns:
        The resolved store and the connection opened for it.

    """
    opened: list[aiosqlite.Connection] = []
    real_connect = aiosqlite.connect

    def _tracking_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", _tracking_connect)
    resolved = await resolve_store()
    assert len(opened) == 1, f"expected exactly one connection, saw {len(opened)}"
    return ResolvedWithConnection(resolved=resolved, connection=opened[0])


async def _read_pragma(connection: aiosqlite.Connection, pragma: str) -> object:
    """Read a pragma back from a live connection.

    Args:
        connection: The connection to interrogate.
        pragma: The pragma name, e.g. ``"busy_timeout"``.

    Returns:
        The pragma's current value as SQLite reports it.

    """
    cursor = await connection.execute(f"PRAGMA {pragma};")
    row = await cursor.fetchone()
    assert row is not None, f"PRAGMA {pragma} returned no row"
    return row[0]


@pytest.fixture
def bare_db_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point resolution at a throwaway database via the bare-DB env var.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the database file.

    """
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "seam.sqlite"))


@pytest.fixture
def yaml_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point resolution at a minimal ``engrava.yaml``.

    The yaml declares only a database path, so it is the closest possible
    counterpart to the bare-database path — any difference in the resulting
    connection settings is a genuine drift between the two entry points.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the config and database files.

    """
    config_file = tmp_path / "engrava.yaml"
    config_file.write_text(
        f"database:\n  path: {tmp_path / 'from-yaml.sqlite'}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))


class TestBareDatabasePragmas:
    """Each concurrency pragma is in effect on the resolved connection."""

    async def test_busy_timeout_matches_the_configured_value(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        # The robustness deliverable: a contended write waits for the lock
        # instead of failing immediately with "database is locked".
        #
        # This is a drift guard, and deliberately so. The configured value
        # happens to equal the stdlib connect default (timeout=5.0 -> 5000 ms),
        # so removing the pragma line alone is unobservable. What this does
        # catch is the two ways the deliverable can actually regress: the
        # constant being changed without the pragma taking effect, and the
        # effective timeout drifting away from the documented value.
        captured = await _resolve_capturing_connection(monkeypatch)
        try:
            assert await _read_pragma(captured.connection, "busy_timeout") == BUSY_TIMEOUT_MS
        finally:
            await captured.resolved.aclose()

    async def test_journal_mode_is_wal(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        captured = await _resolve_capturing_connection(monkeypatch)
        try:
            assert await _read_pragma(captured.connection, "journal_mode") == (
                EXPECTED_JOURNAL_MODE
            )
        finally:
            await captured.resolved.aclose()

    async def test_foreign_keys_are_enforced(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        # Enforcement is asserted twice over — here and by ensure_schema — so
        # this pins the resulting property rather than either line. It fails
        # only if enforcement is lost entirely, which is the condition worth
        # catching (see the behavioural test below).
        captured = await _resolve_capturing_connection(monkeypatch)
        try:
            assert await _read_pragma(captured.connection, "foreign_keys") == FOREIGN_KEYS_ON
        finally:
            await captured.resolved.aclose()

    async def test_synchronous_is_normal(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        captured = await _resolve_capturing_connection(monkeypatch)
        try:
            assert await _read_pragma(captured.connection, "synchronous") == SYNCHRONOUS_NORMAL
        finally:
            await captured.resolved.aclose()


class TestResolvedStoreBehaviour:
    """Real behaviour observed through the store the server actually resolves."""

    async def test_referential_integrity_is_enforced_on_the_resolved_store(
        self, bare_db_env: None
    ) -> None:
        # Settings alone are not the guarantee: this reaches real behaviour
        # through resolve_store rather than a hand-built store, so foreign-key
        # enforcement is proven on the path users run. Verified failable — with
        # enforcement genuinely off, SQLite accepts the dangling edge and no
        # error is raised. That makes this the guard against the realistic
        # regression: the upstream library ceasing to enable enforcement.
        resolved = await resolve_store()
        try:
            created = await store_thought_impl(
                resolved.store,
                essence="seam source thought",
                content="a thought to link from",
            )
            with pytest.raises(ReferentialIntegrityError):
                await link_thoughts_impl(
                    resolved.store,
                    created["thought"]["thought_id"],
                    "no-such-thought",
                    EdgeType.ASSOCIATED,
                )
        finally:
            await resolved.aclose()


class TestEntryPointParity:
    """The yaml and bare-database entry points cannot silently drift apart."""

    async def test_config_path_matches_bare_path_concurrency_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Resolve once through each entry point and compare the resulting
        # connection settings. A change to either path that is not mirrored in
        # the other fails here.
        pragmas = ("busy_timeout", "journal_mode", "foreign_keys", "synchronous")

        monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
        monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "bare.sqlite"))
        bare = await _resolve_capturing_connection(monkeypatch)
        try:
            bare_settings = {name: await _read_pragma(bare.connection, name) for name in pragmas}
        finally:
            await bare.resolved.aclose()

        config_file = tmp_path / "engrava.yaml"
        config_file.write_text(
            f"database:\n  path: {tmp_path / 'from-yaml.sqlite'}\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(DB_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
        from_yaml = await _resolve_capturing_connection(monkeypatch)
        try:
            yaml_settings = {
                name: await _read_pragma(from_yaml.connection, name) for name in pragmas
            }
        finally:
            await from_yaml.resolved.aclose()

        assert bare_settings == yaml_settings
        # And the shared values are the intended ones, so the two paths cannot
        # agree on a wrong setting and still pass.
        assert bare_settings == {
            "busy_timeout": BUSY_TIMEOUT_MS,
            "journal_mode": EXPECTED_JOURNAL_MODE,
            "foreign_keys": FOREIGN_KEYS_ON,
            "synchronous": SYNCHRONOUS_NORMAL,
        }

    async def test_yaml_path_enforces_foreign_keys_behaviourally(
        self, yaml_config_env: None
    ) -> None:
        # The behavioural counterpart on the yaml entry point.
        resolved = await resolve_store()
        try:
            created = await store_thought_impl(
                resolved.store,
                essence="yaml seam source",
                content="a thought to link from",
            )
            with pytest.raises(ReferentialIntegrityError):
                await link_thoughts_impl(
                    resolved.store,
                    created["thought"]["thought_id"],
                    "no-such-thought",
                    EdgeType.ASSOCIATED,
                )
        finally:
            await resolved.aclose()
