"""Guard for the shielded lifespan teardown.

The server's lifespan wraps its connection teardown in
``anyio.CancelScope(shield=True)`` so the close runs to completion even while
the surrounding task is being cancelled — which is what happens on stdio EOF,
i.e. every normal shutdown of this server. Without the shield the close is
interrupted and the database worker thread can outlive the event loop and raise
on a late callback.

The guard only runs on the cancellation path, so no happy-path test reaches it:
``shield=False`` previously passed the entire suite.

**Determinism.** The cancellation here is *forced*, not raced. The surrounding
cancel scope is cancelled while the lifespan body is still active, so by the
time the ``finally`` block runs its teardown the scope is already in a cancelled
state. There is no sleep, no wall-clock timing, and no reliance on one task
winning a race: the ordering is a property of the structure, not of scheduling.

Asserting that the connection merely *reports* itself closed would be hollow —
an interrupted close leaves the connection object closed either way. The
assertion that actually discriminates is whether the teardown **ran to
completion**, which is the guarantee the shield provides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import pytest

import engrava_mcp.server as server_module
from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR, ResolvedStore, resolve_store

if TYPE_CHECKING:
    from pathlib import Path


class TeardownProbe:
    """Records whether the lifespan's connection teardown ran to completion.

    Wraps the real :meth:`ResolvedStore.aclose` without altering it: the flag is
    set only after the genuine close returns normally, so an interrupted
    teardown leaves it unset.

    Attributes:
        completed: ``True`` once the real teardown has returned normally.

    """

    def __init__(self) -> None:
        self.completed = False

    def wrap(self, resolved: ResolvedStore) -> ResolvedStore:
        """Return a ``ResolvedStore`` whose close is observed.

        Args:
            resolved: The genuine resolved store to wrap.

        Returns:
            An equivalent ``ResolvedStore`` that records teardown completion.

        """

        async def _observed_close() -> None:
            await resolved.aclose()
            self.completed = True

        return ResolvedStore(store=resolved.store, _closer=_observed_close)


@pytest.fixture
def bare_db_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point store resolution at a throwaway database file.

    Args:
        monkeypatch: Fixture used to set the environment.
        tmp_path: Temporary directory for the database file.

    """
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "cancellation.sqlite"))


async def _run_lifespan_cancelled_during_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TeardownProbe, bool]:
    """Drive a full lifespan whose surrounding scope is cancelled before teardown.

    Args:
        monkeypatch: Fixture used to install the teardown probe.

    Returns:
        The probe and whether the surrounding scope caught a cancellation.

    """
    probe = TeardownProbe()

    async def _resolve_with_probe() -> ResolvedStore:
        return probe.wrap(await resolve_store())

    monkeypatch.setattr(server_module, "resolve_store", _resolve_with_probe)

    server = server_module.build_server()
    lifespan = server.settings.lifespan
    assert lifespan is not None, "the built server must carry a lifespan"

    with anyio.CancelScope() as scope:
        async with lifespan(server):
            # Cancel while the body is still active: the teardown in the
            # lifespan's ``finally`` therefore begins with the surrounding
            # scope already cancelled. Deterministic — no race, no sleep.
            scope.cancel()

    return probe, scope.cancelled_caught


class TestShieldedTeardown:
    """The teardown completes even when the surrounding task is cancelled."""

    async def test_teardown_completes_despite_cancellation(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        probe, cancelled_caught = await _run_lifespan_cancelled_during_teardown(monkeypatch)

        # The shield's guarantee: the close ran to the end. Unshielded, the
        # cancellation interrupts it and this flag is never set.
        assert probe.completed is True
        # And the pending cancellation never surfaced as an exception through
        # the teardown, so no late-callback error escapes the shutdown path.
        assert cancelled_caught is False

    async def test_uncancelled_lifespan_also_tears_down(
        self, monkeypatch: pytest.MonkeyPatch, bare_db_env: None
    ) -> None:
        # Control: the shield does not change the ordinary shutdown path.
        probe = TeardownProbe()

        async def _resolve_with_probe() -> ResolvedStore:
            return probe.wrap(await resolve_store())

        monkeypatch.setattr(server_module, "resolve_store", _resolve_with_probe)
        server = server_module.build_server()
        lifespan = server.settings.lifespan
        assert lifespan is not None

        async with lifespan(server):
            assert probe.completed is False  # still serving

        assert probe.completed is True
