"""Tests for the ``python -m engrava_mcp`` entry point and ``main``."""

from __future__ import annotations

import runpy
import sys
from typing import TYPE_CHECKING

from engrava_mcp import server

if TYPE_CHECKING:
    import pytest


class TestModuleEntryPoint:
    """The ``__main__`` module wires the package run target to ``main``."""

    def test_importing_main_module_exposes_server_main(self) -> None:
        import engrava_mcp.__main__ as entry

        assert entry.main is server.main

    def test_running_as_module_invokes_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[bool] = []

        def _record() -> None:
            called.append(True)

        monkeypatch.setattr(server, "main", _record)

        # Drop any cached submodule so runpy executes ``__main__`` cleanly
        # (a stale entry triggers a RuntimeWarning about unpredictable order).
        monkeypatch.delitem(sys.modules, "engrava_mcp.__main__", raising=False)
        runpy.run_module("engrava_mcp", run_name="__main__")

        assert called == [True]


class TestMainStartupGuard:
    """``main`` runs the engrava-version check before serving."""

    def test_main_warns_then_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []

        class _FakeServer:
            def run(self) -> None:
                order.append("run")

        def _record_check() -> None:
            order.append("check")

        def _fake_build() -> _FakeServer:
            return _FakeServer()

        monkeypatch.setattr(server, "warn_if_engrava_out_of_range", _record_check)
        monkeypatch.setattr(server, "build_server", _fake_build)

        server.main()

        assert order == ["check", "run"]
