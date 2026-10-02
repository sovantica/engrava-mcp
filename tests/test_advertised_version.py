"""The version this server advertises to MCP clients is its own, not the SDK's.

Before this, ``build_server`` passed no ``version=`` to ``MCPServer`` at all, so the
``initialize`` handshake's ``serverInfo.version`` fell back to whatever the low-level
server defaulted to — the ``mcp`` SDK's own version on the pre-port 1.x line, and an
empty string on 2.x. Neither is a version of anything this project ships.

The defect being fixed is exactly the gap between "we passed the argument" and "the
client is told the right thing", so the tests here read the ``initialize`` response
through a connected client (:func:`tests.inprocess_client.connect_client`), not the
``MCPServer(...)`` constructor call — a test on the call site cannot see this gap.
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava_mcp.server import (
    _DISTRIBUTION_NAME,
    _UNKNOWN_VERSION,
    READ_ONLY_ENV_VAR,
    SERVER_NAME,
    _server_version,
    build_server,
)
from tests.inprocess_client import connect_client

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> None:
    """Set up a minimal, deterministic environment for a ``build_server()`` call.

    Args:
        monkeypatch: The active monkeypatch fixture.
        tmp_path: The test's temporary directory.
        name: A filename stem for the throwaway database this call resolves to.

    """
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / f"{name}.db"))
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)


class TestServerVersionHelper:
    """Direct coverage of the resolution helper: both branches, in isolation."""

    def test_reads_the_installed_engrava_mcp_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _version(name: str) -> str:
            assert name == _DISTRIBUTION_NAME
            return "9.9.9"

        monkeypatch.setattr(importlib.metadata, "version", _version)

        assert _server_version() == "9.9.9"

    def test_missing_distribution_metadata_falls_back_to_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(_DISTRIBUTION_NAME)

        monkeypatch.setattr(importlib.metadata, "version", _raise)

        assert _server_version() == _UNKNOWN_VERSION


class TestInitializeHandshakeAdvertisesOurVersion:
    """The wire, not the constructor call — read through a connected client."""

    async def test_handshake_reports_the_installed_engrava_mcp_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clean_env(monkeypatch, tmp_path, "handshake")
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: "1.2.3" if name == _DISTRIBUTION_NAME else "should-not-be-read",
        )

        server = build_server()
        async with connect_client(server) as client:
            info = client.server_info

        assert info is not None
        assert info.name == SERVER_NAME
        assert info.version == "1.2.3"

    async def test_missing_distribution_metadata_advertises_unknown_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clean_env(monkeypatch, tmp_path, "unknown")

        def _raise(_name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(_DISTRIBUTION_NAME)

        monkeypatch.setattr(importlib.metadata, "version", _raise)

        server = build_server()
        async with connect_client(server) as client:
            info = client.server_info

        assert info is not None
        assert info.version == _UNKNOWN_VERSION

    async def test_bumping_the_sdk_distribution_does_not_change_the_advertised_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Demonstrates SDK-bump independence instead of arguing for it: the old
        # fallback tracked whatever the `mcp` SDK's own distribution reported, so a
        # Dependabot bump of `mcp` silently changed what this server claimed to be.
        # Simulate `mcp` reporting a bumped version here and assert the wire is
        # unaffected: it must come from `engrava-mcp`'s own metadata, not `mcp`'s,
        # regardless of what the SDK distribution reports.
        _clean_env(monkeypatch, tmp_path, "sdkbump")
        real_version = importlib.metadata.version

        def _bumped(name: str) -> str:
            if name == "mcp":
                return "99.0.0"
            return real_version(name)

        monkeypatch.setattr(importlib.metadata, "version", _bumped)

        server = build_server()
        async with connect_client(server) as client:
            info = client.server_info

        assert info is not None
        assert info.version == real_version(_DISTRIBUTION_NAME)
        assert info.version != "99.0.0"
