"""Guards on the registry manifest's declared environment variables.

``server.json`` is what the MCP Registry publishes and what a host reads to
build the launch command for this server. The server reads exactly three
environment variables and refuses to start unless a store variable is set.
This module guards that the manifest declares them, and proves that a host
supplying only what the manifest asks for gets a server that starts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava_mcp.server import READ_ONLY_ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: This repository's ``src`` directory. Used as the manifest-only launch's
#: working directory -- see :func:`test_manifest_only_launch_starts_a_working_server`.
SRC_DIR = REPO_ROOT / "src"

#: Expected declarations, in the order the manifest must list them. Keyed
#: from the package's own constants rather than retyped, so a rename in the
#: code and a stale manifest both fail the drift guard. ``ENGRAVA_DB_PATH``
#: is the only required one: it is the one route that needs no other file,
#: so a host that fills in only what is required still gets a working
#: server.
EXPECTED_DECLARATIONS: Mapping[str, Mapping[str, object]] = {
    DB_PATH_ENV_VAR: {"isRequired": True, "format": "filepath"},
    CONFIG_ENV_VAR: {"isRequired": False, "format": "filepath"},
    READ_ONLY_ENV_VAR: {"isRequired": False, "format": "boolean"},
}


def _load_server_json() -> dict[str, object]:
    """Parse ``server.json`` from the repository root.

    Returns:
        The parsed JSON document as a mapping.

    """
    return json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))


def _manifest_environment_variables() -> list[dict[str, object]]:
    """Read the declared environment variables off the first package.

    A manifest with the field absent reads as declaring none, so callers
    (the manifest-only launch) fall through to an empty environment and let
    the *server* fail, rather than raising ``KeyError`` here.

    Returns:
        The manifest's ``packages[0].environmentVariables``, or an empty list
        when the field is absent.

    """
    manifest = _load_server_json()
    packages = manifest["packages"]
    assert isinstance(packages, list)
    first_package = packages[0]
    assert isinstance(first_package, dict)
    declared = first_package.get("environmentVariables", [])
    assert isinstance(declared, list)
    return declared


def test_manifest_declares_the_three_environment_variables() -> None:
    """The manifest declares exactly the three variables the server reads, in order.

    The ordered list of names is asserted exactly against
    :data:`EXPECTED_DECLARATIONS`, which also rules out duplicates. Each
    entry's key set must be exactly ``{"name", "description", "isRequired",
    "format"}`` -- so a ``default``, ``value``, ``placeholder`` or
    ``isSecret`` fails this test too -- with a non-empty description and the
    ``isRequired`` / ``format`` from :data:`EXPECTED_DECLARATIONS`.

    A manifest without the field fails here.
    """
    declared = _manifest_environment_variables()

    names = [entry["name"] for entry in declared]
    assert names == list(EXPECTED_DECLARATIONS)

    for entry in declared:
        assert entry.keys() == {"name", "description", "isRequired", "format"}
        assert isinstance(entry["description"], str)
        assert entry["description"]

        expected = EXPECTED_DECLARATIONS[entry["name"]]
        assert entry["isRequired"] == expected["isRequired"], entry["name"]
        assert entry["format"] == expected["format"], entry["name"]


def _build_manifest_only_env(db_path: Path) -> dict[str, str]:
    """Build a launch environment from the manifest's variables alone.

    Returns only the manifest-derived store variables -- what a host
    configuring this server from the MCP Registry manifest would supply.
    The MCP SDK's stdio client merges this under its own default inherited
    variables (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM`` and
    ``USER`` on POSIX -- none of them a store variable), so this function
    does not add those itself.

    Each required variable gets a value chosen for what it means:
    :data:`~engrava_mcp.config.DB_PATH_ENV_VAR` gets a path inside
    ``tmp_path``. A manifest that requires any other variable fails this
    test outright, naming it. Optional variables are left unset.

    Args:
        db_path: Where the required database-path variable should point.

    Returns:
        The manifest-derived environment mapping for the child process.

    """
    env: dict[str, str] = {}
    for entry in _manifest_environment_variables():
        if not entry.get("isRequired", False):
            continue
        name = entry["name"]
        assert isinstance(name, str)
        if name == DB_PATH_ENV_VAR:
            env[name] = str(db_path)
        else:
            pytest.fail(f"manifest requires {name!r}, which this test does not know how to satisfy")

    return env


async def test_manifest_only_launch_starts_a_working_server(tmp_path: Path) -> None:
    """A host that supplies only what the manifest requires gets a working server.

    Launches the server with an environment built from
    ``packages[0].environmentVariables`` alone (see
    :func:`_build_manifest_only_env`) and ``cwd`` set to :data:`SRC_DIR`, so
    the child imports this repository's package -- ``python -m
    engrava_mcp`` puts its working directory first on ``sys.path`` -- then
    initializes over stdio and calls ``memory_stats``.

    A manifest without the field fails here too: no environment variable is
    declared, so the subprocess exits during startup with
    ``StoreResolutionError``'s "No engrava store configured" message.

    Args:
        tmp_path: Pytest temp directory holding the throwaway database path.

    """
    env = _build_manifest_only_env(tmp_path / "manifest-only.sqlite")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "engrava_mcp"],
        env=env,
        cwd=SRC_DIR,
    )

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        init_result = await session.initialize()
        assert init_result.server_info.name

        result = await session.call_tool("memory_stats", {})
        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["thought_count"] == 0
