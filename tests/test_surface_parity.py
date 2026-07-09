"""Surface-parity guard for the engrava-mcp server.

This server consumes engrava's *public* API; it must not reach into engrava's
internal modules.  These tests pin the externally observable surface so the
package keeps exposing exactly what an MCP client (and the registry listing)
expects:

* the full **11 tools / 3 resources / 3 prompts** in the default deployment;
* write-gating via ``ENGRAVA_MCP_READ_ONLY`` (the five write tools disappear
  while the read tools, resources, and prompts remain);
* runnable entry points for both the console script and ``python -m`` running;
* **no import of an engrava private module** — only the documented public API.
"""

from __future__ import annotations

import ast
import importlib.metadata as importlib_metadata
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect_client

from engrava_mcp import build_server
from engrava_mcp.config import CONFIG_ENV_VAR, DB_PATH_ENV_VAR
from engrava_mcp.server import READ_ONLY_ENV_VAR, main

#: Every tool the default deployment must advertise (6 read + 5 write = 11).
EXPECTED_READ_TOOLS = frozenset(
    {
        "get_thought",
        "search_memory",
        "search_keywords",
        "list_memory",
        "query_memory",
        "memory_stats",
    }
)
EXPECTED_WRITE_TOOLS = frozenset(
    {"store_thought", "update_thought", "link_thoughts", "delete_thought", "delete_edge"}
)
EXPECTED_TOOLS = EXPECTED_READ_TOOLS | EXPECTED_WRITE_TOOLS

#: Every resource URI the server must advertise (3).
EXPECTED_RESOURCE_URIS = frozenset(
    {"engrava://stats", "engrava://recent"}
)  # plus the templated engrava://thought/{thought_id}, asserted separately
EXPECTED_RESOURCE_TEMPLATE_URIS = frozenset({"engrava://thought/{thought_id}"})

#: Every prompt the server must advertise (3).
EXPECTED_PROMPTS = frozenset({"summarize_recent_memory", "find_related", "reflect_on_topic"})

#: Engrava import roots that are part of the documented public surface.  The
#: top-level ``engrava`` package is public; ``engrava.domain.exceptions`` is the
#: documented home of ``ReferentialIntegrityError`` (not re-exported from the
#: top level); ``engrava.domain.models`` / ``engrava.domain.enums`` /
#: ``engrava.mindql`` deep-import paths are shown in the engrava docs.  Any other
#: ``engrava.<...>`` import (e.g. ``engrava.infrastructure``) is a private module.
_PUBLIC_ENGRAVA_PREFIXES = (
    "engrava.domain.exceptions",
    "engrava.domain.enums",
    "engrava.domain.models",
    "engrava.mindql",
)


@pytest.fixture(autouse=True)
def _bare_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the server at a throwaway database and clear config overrides."""
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(tmp_path / "parity.db"))
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(READ_ONLY_ENV_VAR, raising=False)


async def test_default_surface_is_eleven_tools_three_resources_three_prompts() -> None:
    """The default deployment exposes exactly 11 tools, 3 resources, 3 prompts."""
    server = build_server()
    async with connect_client(server) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

    tool_names = {tool.name for tool in tools.tools}
    assert tool_names == EXPECTED_TOOLS
    assert len(tool_names) == 11

    static_uris = {str(resource.uri) for resource in resources.resources}
    template_uris = {str(template.uriTemplate) for template in templates.resourceTemplates}
    assert static_uris == EXPECTED_RESOURCE_URIS
    assert template_uris == EXPECTED_RESOURCE_TEMPLATE_URIS
    assert len(static_uris) + len(template_uris) == 3

    prompt_names = {prompt.name for prompt in prompts.prompts}
    assert prompt_names == EXPECTED_PROMPTS
    assert len(prompt_names) == 3


async def test_read_only_gating_hides_only_write_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-only mode drops the 5 write tools but keeps resources and prompts."""
    monkeypatch.setenv(READ_ONLY_ENV_VAR, "1")
    server = build_server()
    async with connect_client(server) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

    assert {tool.name for tool in tools.tools} == EXPECTED_READ_TOOLS
    # Resources and prompts are reads by definition, so gating never hides them.
    static_uris = {str(resource.uri) for resource in resources.resources}
    template_uris = {str(template.uriTemplate) for template in templates.resourceTemplates}
    assert static_uris == EXPECTED_RESOURCE_URIS
    assert template_uris == EXPECTED_RESOURCE_TEMPLATE_URIS
    assert {prompt.name for prompt in prompts.prompts} == EXPECTED_PROMPTS


def test_console_script_entry_point_targets_server_main() -> None:
    """The packaged ``engrava-mcp`` console script resolves to ``server:main``."""
    scripts = importlib_metadata.entry_points(group="console_scripts")
    entry = {ep.name: ep.value for ep in scripts}
    assert entry.get("engrava-mcp") == "engrava_mcp.server:main"
    # And the target is importable and callable.
    assert callable(main)


def test_module_run_guards_present() -> None:
    """Both ``python -m engrava_mcp`` and ``python -m engrava_mcp.server`` run."""
    package_root = Path(__file__).resolve().parent.parent / "src" / "engrava_mcp"
    for module in ("__main__.py", "server.py"):
        source = (package_root / module).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source, f"{module} is missing a module-run guard"
        assert "main()" in source


def test_no_private_engrava_imports() -> None:
    """No module under ``engrava_mcp`` imports an engrava private module."""
    package_root = Path(__file__).resolve().parent.parent / "src" / "engrava_mcp"
    offenders: list[str] = []
    for py_file in package_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            for module in modules:
                if module == "engrava" or not module.startswith("engrava."):
                    # The top-level public package and non-engrava imports are fine.
                    continue
                if not module.startswith(_PUBLIC_ENGRAVA_PREFIXES):
                    offenders.append(f"{py_file.name}: {module}")
    assert not offenders, (
        "engrava_mcp must consume only engrava's public API; private-module "
        f"imports found: {offenders}"
    )
