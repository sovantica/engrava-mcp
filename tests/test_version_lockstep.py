"""Lockstep guard tying ``_compat``, ``pyproject.toml``, and ``server.json``.

The engrava-supported range is declared in three places that must never drift
apart: the dependency pins in ``pyproject.toml``, the parsed constants in
:mod:`engrava_mcp._compat`, and — for the version — the registry manifest
``server.json``. These tests fail loudly if any one is bumped without the
others, which is the exact failure mode a range move risks.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from engrava_mcp import _compat

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The engrava range every engrava pin and the compat constants must agree on.
EXPECTED_RANGE = ">=0.6,<0.7"

#: The same range as a parsed specifier, compared semantically so format
#: variants (extra whitespace, clause order) do not matter and a stale range
#: still fails.
EXPECTED_SPECIFIER = SpecifierSet(EXPECTED_RANGE)

#: The canonical (PEP 503-normalized) distribution name we pin.
ENGRAVA_CANONICAL_NAME = canonicalize_name("engrava")

#: The package/manifest version this release declares everywhere.
EXPECTED_VERSION = "0.6.0"

#: The exact set of engrava requirements the project must carry, keyed by
#: ``(name, extras)``: the bare ``engrava`` dependency, the ``[vec]`` default
#: bundle, and the four provider extras (openai/ollama/hf/local). Asserting the
#: exact set means dropping, renaming, or adding any engrava requirement fails
#: the guard — a count check alone could not catch a swap.
EXPECTED_ENGRAVA_REQUIREMENTS = frozenset(
    {
        (ENGRAVA_CANONICAL_NAME, frozenset()),
        (ENGRAVA_CANONICAL_NAME, frozenset({"vec"})),
        (ENGRAVA_CANONICAL_NAME, frozenset({"embeddings-openai"})),
        (ENGRAVA_CANONICAL_NAME, frozenset({"embeddings-ollama"})),
        (ENGRAVA_CANONICAL_NAME, frozenset({"embeddings-hf"})),
        (ENGRAVA_CANONICAL_NAME, frozenset({"embeddings-local"})),
    }
)


def _load_pyproject() -> dict[str, object]:
    """Parse ``pyproject.toml`` from the repository root.

    Returns:
        The parsed TOML document as a mapping.

    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _load_server_json() -> dict[str, object]:
    """Parse ``server.json`` from the repository root.

    Returns:
        The parsed JSON document as a mapping.

    """
    return json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))


def _engrava_requirements() -> list[Requirement]:
    """Collect every engrava requirement across the project metadata.

    Gathers the ``[project].dependencies`` list and every
    ``[project.optional-dependencies]`` extra, parses each with
    :class:`packaging.requirements.Requirement`, and keeps only those whose
    canonical distribution name is ``engrava`` (with or without an extras
    group).  Canonicalizing the name means a miscased ``Engrava`` cannot slip
    past the filter — exactly the smuggling the exact-set guard must catch.

    Returns:
        The parsed engrava requirements; never empty for a valid project.

    """
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)

    candidates: list[str] = []
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    candidates.extend(dependencies)

    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict)
    for extra in optional.values():
        assert isinstance(extra, list)
        candidates.extend(extra)

    parsed = [Requirement(dep) for dep in candidates]
    return [req for req in parsed if canonicalize_name(req.name) == ENGRAVA_CANONICAL_NAME]


def test_engrava_requirement_set_is_exact_and_pins_the_expected_range() -> None:
    """The engrava requirements are exactly the expected set, each on the range.

    Asserting the exact ``(name, extras)`` set (not a count) means a dropped,
    renamed, or added extra fails the guard; comparing parsed specifiers means
    a stale or reformatted range fails too.
    """
    requirements = _engrava_requirements()

    found = {(canonicalize_name(req.name), frozenset(req.extras)) for req in requirements}
    assert found == EXPECTED_ENGRAVA_REQUIREMENTS

    # No engrava requirement was declared twice under the same extras group.
    assert len(requirements) == len(EXPECTED_ENGRAVA_REQUIREMENTS)

    for req in requirements:
        assert req.specifier == EXPECTED_SPECIFIER, (
            f"engrava pin {req!s} does not pin {EXPECTED_RANGE!r}"
        )


def test_compat_constants_match_the_expected_range() -> None:
    """The parsed ``_compat`` constants match the pinned range."""
    assert _compat.ENGRAVA_MIN_VERSION == (0, 6)
    assert _compat.ENGRAVA_MAX_VERSION_EXCLUSIVE == (0, 7)
    assert _compat.ENGRAVA_SUPPORTED_RANGE == EXPECTED_RANGE


def test_pyproject_version_matches_expected() -> None:
    """The distribution version in ``pyproject.toml`` is the release version."""
    project = _load_pyproject()["project"]
    assert isinstance(project, dict)
    assert project["version"] == EXPECTED_VERSION


def test_server_json_versions_match_expected() -> None:
    """Both ``server.json`` version fields are the release version."""
    manifest = _load_server_json()
    assert manifest["version"] == EXPECTED_VERSION
    packages = manifest["packages"]
    assert isinstance(packages, list)
    assert packages[0]["version"] == EXPECTED_VERSION
