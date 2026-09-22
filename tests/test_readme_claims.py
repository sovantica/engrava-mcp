"""Tests that the README's checkable claims match the release it ships with.

Two claims are checked, and each went stale once because nothing tied it to the code:

* The compatibility table's newest row names the same ``engrava`` range as the dependency
  declared in ``pyproject.toml``. The table is maintained by hand; this test is what notices
  when a range move leaves it a release behind.
* The example ``engrava.yaml`` loads with Engrava's own configuration loader. The example is
  read out of the README rather than copied here, so it is the README that gets tested.

Neither test skips or passes vacuously: a table or an example that cannot be found is a
failure, because a claim that has gone missing is the same defect as one that is wrong.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from engrava import ConfigError, load_config
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The canonical (PEP 503-normalized) distribution name of the library this server wraps.
ENGRAVA_CANONICAL_NAME = canonicalize_name("engrava")

#: The heading of the README section that holds the compatibility table.
_COMPATIBILITY_HEADING = re.compile(r"^## Compatibility[ \t]*$", re.MULTILINE)

#: A row of the compatibility table, e.g. ``| `0.7.x` | `>=0.7,<0.8` |``.
_COMPATIBILITY_ROW = re.compile(
    r"^\|\s*`(?P<major>\d+)\.(?P<minor>\d+)\.x`\s*\|\s*`(?P<range>[^`]+)`\s*\|\s*$",
    re.MULTILINE,
)

#: Any Markdown heading line, used to bound a section.
_HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)

#: The heading that introduces the README's example configuration file.
_EXAMPLE_CONFIG_HEADING = re.compile(r"^### Example `engrava\.yaml`[ \t]*$", re.MULTILINE)

#: A fenced YAML block, capturing its body.
_YAML_FENCE = re.compile(r"^```yaml[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

#: Value the example's ``${OPENAI_API_KEY}`` reference resolves to under test.
_FAKE_API_KEY = "not-a-real-key"


def _readme_text() -> str:
    """Read ``README.md`` as text.

    Returns:
        The full contents of the repository README.

    """
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _compatibility_table_rows(readme: str) -> list[tuple[tuple[int, int], str]]:
    """Parse the compatibility table out of the README.

    Args:
        readme: The full README text.

    Returns:
        One ``((major, minor), engrava_range)`` pair per table row, in README order.

    """
    heading = _COMPATIBILITY_HEADING.search(readme)
    if heading is None:
        pytest.fail("README has no '## Compatibility' section")
    section = readme[heading.end() :]
    next_section = re.search(r"^## ", section, re.MULTILINE)
    if next_section is not None:
        section = section[: next_section.start()]

    rows = [
        ((int(match["major"]), int(match["minor"])), match["range"])
        for match in _COMPATIBILITY_ROW.finditer(section)
    ]
    if not rows:
        pytest.fail("README's '## Compatibility' section has no rows like | `X.Y.x` | `range` |")
    return rows


def _declared_engrava_range() -> SpecifierSet:
    """Read the plain ``engrava`` requirement from ``pyproject.toml``.

    The ``engrava[vec]`` and provider-extra requirements are deliberately skipped: the
    compatibility table describes the base dependency.

    Returns:
        The specifier set of the one ``engrava`` entry in ``[project].dependencies`` that
        carries no extras.

    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)

    plain = [
        requirement
        for requirement in map(Requirement, dependencies)
        if canonicalize_name(requirement.name) == ENGRAVA_CANONICAL_NAME and not requirement.extras
    ]
    assert len(plain) == 1, f"expected exactly one plain engrava dependency, found {plain}"
    return plain[0].specifier


def _example_config_yaml(readme: str) -> str:
    """Extract the README's example ``engrava.yaml`` as text.

    Args:
        readme: The full README text.

    Returns:
        The body of the first fenced YAML block under the example-config heading.

    """
    heading = _EXAMPLE_CONFIG_HEADING.search(readme)
    if heading is None:
        pytest.fail("README has no '### Example `engrava.yaml`' heading")
    after = readme[heading.end() :]

    fence = _YAML_FENCE.search(after)
    if fence is None:
        pytest.fail("README has no fenced yaml block under '### Example `engrava.yaml`'")
    if _HEADING.search(after[: fence.start()]):
        pytest.fail("the first yaml block after the example heading belongs to a later section")
    body = fence["body"]
    if not body.strip():
        pytest.fail("README's example `engrava.yaml` block is empty")
    return body


def test_compatibility_table_newest_row_matches_the_dependency_range() -> None:
    """The table's newest row names the range ``pyproject.toml`` requires of ``engrava``.

    The newest row is chosen by version, not by position, so a row added out of order is
    still the one compared.
    """
    version, table_range = max(_compatibility_table_rows(_readme_text()), key=lambda row: row[0])
    declared = _declared_engrava_range()

    assert SpecifierSet(table_range) == declared, (
        f"README compatibility table's newest row is {version[0]}.{version[1]}.x -> "
        f"{table_range!r}, but pyproject.toml requires engrava{declared}"
    )


def test_example_engrava_yaml_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The README's example ``engrava.yaml`` is accepted by Engrava's own loader.

    The loader only parses and validates: no store is opened, no embedding provider is
    built, and nothing touches the network.
    """
    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_API_KEY)
    config_path = tmp_path / "engrava.yaml"
    config_path.write_text(_example_config_yaml(_readme_text()), encoding="utf-8")

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        pytest.fail(f"README's example engrava.yaml does not load: {exc}", pytrace=False)

    # The `${OPENAI_API_KEY}` reference resolved, so the embeddings block was really parsed.
    assert config.embeddings is not None
    assert config.embeddings.api_key == _FAKE_API_KEY
