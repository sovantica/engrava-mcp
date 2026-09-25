"""Tests that the README's checkable claims match the release it ships with.

Four claims are checked, and the first two went stale once each because nothing tied
them to the code:

* The compatibility table's newest row names the same ``engrava`` range as the dependency
  declared in ``pyproject.toml``. The table is maintained by hand; this test is what notices
  when a range move leaves it a release behind.
* The example ``engrava.yaml`` loads with Engrava's own configuration loader. The example is
  read out of the README rather than copied here, so it is the README that gets tested.
* The paragraph on which writes get embedded agrees with what loading that same example
  through Engrava's own config loader actually yields for ``embeddings.auto_embed`` -- the
  server's own path, not a value read off a dataclass in isolation -- so a change to
  either the loader's default or the example turns this claim red instead of quietly
  going stale.
* The example ``engrava.yaml``'s ``database.path`` is an absolute path.

None of the tests skip or pass vacuously: a table, example, or paragraph that cannot be
found is a failure, because a claim that has gone missing is the same defect as one that
is wrong.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

import pytest
import yaml
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

#: The heading of the README section that documents store resolution and setup.
_CONFIGURATION_HEADING = re.compile(r"^## Configuration[ \t]*$", re.MULTILINE)

#: The paragraph that follows the **Recommended:** paragraph, captured up to the blank
#: line that ends it. Searched only within the bounded Configuration section, so a
#: **Recommended:** paragraph appearing in some later section could not be matched instead.
_WRITE_EMBEDDING_PARAGRAPH = re.compile(
    r"^\*\*Recommended:\*\*.*?\n\n(?P<paragraph>.*?)\n\n", re.MULTILINE | re.DOTALL
)

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


def _configuration_section(readme: str) -> str:
    """Extract the README's ``## Configuration`` section.

    Args:
        readme: The full README text.

    Returns:
        The section's text, from just after the ``## Configuration`` heading up to (not
        including) the next heading of any level.

    """
    heading = _CONFIGURATION_HEADING.search(readme)
    if heading is None:
        pytest.fail("README has no '## Configuration' section")
    section = readme[heading.end() :]
    next_heading = _HEADING.search(section)
    if next_heading is not None:
        section = section[: next_heading.start()]
    return section


def _write_embedding_paragraph(readme: str) -> str:
    """Extract the paragraph that follows the Configuration section's **Recommended:** one.

    Args:
        readme: The full README text.

    Returns:
        The paragraph's text, with internal newlines collapsed to single spaces so a
        rewrap that changes nothing but line breaks does not change what a substring
        check sees.

    """
    match = _WRITE_EMBEDDING_PARAGRAPH.search(_configuration_section(readme))
    if match is None:
        pytest.fail(
            "README has no paragraph following '**Recommended:**' in its Configuration section"
        )
    return " ".join(match["paragraph"].split())


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


def test_readme_says_which_writes_get_embedded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The write-embedding paragraph agrees with what the README's own example loads to.

    Two things are pinned. First, that loading the README's example ``engrava.yaml`` through
    Engrava's own config loader -- the same path the server itself takes, not a value read
    off a dataclass in isolation -- yields ``auto_embed`` off; the example never sets it, so
    this is the default a reader following the README actually gets. If Engrava's loader
    default (or the example) ever makes that ``True``, this goes red instead of leaving the
    README's claim quietly wrong. Second, that the paragraph explaining what that setting
    means still makes every point: which provider embeds the query, that `store_thought`
    gets no embedding while off, that `search_memory`'s vector ranking (not its keyword
    ranking) is what can't match it, that `update_thought` leaves an existing embedding
    unrefreshed, and exactly which writes call the provider once it is on.
    """
    readme = _readme_text()
    example_yaml = _example_config_yaml(readme)

    # Without the check below, an example that set `auto_embed` itself would keep this test
    # green while it stopped testing Engrava's default at all.
    parsed_example = yaml.safe_load(example_yaml)
    assert isinstance(parsed_example, dict)
    embeddings_section = parsed_example.get("embeddings")
    assert isinstance(embeddings_section, dict)
    assert "auto_embed" not in embeddings_section, (
        "README's example engrava.yaml now sets `embeddings.auto_embed` explicitly, so "
        "the assertion below no longer pins Engrava's default -- it pins the example instead"
    )

    monkeypatch.setenv("OPENAI_API_KEY", _FAKE_API_KEY)
    config_path = tmp_path / "engrava.yaml"
    config_path.write_text(example_yaml, encoding="utf-8")

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        pytest.fail(f"README's example engrava.yaml does not load: {exc}", pytrace=False)

    # The example never sets `auto_embed`, so this is the loader's real default, on the
    # server's own path -- not a dataclass default read in isolation.
    assert config.embeddings is not None
    assert config.embeddings.auto_embed is False

    paragraph = _write_embedding_paragraph(readme)
    assert "the server embeds the query with the provider this `yaml` declares" in paragraph
    assert "`embeddings.auto_embed`" in paragraph
    assert "leaves off by default" in paragraph
    assert "`store_thought` gets no embedding" in paragraph
    assert "`search_memory`'s vector ranking cannot match it" in paragraph
    assert "keyword ranking still can" in paragraph
    assert (
        "an `update_thought` leaves whatever embedding the thought already had "
        "as it was, not refreshed"
    ) in paragraph
    assert (
        "creating a thought, or changing its `essence` or `content`, also calls the provider"
    ) in paragraph


def test_example_engrava_yaml_uses_an_absolute_database_path() -> None:
    """The README's example ``engrava.yaml`` gives ``database.path`` as an absolute path.

    A relative path resolves against the server process's working directory, which the
    MCP client chooses, not against the yaml's folder.
    """
    parsed_example = yaml.safe_load(_example_config_yaml(_readme_text()))
    assert isinstance(parsed_example, dict)
    database_section = parsed_example.get("database")
    assert isinstance(database_section, dict)
    path = database_section.get("path")
    assert isinstance(path, str)

    assert PurePosixPath(path).is_absolute(), (
        f"README's example engrava.yaml gives database.path as {path!r}, which is not "
        "an absolute path"
    )
