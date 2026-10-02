"""Guard that every third-party module ``src/`` and ``tests/`` import is declared.

A module this repository imports directly but does not declare is installed only because
another package declares it, so the import can fail once that package stops declaring it.
``aiosqlite`` and ``pydantic`` reached ``src/`` that way, and ``packaging`` and ``yaml``
reached ``tests/``; ``anyio`` had the same gap before. This test reads the absolute imports out
of the source, not out of the installed environment, and fails when one of them names a
third-party module (anything but the standard library, ``engrava_mcp`` and ``tests``) whose
distribution is not declared in ``pyproject.toml``.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Modules whose PyPI distribution name differs from the module's own name.
#: Deliberately hand-written rather than read from the installed environment
#: (no ``importlib.metadata``, no ``packages_distributions``): the guard must
#: keep working against a ``pyproject.toml`` change nothing has installed yet.
MODULE_TO_DISTRIBUTION = {
    "yaml": "PyYAML",
}

#: Never a third-party distribution: the standard library, this package's own
#: name, the test package's own name (``from tests.inprocess_client import
#: ...`` resolves to top-level module ``tests``), and the ``__future__``
#: pseudo-module every file starts with.
_IGNORED_MODULES = frozenset(sys.stdlib_module_names) | {"engrava_mcp", "tests", "__future__"}


def _imported_modules(py_file: Path) -> set[str]:
    """Collect every absolute import's top-level module name in a file.

    Walks the full AST rather than only the module body, so an import inside
    a function, a ``TYPE_CHECKING`` block or a ``try`` counts exactly like a
    top-level one.

    Args:
        py_file: The source file to parse.

    Returns:
        The set of top-level module names the file imports absolutely, minus
        :data:`_IGNORED_MODULES`. A relative import (``level > 0``) can only
        name this project's own packages, so it is never collected.

    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules - _IGNORED_MODULES


def _imports_by_module(directory: Path) -> dict[str, set[Path]]:
    """Map every third-party module imported under a directory to the files that import it.

    Args:
        directory: A directory under the repository root, walked recursively
            for ``.py`` files.

    Returns:
        A mapping from module name to the set of files (relative to the
        repository root, for a readable failure message) that import it.

    """
    imports: dict[str, set[Path]] = {}
    for py_file in sorted(directory.rglob("*.py")):
        for module in _imported_modules(py_file):
            imports.setdefault(module, set()).add(py_file.relative_to(REPO_ROOT))
    return imports


def _distribution_name(module: str) -> str:
    """Map a module name to its canonical PyPI distribution name.

    Args:
        module: A top-level module name.

    Returns:
        The canonical (PEP 503-normalized) distribution name: the
        :data:`MODULE_TO_DISTRIBUTION` entry if the module has one,
        otherwise the module's own name.

    """
    return canonicalize_name(MODULE_TO_DISTRIBUTION.get(module, module))


def _declared_distributions(*requirement_lists: list[str]) -> set[str]:
    """Parse requirement-string lists into their canonical distribution names.

    A requirement guarded by an environment marker still counts as declared:
    this guard checks names only, never whether a marker would select the
    requirement in a given environment.

    Args:
        *requirement_lists: Any number of PEP 508 requirement-string lists,
            e.g. ``[project].dependencies`` or one extra's requirement list.

    Returns:
        The canonical distribution names declared across all the given
        lists. ``engrava[vec]`` and a bare ``engrava`` both yield ``engrava``.

    """
    names: set[str] = set()
    for requirements in requirement_lists:
        for raw in requirements:
            names.add(canonicalize_name(Requirement(raw).name))
    return names


def _pyproject_dependencies() -> tuple[list[str], list[str]]:
    """Read the runtime dependencies and the ``dev`` extra from ``pyproject.toml``.

    Returns:
        A ``(dependencies, dev_extra)`` pair of raw PEP 508 requirement
        strings, exactly as declared.

    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    dev = project["optional-dependencies"]["dev"]
    assert isinstance(dev, list)
    return dependencies, dev


def _undeclared(imports: dict[str, set[Path]], declared: set[str]) -> list[str]:
    """Build one failure line per imported module missing from ``declared``.

    Args:
        imports: A mapping from module name to its importing files, as
            returned by :func:`_imports_by_module`.
        declared: The canonical distribution names considered declared for
            this check.

    Returns:
        One formatted line per undeclared module, sorted by module name, each
        naming the module, the distribution name it was checked under, the
        importing file(s), and what to do about it.

    """
    lines: list[str] = []
    for module in sorted(imports):
        distribution = _distribution_name(module)
        if distribution in declared:
            continue
        files = ", ".join(str(f) for f in sorted(imports[module]))
        lines.append(
            f"  {module!r} (checked as distribution {distribution!r}), imported by "
            f"{files}: declare it in pyproject.toml, or add a MODULE_TO_DISTRIBUTION "
            "entry here if its distribution name differs from the module name"
        )
    return lines


def test_every_third_party_import_is_declared() -> None:
    """Every third-party module ``src/`` and ``tests/`` import is covered by ``pyproject.toml``.

    ``src/`` imports must be covered by ``[project].dependencies`` alone,
    because that is what a user installing this package gets. ``tests/``
    imports may additionally rely on the ``dev`` extra, because that is what
    the test suite itself is installed with.
    """
    dependencies, dev = _pyproject_dependencies()
    runtime = _declared_distributions(dependencies)
    runtime_and_dev = _declared_distributions(dependencies, dev)

    failures = [
        *_undeclared(_imports_by_module(REPO_ROOT / "src"), runtime),
        *_undeclared(_imports_by_module(REPO_ROOT / "tests"), runtime_and_dev),
    ]
    assert not failures, "undeclared imports found:\n" + "\n".join(failures)
