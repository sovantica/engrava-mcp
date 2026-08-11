"""Guard on the linter pin in the ``dev`` extra.

The lint gate runs ``ruff check`` with ``select = ["ALL"]``, so every rule a new
ruff release adds is opted into the moment an environment resolves it. With a
floor-only requirement, CI and a developer can install different ruff versions
from the same ``pyproject.toml`` and reach different verdicts on identical
source. The guard below asserts the *shape* of the requirement — a single
``==`` clause naming one concrete version — so moving the pin forward stays
free while loosening it back to a floor or a ceiling fails.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

#: Repository root, located from this test file's path (never hardcoded).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The canonical (PEP 503-normalized) name of the linter CI runs.
RUFF_CANONICAL_NAME = canonicalize_name("ruff")


def _dev_requirements() -> list[Requirement]:
    """Parse every requirement declared by the ``dev`` extra.

    Returns:
        The parsed requirements of ``[project.optional-dependencies].dev``.

    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    project = pyproject["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    dev = optional["dev"]
    assert isinstance(dev, list)

    return [Requirement(dep) for dep in dev]


def test_dev_extra_pins_ruff_to_an_exact_version() -> None:
    """The ``dev`` extra names ruff exactly once, on a single ``==`` clause.

    The assertion reads the requirement out of ``pyproject.toml`` rather than
    comparing against a hardcoded version string: the invariant is that CI and
    a developer resolve the same linter, not that the linter stays on any
    particular release.
    """
    ruff_requirements = [
        req for req in _dev_requirements() if canonicalize_name(req.name) == RUFF_CANONICAL_NAME
    ]

    assert len(ruff_requirements) == 1, (
        f"expected exactly one ruff requirement in the dev extra, found {len(ruff_requirements)}"
    )
    requirement = ruff_requirements[0]

    # A marker would let an environment skip the pin entirely, and extras on a
    # linter would change what gets installed under the same version.
    assert requirement.marker is None, f"ruff pin {requirement!s} is conditional on a marker"
    assert not requirement.extras, f"ruff pin {requirement!s} declares extras"

    clauses = list(requirement.specifier)
    assert len(clauses) == 1, (
        f"ruff pin {requirement!s} must be a single clause, found {len(clauses)}"
    )
    clause = clauses[0]
    assert clause.operator == "==", (
        f"ruff pin {requirement!s} must use '==' so CI and a developer resolve the same linter"
    )
    # `==0.15.*` is a range in `==` clothing; the pin must name one release.
    assert not clause.version.endswith(".*"), (
        f"ruff pin {requirement!s} is a wildcard, not an exact version"
    )
