"""requirements.txt must not drift away from pyproject.toml.

Two files listing the same dependencies is two files that disagree in six
months. pyproject.toml is the source of truth - it is what `pip install -e
".[dev]"` and the build backend read - so requirements.txt is checked against
it rather than the other way round.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

# tomllib is 3.11+; on 3.9/3.10 there is no stdlib TOML parser and tomli is
# deliberately not a dependency, so this module simply does not run there.
tomllib = pytest.importorskip("tomllib")

pytestmark = pytest.mark.skipif(
    not PYPROJECT.exists(), reason="running against an installed copy, not the repo"
)


def _requirements() -> set[str]:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    return {
        stripped
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith("#")
    }


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_requirements_matches_the_dev_extra():
    """requirements.txt == the hard deps plus the `dev` extra, exactly."""
    project = _pyproject()["project"]
    expected = set(project["dependencies"]) | set(
        project["optional-dependencies"]["dev"]
    )
    assert _requirements() == expected


def test_numpy_is_still_the_only_hard_dependency():
    """The whole point of the fork: `pip install rgbox` pulls in numpy alone.

    A CI job asserts this from the outside by installing the package and
    checking pandas/sklearn/scipy are absent. This asserts it from the inside,
    so the failure names the culprit instead of surfacing as an import error
    three jobs later.
    """
    assert _pyproject()["project"]["dependencies"] == ["numpy>=1.22"]
