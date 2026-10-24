"""Keep the declared support window in step with the versions CI exercises.

``pyproject.toml`` decides what pip is willing to install, while the matrix in
``.github/workflows/ci.yml`` decides what is actually built and tested. If the
two drift, the project either advertises a Python release that no job covers or
hides a release that the job actually supports. These checks fail on that drift
instead of leaving it for a release audit.
"""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"

PYTHON_CLASSIFIER = "Programming Language :: Python :: "


def _load_pyproject():
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _version_tuple(text):
    major, minor = text.strip().strip('"').split(".")
    return int(major), int(minor)


def _ci_matrix_versions():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"python-version:\s*\[([^\]]+)\]", workflow)
    assert match, "ci.yml no longer contains a python-version matrix"
    return sorted(_version_tuple(item) for item in match.group(1).split(","))


def _requires_python_floor(specifier):
    match = re.fullmatch(r">=(\d+)\.(\d+)", specifier.strip())
    assert match, f"unexpected requires-python specifier: {specifier!r}"
    return int(match.group(1)), int(match.group(2))


def test_requires_python_matches_ci_floor():
    project = _load_pyproject()["project"]
    floor = _requires_python_floor(project["requires-python"])

    assert floor == _ci_matrix_versions()[0], (
        "requires-python starts at "
        f"{floor[0]}.{floor[1]} but the oldest CI job uses "
        f"{_ci_matrix_versions()[0][0]}.{_ci_matrix_versions()[0][1]}"
    )


def test_python_classifiers_match_ci_matrix():
    classifiers = _load_pyproject()["project"]["classifiers"]
    declared = sorted(
        _version_tuple(item[len(PYTHON_CLASSIFIER):])
        for item in classifiers
        if item.startswith(PYTHON_CLASSIFIER)
        and re.fullmatch(r"\d+\.\d+", item[len(PYTHON_CLASSIFIER):])
    )

    assert declared == _ci_matrix_versions()


def test_build_backend_supports_pep_621_metadata():
    requires = _load_pyproject()["build-system"]["requires"]
    setuptools_requirement = next(
        (item for item in requires if item.startswith("setuptools")), None
    )
    assert setuptools_requirement is not None

    match = re.fullmatch(r"setuptools>=(\d+)(?:\.\d+)*", setuptools_requirement)
    assert match, (
        "build-system must pin setuptools with a >= lower bound, found "
        f"{setuptools_requirement!r}"
    )
    # PEP 621 [project] metadata is ignored by setuptools releases before 61,
    # which silently produces UNKNOWN-0.0.0 artifacts under --no-build-isolation.
    assert int(match.group(1)) >= 61
