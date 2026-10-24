"""Tests for the ADAPTQ_MARCH handling in setup.py.

setup.py is loaded as a module instead of being executed as a build script:
the setup() call is guarded behind __name__ == "__main__", which is also how
setuptools.build_meta runs the file during a PEP 517 build. These tests pin
the portability contract from the 0.2.2 release notes, which removed
-march=native from Python builds so wheels run on any x86 CPU.
"""
import importlib.util
import os
import platform
from pathlib import Path

import pytest

SETUP_PATH = Path(__file__).resolve().parents[1] / "setup.py"


@pytest.fixture(scope="module")
def build_setup():
    # Load with ADAPTQ_MARCH cleared so the module-level flag list reflects
    # the default configuration regardless of the developer's shell.
    saved = os.environ.pop("ADAPTQ_MARCH", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "adaptq_setup_under_test", SETUP_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is not None:
            os.environ["ADAPTQ_MARCH"] = saved
    return module


def test_default_build_stays_portable(monkeypatch, build_setup):
    monkeypatch.delenv("ADAPTQ_MARCH", raising=False)
    assert build_setup.march_flags() == []
    assert "-march=native" not in build_setup._compile_args
    assert "-mavx2" not in build_setup._compile_args
    assert "-mfma" not in build_setup._compile_args


def test_portable_keyword_is_equivalent_to_unset(monkeypatch, build_setup):
    monkeypatch.setenv("ADAPTQ_MARCH", "portable")
    assert build_setup.march_flags() == []


def test_blank_override_is_ignored(monkeypatch, build_setup):
    monkeypatch.setenv("ADAPTQ_MARCH", "   ")
    assert build_setup.march_flags() == []


def test_native_override_enables_host_tuning(monkeypatch, build_setup):
    monkeypatch.setenv("ADAPTQ_MARCH", "native")
    assert build_setup.march_flags() == ["-march=native"]


def test_native_override_is_case_insensitive(monkeypatch, build_setup):
    monkeypatch.setenv("ADAPTQ_MARCH", "Native")
    assert build_setup.march_flags() == ["-march=native"]


def test_explicit_target_is_passed_through(monkeypatch, build_setup):
    monkeypatch.setenv("ADAPTQ_MARCH", "haswell")
    assert build_setup.march_flags() == ["-march=haswell"]

    monkeypatch.setenv("ADAPTQ_MARCH", "x86-64-v3")
    assert build_setup.march_flags() == ["-march=x86-64-v3"]


def test_native_override_reaches_the_compiler_flags(monkeypatch):
    if platform.system() in {"Windows", "Darwin"}:
        pytest.skip("ISA baseline is only applied on the Linux/GCC path")

    monkeypatch.setenv("ADAPTQ_MARCH", "native")
    spec = importlib.util.spec_from_file_location(
        "adaptq_setup_native_under_test", SETUP_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "-march=native" in module._compile_args
