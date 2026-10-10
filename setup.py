"""
setup.py for adaptq.

Builds the adaptq_py C++17 extension (pybind11 binding to the AdapTQ
FWHT+quantization core). On Linux/macOS, OpenMP is enabled when available;
on Windows MSVC, /openmp is used.
"""
import os
import platform
import sys

from setuptools import Extension, find_packages, setup

try:
    import pybind11
    _pb11_includes = [pybind11.get_include()]
except ImportError:
    # Allow the package to be importable without pybind11 (C ext unavailable)
    _pb11_includes = []


# ---------------------------------------------------------------------------
# Compiler / linker flags — platform-aware
# ---------------------------------------------------------------------------
_compile_args = ["-std=c++17", "-O3"]
_link_args: list = []

if platform.system() == "Windows":
    # MSVC-style
    _compile_args = ["/std:c++17", "/O2", "/openmp"]
    _link_args = []
elif platform.system() == "Darwin":
    # macOS: OpenMP optional (not bundled with Apple Clang)
    _compile_args += ["-Xpreprocessor", "-fopenmp"]
    _link_args += ["-lomp"]
else:
    # Linux/GCC — OpenMP available via libgomp1
    _compile_args += ["-fopenmp", "-march=native"]
    _link_args += ["-fopenmp"]

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
_core_sources = [
    "adapters/adapter_python.cpp",
    "core/fwht.cpp",
    "core/codebook.cpp",
    "core/quantizer.cpp",
    "core/adaptq_c_api.cpp",
    "core/adaptq_backend_vtable.cpp",
    "cache/ring_buffer.cpp",
    "attention/attention.cpp",
    "kernels/scalar/kdot_scalar.cpp",
    "utils/timer.cpp",
]

ext_modules = []
if _pb11_includes:
    ext_modules = [
        Extension(
            "adaptq_py",
            sources=_core_sources,
            include_dirs=["include", "storage", "strategies"] + _pb11_includes,
            language="c++",
            extra_compile_args=_compile_args,
            extra_link_args=_link_args,
        )
    ]

setup(
    name="adaptq",
    packages=find_packages(include=["adaptq", "adaptq.*"]),
    ext_modules=ext_modules,
    zip_safe=False,
)
