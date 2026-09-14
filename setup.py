from setuptools import setup, Extension, find_packages
import pybind11

ext_modules = [
    Extension(
        "adaptq_py",
        [
            "adapters/adapter_python.cpp",
            "core/fwht.cpp",
            "core/codebook.cpp",
            "core/quantizer.cpp",
            "core/adaptq_c_api.cpp",
            "core/adaptq_backend_vtable.cpp",
            "cache/ring_buffer.cpp",
            "attention/attention.cpp",
            "utils/timer.cpp"
        ],
        include_dirs=["include", pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="adaptq",
    packages=find_packages(include=["adaptq", "adaptq.*"]),
    ext_modules=ext_modules,
    zip_safe=False,
)
