# Contributing to AdapTQ

Thank you for your interest in contributing to AdapTQ! We welcome contributions from the community that help improve the project. AdapTQ is a highly-optimized, research-focused KV-cache quantization library, and we expect contributions to uphold high standards for performance, accuracy, and code quality.

## Areas of Focus

We actively welcome contributions and improvements in the following areas:

- **Core Algorithms:** Improvements to KV-cache quantization and compression logic.
- **Performance:** CPU/GPU optimizations, memory management, and throughput improvements.
- **Accuracy:** Enhancements to numerical stability, validation, and benchmarking precision.
- **Architecture:** C++17 core improvements and cross-platform portability (Linux, Windows, MSVC, GCC).
- **Integrations:** Expanding and maintaining Python, PyTorch, and Transformers integrations.
- **Security & Quality:** Fuzzing, regression tests, robust memory bounds checking, and vulnerability fixes.

## General Guidelines

To ensure the library remains stable and efficient, please adhere to the following principles:

1. **Discuss Major Changes:** Please open an issue to discuss proposed algorithm or API changes _before_ starting work. Significant architectural deviations without prior discussion may not be accepted.
2. **Avoid Unrelated Refactors:** Keep Pull Requests (PRs) narrowly focused on resolving a specific issue or adding a single feature. Do not include cosmetic formatting or unrelated refactoring in feature PRs.
3. **Prove Your Claims:** If your PR introduces performance optimizations or modifies core algorithms, you **must** include reproducible benchmarks and quantitative evidence supporting your accuracy/performance claims.
4. **Test Everything:** New features must include comprehensive unit and integration tests. Regressions must include a failing test case that your PR fixes.

## Pull Request Expectations

- Submit **clean, minimal, and well-documented PRs**.
- Ensure all CI workflows (Linux and Windows native builds, CTest, and Python test suites) pass successfully.
- Include necessary documentation updates in your PR.

## Local Development & Testing

We strongly recommend building and running the complete test suite locally before pushing your changes.

If you are starting from a fresh checkout:

```bash
git clone https://github.com/l3tchupkt/adaptq.git
cd adaptq
```

Install the development dependencies and package in editable mode:

```bash
pip install build pytest twine
pip install -e .[dev]
```

### 1. Build the C++ Core

```bash
# Configure the project with testing enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release -DADAPTQ_BUILD_TESTS=ON

# Build the project
cmake --build build --config Release --parallel
```

### 2. Run C++ Tests

```bash
cd build
ctest --build-config Release --output-on-failure --parallel 4
cd ..
```

### 3. Run Python Integration Tests

Ensure you have a Python 3.8+ virtual environment set up.

```bash
# Install the package and test dependencies
pip install -e .[dev]

# Run exact accuracy and ABI integration checks
python tests/test_accuracy.py
python tests/run_tests.py

# Run the complete Pytest suite
pytest tests integration_tests -m "not gpu_required"
```

Before submitting a pull request, also run:

```bash
ruff check . --fix
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
pytest integration_tests/
```

## Security Vulnerabilities

**DO NOT** publicly disclose security vulnerabilities in GitHub issues or Pull Requests. If you discover a security flaw (e.g., out-of-bounds memory access, malicious snapshot injection vulnerabilities), please refer to our [SECURITY.md](SECURITY.md) file for instructions on how to securely report it to the maintainers.
