# Contributing to AdapTQ

Contributions are highly welcome! Whether it's adding an Apple Silicon (MLX) backend, optimizing the AVX2 kernels, or improving the documentation, please submit a Pull Request. 

## Development Setup

To develop or build from source, follow these steps:

1. **Install build dependencies:**
   `pip install build pytest twine`

2. **Build the C++ extension and install in editable mode:**
   `pip install -e .[dev]`

3. **Run the full C++ and Python test suite natively:**
   `cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release -DADAPTQ_BUILD_TESTS=ON`
   `cmake --build build_release --parallel`
   `cd build_release && ctest --output-on-failure`
   `cd .. && python tests/run_tests.py`

1. **Discuss Major Changes:** Please open an issue to discuss proposed algorithm or API changes _before_ starting work. Significant architectural deviations without prior discussion may not be accepted.
2. **Avoid Unrelated Refactors:** Keep Pull Requests (PRs) narrowly focused on resolving a specific issue or adding a single feature. Do not include cosmetic formatting or unrelated refactoring in feature PRs.
3. **Prove Your Claims:** If your PR introduces performance optimizations or modifies core algorithms, you **must** include reproducible benchmarks and quantitative evidence supporting your accuracy/performance claims.
4. **Test Everything:** New features must include comprehensive unit and integration tests. Regressions must include a failing test case that your PR fixes.
## Pull Request Guidelines
Before submitting a pull request, please ensure your code passes the following checks:
* Run Python linting: `ruff check . --fix`
* Verify native tests pass via CMake/ctest
* Verify Python integration tests pass: `pytest integration_tests/`




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
