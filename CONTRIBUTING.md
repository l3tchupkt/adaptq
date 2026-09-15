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

## Pull Request Guidelines
Before submitting a pull request, please ensure your code passes the following checks:
* Run Python linting: `ruff check . --fix`
* Verify native tests pass via CMake/ctest
* Verify Python integration tests pass: `pytest integration_tests/`




