# Publishing EngramDB Python Package to PyPI

This document outlines the process for building and publishing the EngramDB Python package to PyPI.

## Prerequisites

- Python 3.8 or higher
- Rust toolchain (rustc, cargo)
- PyPI account with API token

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install maturin (Python build tool for Rust extensions):

```bash
pip install maturin
```

## Prepare Package for Publishing

1. Configure your package in `pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "engramdb-py"
description = "Python bindings for the EngramDB database"
authors = [
    {name = "Author", email = "author@example.com"}
]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Rust",
    "License :: OSI Approved :: MIT License",
    # Add other relevant classifiers
]
dependencies = [
    "numpy>=1.20.0",
    # Add other dependencies
]
version = "0.1.0"

[project.urls]
Repository = "https://github.com/yourusername/engramdb"
Documentation = "https://github.com/yourusername/engramdb/python/README.md"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "engramdb"
module-name = "engramdb_py"
include = ["Cargo.toml", "src/**/*"]
sdist-include = ["python/**/*"]
```

2. Configure Python package metadata in `Cargo.toml`:

```toml
[package]
name = "engramdb-py"
version = "0.1.0"
edition = "2021"
description = "Python bindings for EngramDB database"
authors = ["Niko <engramdb@disequi.com>"]
license = "MIT"

[lib]
name = "_engramdb"
crate-type = ["cdylib"]

[dependencies]
engramdb = { path = ".." }
pyo3 = { version = "0.19", features = ["extension-module"] }
numpy = "0.19"
# Other dependencies
```

## Building the Package

### Building for the Current Platform

1. Build the package with maturin:

```bash
cd /path/to/engramdb/python
maturin build --release
```

This will create wheel files in the `target/wheels/` directory.

### Cross-Platform Building

Maturin supports building for multiple platforms from a single machine:

#### macOS-specific Builds

For universal macOS binaries (Intel and Apple Silicon):

```bash
maturin build --release --universal2
```

#### Linux and Windows Builds

For cross-platform builds on macOS, you'll need to use Docker for Linux targets:

```bash
# Linux x86_64 (using manylinux Docker container)
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --manylinux 2014 -i python3.8 python3.9 python3.10 python3.11

# Alternative: Linux via Docker with specific Python version
docker run --rm -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 bash -c "cd /io && /opt/python/cp310-cp310/bin/pip install maturin && /opt/python/cp310-cp310/bin/maturin build --release"

# Windows (requires MinGW on macOS)
brew install mingw-w64
rustup target add x86_64-pc-windows-gnu
maturin build --release --target x86_64-pc-windows-gnu -i python3
```

#### Building for Multiple Platforms

Maturin does not support building for multiple targets in a single command. You'll need to run separate commands for each target. When cross-compiling, you must specify the Python interpreter to use with the `-i` flag:

```bash
# Build for macOS Intel
maturin build --release --target x86_64-apple-darwin -i python3

# Build for macOS Apple Silicon
maturin build --release --target aarch64-apple-darwin -i python3

# Build for Linux
maturin build --release --target x86_64-unknown-linux-gnu -i python3

# Build for Windows
maturin build --release --target x86_64-pc-windows-msvc -i python3
```

You can also specify multiple Python interpreters to build for different Python versions:

```bash
maturin build --release -i python3.8 -i python3.9 -i python3.10 -i python3.11
```

For building different platform wheels efficiently, use CI/CD workflows that run on the appropriate platforms or use Docker containers for Linux targets.

Note: Cross-compilation requires appropriate toolchains to be installed for each target platform.

## Testing Locally (Optional)

1. Install the wheel file locally:

```bash
pip install --force-reinstall target/wheels/engramdb_py-0.1.0-*.whl
```

2. Test that the package works as expected.

## Publishing to PyPI

1. Generate a PyPI API token:
   - Log in to your PyPI account at [pypi.org](https://pypi.org)
   - Go to Account Settings → API tokens → Add API token
   - Select appropriate scope (project-specific recommended for security)
   - Save the token securely (it will only be shown once)

2. Publish the package using your API token:

```bash
cd /path/to/engramdb/python
maturin publish --no-sdist --username __token__ --password "pypi-<YOUR_TOKEN>"
```

Note: The `--no-sdist` flag is used to avoid source distribution issues with README files.

3. Verify the package is available on PyPI and can be installed:

```bash
pip install engramdb-py
```

## Publishing to TestPyPI (Optional)

For testing purposes, you can publish to TestPyPI first:

1. Generate a TestPyPI token at [test.pypi.org](https://test.pypi.org)

2. Publish to TestPyPI:

```bash
maturin publish --repository testpypi --no-sdist --username __token__ --password "pypi-<YOUR_TEST_TOKEN>"
```

3. Install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ engramdb-py
```

## Automating with CI/CD

For a production setup, consider automating the publishing process using GitHub Actions or another CI/CD system:

1. Store your PyPI token as a repository secret
2. Create a workflow that builds and publishes on new releases or tags

### Cross-platform CI/CD Example

Here's an example GitHub Actions workflow for cross-platform builds:

```yaml
name: Build and Publish

on:
  release:
    types: [created]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, '3.10', '3.11']
        include:
          - os: macos-latest
            target: universal2

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          profile: minimal
          toolchain: stable
          override: true

      - name: Install maturin
        run: pip install maturin

      - name: Build wheels
        working-directory: ./python
        run: |
          if [ "${{ matrix.os }}" = "macos-latest" ]; then
            maturin build --release --universal2 -i python${{ matrix.python-version }}
          elif [ "${{ matrix.os }}" = "ubuntu-latest" ]; then
            # Use manylinux Docker image
            docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --manylinux 2014 -i python${{ matrix.python-version }}
          else
            # Windows
            maturin build --release -i python${{ matrix.python-version }}
          fi
        shell: bash

      - name: Upload wheels
        uses: actions/upload-artifact@v2
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
          path: target/wheels/*.whl

  publish:
    name: Publish to PyPI
    needs: build_wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
          
      - name: Install maturin
        run: pip install maturin twine
        
      - name: Download all wheels
        uses: actions/download-artifact@v2
        with:
          path: dist/
          
      - name: Flatten directory structure
        run: find dist -name "*.whl" -exec mv {} dist/ \;
        
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*.whl
```

This workflow builds wheels for multiple Python versions on Ubuntu, Windows, and macOS (with universal2 binaries for macOS), and then publishes them to PyPI.

## Troubleshooting

- **README conflicts**: If you encounter errors about duplicate README files, use the `include` and `sdist-include` options in `[tool.maturin]` to specify exactly which files to include, or use the `--no-sdist` flag.

- **Module naming issues**: If you see errors like `Couldn't find the symbol PyInit__engramdb in the native library`, ensure that:
  - The module name in `pyproject.toml` matches the function name in your Rust code:
    ```toml
    # pyproject.toml
    [tool.maturin]
    module-name = "_engramdb"  # This must match the #[pymodule] name
    ```
    ```rust
    // lib.rs
    #[pymodule]
    fn _engramdb(_py: Python, m: &PyModule) -> PyResult<()> {  // Must match module-name
        // ...
    }
    ```
  - Your Python module's `__init__.py` correctly imports from the Rust module:
    ```python
    try:
        from ._engramdb import *  # Use the correct module name
    except ImportError as e:
        # Fallback implementation
    ```

- **Authentication errors**: Double-check your token and ensure it has the correct permissions.

- **Cross-compilation issues**: 
  - For Linux targets from macOS: You MUST use Docker with the `--manylinux` option:
    ```bash
    # Using the maturin Docker image
    docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release --manylinux 2014
    
    # Or using the manylinux image
    docker run --rm -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 bash -c "cd /io && /opt/python/cp310-cp310/bin/pip install maturin && /opt/python/cp310-cp310/bin/maturin build --release"
    ```
  - For Windows targets from macOS: Install MinGW and use the GNU target:
    ```bash
    brew install mingw-w64
    rustup target add x86_64-pc-windows-gnu
    maturin build --release --target x86_64-pc-windows-gnu -i python3
    ```
  - For macOS universal2 on non-macOS: Not directly possible; use CI runners on macOS or build natively on macOS.
  - If you get the error "Invalid python interpreter version", it means you're trying to cross-compile without Docker or the appropriate toolchain.
  - If you get the error "Couldn't find any python interpreters", ensure you specify the Python interpreter with `-i`.
  - Remember that maturin doesn't support multiple `--target` flags in a single command - run separate commands for each target.

- **Compatibility issues**: Test wheels on their target platforms before publishing.

- **Missing dependencies**: Ensure all required system libraries are available in your build environment.