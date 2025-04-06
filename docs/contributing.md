# Contributing to EngramDB

Thank you for your interest in contributing to EngramDB! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to EngramDB. We aim to foster an inclusive and welcoming community.

## Getting Started

### Prerequisites

- Rust 2021 edition
- Cargo
- Python 3.7+ (for Python bindings)
- Git

### Setting Up the Development Environment

1. Clone the repository:

```bash
git clone https://github.com/yourusername/engramdb.git
cd engramdb
```

2. Build the project:

```bash
cargo build
```

3. Run the tests:

```bash
cargo test
```

4. For Python development:

```bash
cd python
pip install maturin
maturin develop
```

## Development Workflow

### Branching Strategy

- `main`: The main development branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release branches

Please create your branches from `main` and submit pull requests back to `main`.

### Commit Messages

Follow these guidelines for commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add vector similarity search with cosine distance

- Implement cosine similarity function
- Add search method to VectorIndex
- Add tests for vector search
- Update documentation

Fixes #123
```

## Pull Request Process

1. Ensure your code passes all tests
2. Update the documentation if necessary
3. Add tests for new functionality
4. Ensure your code follows the project's coding style
5. Submit a pull request with a clear description of the changes

## Coding Standards

### Rust

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` to format your code
- Use `clippy` to catch common mistakes
- Write documentation comments for public API
- Add tests for new functionality

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy

# Run tests
cargo test
```

### Python

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Write docstrings for public API
- Add tests for new functionality

```bash
# Format code
cd python
black .

# Run tests
pytest tests/
```

## Testing

### Rust Tests

- Unit tests should be in the same file as the code they test
- Integration tests should be in the `tests/` directory
- Use `cargo test` to run all tests

### Python Tests

- Tests should be in the `python/tests/` directory
- Use `pytest` to run tests

## Documentation

- Update the documentation when adding or changing features
- Write clear and concise documentation
- Include examples where appropriate
- Use Markdown for documentation files

## Feature Requests and Bug Reports

- Use the GitHub issue tracker to submit feature requests and bug reports
- Clearly describe the issue or feature
- Include steps to reproduce for bugs
- Include expected and actual behavior

## Project Structure

```
engramdb/
├── src/                  # Rust source code
│   ├── core/             # Core data structures
│   ├── database.rs       # Database implementation
│   ├── query/            # Query system
│   ├── storage/          # Storage engines
│   ├── vector/           # Vector index
│   └── lib.rs            # Library entry point
├── python/               # Python bindings
│   ├── src/              # Rust code for Python bindings
│   ├── engramdb/         # Python package
│   └── tests/            # Python tests
├── examples/             # Example code
├── tests/                # Integration tests
├── benches/              # Benchmarks
├── docs/                 # Documentation
└── web/                  # Web interface
```

## Release Process

1. Update version numbers in:
   - `Cargo.toml`
   - `python/Cargo.toml`
   - `python/pyproject.toml`
2. Update the CHANGELOG.md
3. Create a new release branch: `release/vX.Y.Z`
4. Create a pull request to `main`
5. After merging, tag the release: `git tag vX.Y.Z`
6. Push the tag: `git push origin vX.Y.Z`
7. Create a new release on GitHub
8. Publish to crates.io: `cargo publish`
9. Publish to PyPI: `cd python && maturin publish`

## License

By contributing to EngramDB, you agree that your contributions will be licensed under the project's MIT License.

## Contact

If you have questions or need help, you can:

- Open an issue on GitHub
- Reach out to the maintainers directly

Thank you for contributing to EngramDB!
