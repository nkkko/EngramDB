# CLAUDE.md - Development Guidelines

## Build/Test Commands
- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test single: `cargo test test_name`
- Lint: `cargo clippy`
- Format: `cargo fmt`
- Benchmark: `cargo bench`
- Check: `cargo check`
- Code coverage: `cargo tarpaulin`

## Testing Guidelines
- All new features require comprehensive tests including:
  - Unit tests: Test individual functions and methods
  - Integration tests: Test component interaction
  - Thread safety tests: For concurrent components
  - Performance benchmarks: For performance-critical code
- Maintain test coverage above 80% for all new code
- Test edge cases and error conditions explicitly
- Use property-based testing for complex algorithms
- All public APIs must have tests for all documented behavior
- Run the complete test suite before submitting a PR

## Test Naming Convention
- Unit tests: `test_unit_name_scenario_expected_result`
- Integration tests: `test_integration_components_scenario`
- Benchmarks: `bench_operation_scenario_configuration`
- Performance tests: Add baseline metrics as comments

## CI Pipeline Usage
- All PRs will run through the CI pipeline which includes:
  - Build verification on multiple platforms
  - Unit and integration tests
  - Performance regression checks
  - Linting and formatting verification
  - Static analysis
- Tests must pass on all platforms to be merged

## Code Style Guidelines
- Use Rust 2021 edition
- Follow Rust API guidelines: https://rust-lang.github.io/api-guidelines/
- Max line length: 100 characters
- Use descriptive variable names in snake_case
- Struct fields: snake_case
- Methods: snake_case
- Traits/Enums/Structs: PascalCase
- Use Result<T, E> for error handling with detailed error types
- Document public APIs with rustdoc comments
- Organize imports: std first, then external crates, then local modules
- Prefer owned types over references when appropriate
- Use strong typing (avoid String/&str when a specific type would be better)
- Place tests in a tests module with #[cfg(test)]
- Make all traits that may be used across threads (like VectorSearchIndex) bound by Send + Sync
- Avoid reference borrowing issues by cloning when necessary, especially in vector indexing code
- Never mention Claude in commits and never add Co-Authored-By: Claude <noreply@anthropic.com>"
- Always create new git branch for new feature or fix.
- Never use mock implementations, always write functional code.