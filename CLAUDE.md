# CLAUDE.md - Development Guidelines

## Build/Test Commands
- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test single: `cargo test test_name`
- Lint: `cargo clippy`
- Format: `cargo fmt`

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
- Never mention Claude in commits.
- Always create new git branch for new feature or fix.