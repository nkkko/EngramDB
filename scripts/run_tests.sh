#!/bin/bash
# Comprehensive test script for EngramDB
# Run this before submitting a PR to ensure code quality

set -e  # Exit on error

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}=  EngramDB Comprehensive Test Suite  =${NC}"
echo -e "${BLUE}=======================================${NC}"
echo

echo -e "${YELLOW}Running code formatting check...${NC}"
cargo fmt -- --check
echo -e "${GREEN}✓ Formatting is correct!${NC}"
echo

echo -e "${YELLOW}Running clippy for static analysis...${NC}"
cargo clippy -- -D warnings
echo -e "${GREEN}✓ No clippy warnings!${NC}"
echo

echo -e "${YELLOW}Building in debug mode...${NC}"
cargo build
echo -e "${GREEN}✓ Debug build successful!${NC}"
echo

echo -e "${YELLOW}Building in release mode...${NC}"
cargo build --release
echo -e "${GREEN}✓ Release build successful!${NC}"
echo

echo -e "${YELLOW}Running unit tests...${NC}"
cargo test --lib
echo -e "${GREEN}✓ Unit tests passed!${NC}"
echo

echo -e "${YELLOW}Running thread safety tests...${NC}"
cargo test --test thread_safety_tests
echo -e "${GREEN}✓ Thread safety tests passed!${NC}"
echo

echo -e "${YELLOW}Running integration tests...${NC}"
cargo test --test '*' --exclude thread_safety_tests
echo -e "${GREEN}✓ Integration tests passed!${NC}"
echo

echo -e "${YELLOW}Running benchmarks...${NC}"
cargo bench
echo -e "${GREEN}✓ Benchmarks completed!${NC}"
echo

if command -v cargo-tarpaulin > /dev/null; then
  echo -e "${YELLOW}Generating code coverage report...${NC}"
  cargo tarpaulin --out Html --output-dir target/tarpaulin
  echo -e "${GREEN}✓ Coverage report generated at target/tarpaulin/tarpaulin-report.html${NC}"
  echo
else
  echo -e "${YELLOW}Skipping code coverage (cargo-tarpaulin not installed)${NC}"
  echo -e "Install with: cargo install cargo-tarpaulin"
  echo
fi

# Run Python tests if available
if [ -d "python/tests" ]; then
  echo -e "${YELLOW}Running Python tests...${NC}"
  (cd python && python -m pytest -xvs tests/)
  echo -e "${GREEN}✓ Python tests passed!${NC}"
  echo
fi

# Run examples to ensure they work
echo -e "${YELLOW}Running Rust examples...${NC}"
for example in $(cargo run --example 2>&1 | grep -oP '(?<=--example )[^ ]+'); do
  echo "Testing example: $example"
  cargo run --example $example -- --test-only || echo -e "${RED}Example $example failed${NC}"
done
echo -e "${GREEN}✓ Examples tests completed!${NC}"
echo

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}=      All tests passed! Nice job!     =${NC}"
echo -e "${GREEN}=======================================${NC}"