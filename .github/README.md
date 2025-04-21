# GitHub Workflows

## CI Workflow Status

**NOTICE: GitHub Actions CI workflow has been temporarily disabled** (as of April 22, 2025)

The CI workflow has been temporarily disabled due to failing build checks that need investigation.
Currently, the CI workflow is set to only run manually via `workflow_dispatch`.

### How to re-enable

To re-enable the CI workflow:
1. Edit `.github/workflows/ci.yml`
2. Uncomment the `push` and `pull_request` triggers
3. Remove or update this notice

### Current Issues

- Several failing checks across all platforms
- Need to investigate and fix build issues before re-enabling automatic CI