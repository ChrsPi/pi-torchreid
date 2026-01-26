# GitHub Actions CI/CD

This document describes the GitHub Actions workflows for automated testing and linting.

## Workflows

### `ci.yml` - Continuous Integration

Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
1. **Lint** - Runs ruff for code quality checks
   - Checks code style and formatting
   - Validates that code follows project standards
   - Runs on Python 3.11

2. **Test** - Runs the full test suite
   - Tests on Python 3.10, 3.11, and 3.12
   - Uses pytest to run all tests
   - Uploads coverage reports as artifacts (if generated)

## Workflow Features

- **Fast feedback**: Linting and testing run in parallel
- **Multi-version testing**: Ensures compatibility across Python 3.10-3.12
- **uv integration**: Uses `uv` for fast dependency management
- **Artifact uploads**: Coverage reports are saved for later analysis

## Status Badges

You can add status badges to your README.md:

```markdown
![CI](https://github.com/YOUR_USERNAME/pi-torchreid/actions/workflows/ci.yml/badge.svg)
```

Replace `YOUR_USERNAME` with your GitHub username.

## GitHub Free Tier

This setup is optimized for GitHub's free tier:
- ✅ Public repositories have **unlimited** GitHub Actions minutes
- ✅ Workflows run on `ubuntu-latest` (most cost-effective)
- ✅ Jobs run in parallel for faster feedback
- ✅ Uses matrix strategy to test multiple Python versions efficiently

## Local Testing

You can test the workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the CI workflow
act -j lint
act -j test
```

## Code Review

This repository uses [Greptile](https://www.greptile.com/) for AI-assisted code reviews. Greptile automatically reviews pull requests when they're opened. You can also manually trigger a review by commenting `@greptileai` on any PR.
