# Codecov Setup Instructions

This PR adds automated coverage reporting using GitHub Actions and Codecov. Follow these steps to complete the setup.

## What This PR Adds

1. **GitHub Actions CI Workflow** (`.github/workflows/test.yml`)
   - Runs tests on Python 3.11, 3.12, and 3.13
   - Executes linting, type checking, and full test suite
   - Generates coverage reports and uploads to Codecov
   - Runs automatically on every push to main and every PR

2. **Codecov Configuration** (`.codecov.yml`)
   - Maintains 97% coverage target (current level)
   - Allows 1% threshold variation
   - Requires 90% coverage for new code in patches

3. **Dynamic Coverage Badge**
   - Live badge that updates automatically after each test run
   - Links to detailed coverage reports on Codecov

## Setup Steps (One-Time)

### 1. Sign Up for Codecov

Go to https://codecov.io and:
- Click "Sign in with GitHub"
- Authorize Codecov to access your GitHub account

### 2. Add the Repository

In Codecov:
- Click "Add new repository" or go to https://app.codecov.io/gh/ashita-ai
- Find and select `arbiter`
- Codecov will generate a repository token
- **Copy this token** - you'll need it in the next step

### 3. Add Token to GitHub Secrets

In your browser:
1. Go to https://github.com/ashita-ai/arbiter/settings/secrets/actions
2. Click "New repository secret"
3. Name: `CODECOV_TOKEN`
4. Value: [paste the token from Codecov]
5. Click "Add secret"

### 4. Merge This PR

Once the secret is added:
1. Merge this PR to main
2. The GitHub Actions workflow will run automatically
3. Coverage report will be uploaded to Codecov
4. The badge in README will update to show live coverage

## What Happens After Merge

**On Every Push/PR:**
- CI runs full test suite across Python 3.11, 3.12, 3.13
- Linting and type checking execute
- Coverage report generates and uploads to Codecov
- Codecov comments on PRs with coverage changes
- Badge updates automatically with current coverage

**Benefits:**
- ✅ Automatic coverage tracking - no manual updates needed
- ✅ Coverage enforcement - CI fails if coverage drops below 97%
- ✅ Visual feedback - see coverage trends over time
- ✅ PR integration - see coverage impact before merging
- ✅ Multi-version testing - ensure compatibility across Python versions

## Troubleshooting

### Badge Not Updating

If the badge shows "unknown" after merging:
1. Wait 5-10 minutes for first upload to complete
2. Check GitHub Actions run at https://github.com/ashita-ai/arbiter/actions
3. Verify CODECOV_TOKEN secret is set correctly
4. Check Codecov dashboard at https://app.codecov.io/gh/ashita-ai/arbiter

### CI Failing

If tests fail in CI but pass locally:
1. Check Python version differences (CI runs 3.11, 3.12, 3.13)
2. Verify all dependencies in `pyproject.toml` are pinned correctly
3. Check GitHub Actions logs for specific error messages

### Coverage Drops

If coverage drops below 97%:
1. Add tests for new code
2. Check which lines are uncovered in Codecov report
3. Update `.codecov.yml` target if intentional (requires approval)

## Questions?

- GitHub Actions docs: https://docs.github.com/en/actions
- Codecov docs: https://docs.codecov.com
- Open an issue if you encounter problems

---

**Note:** This is a one-time setup. After merging, coverage reporting will be fully automated.
