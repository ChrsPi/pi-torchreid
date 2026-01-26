# Phase 1: UV Migration - Detailed Plan

## Overview

Migrate from `setup.py` + `requirements.txt` to modern `pyproject.toml` with uv for dependency management, and replace legacy linting tools (flake8, yapf, isort) with ruff.

**Estimated Time:** 2-3 hours  
**Risk Level:** Low (mostly additive changes, can test incrementally)

---

## Step-by-Step Execution Plan

### Step 1: Install and Verify uv

NOTE: We already have latest uv version.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**Verification:** `uv --version` should show version 0.9.0 or higher.

---

### Step 2: Create pyproject.toml

**File:** `pyproject.toml` (new file)

**Key decisions:**
- **Python version:** Bump from 3.7 to 3.10 (EOL June 2023 → still supported)
- **Dependencies:** Migrate from `requirements.txt`, remove Python 2 compat (`six`, `future`), replace `tb-nightly` with `tensorboard`
- **Build system:** Keep setuptools for Cython extension compatibility
- **Ruff config:** Match/Update existing flake8/isort settings ((Follow current best practices and add new line length: 120), match what is good etc.)

**Dependencies to migrate:**
- Core: `numpy`, `Cython`, `h5py`, `Pillow`, `scipy`, `opencv-python`, `matplotlib`, `yacs`, `gdown`, `imageio`, `chardet`
- PyTorch: `torch>=2.0`, `torchvision>=0.15` (add explicit versions)
- Remove: `six`, `future` (Python 2 compat)
- Replace: `tb-nightly` → `tensorboard>=2.10`
- Remove from main deps: `flake8`, `yapf`, `isort` (move to dev extras)

**Action items:**
1. Create `pyproject.toml` with project metadata
2. Add all dependencies with appropriate version constraints
3. Configure ruff to match existing linting rules
4. Configure pytest for future test suite

**Verification:** File should parse correctly with `uv project info`

---

### Step 3: Simplify setup.py

**File:** `setup.py` (modify existing)

**Current state:** Full setup.py with package discovery, requirements reading, version extraction

**Target state:** Minimal setup.py that only handles Cython extension build

**Changes:**
- Remove `readme()`, `find_version()`, `get_requirements()` functions
- Remove `find_packages()` - handled by pyproject.toml
- Keep only Cython extension definition
- Use `Extension` from `setuptools` instead of `distutils.extension` (distutils deprecated in Python 3.12)

**Code structure:**
```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(ext_modules))
```

**Verification:** `python setup.py build_ext --inplace` should still build Cython extension

---

### Step 4: Remove Legacy Linting Files

**Files to delete:**
- `requirements.txt` (dependencies now in pyproject.toml)
- `.flake8` (replaced by ruff config in pyproject.toml)
- `.isort.cfg` (replaced by ruff's isort integration)
- `.style.yapf` (replaced by ruff format)
- `linter.sh` (replaced by `uv run ruff check .` and `uv run ruff format .`)

**Action items:**
1. Delete each file
2. Verify no other scripts reference these files (grep for `linter.sh`, `.flake8`, etc.)

**Verification:** 
- `git status` should show deleted files
- No broken references in codebase

---

### Step 5: Generate uv.lock

**Commands:**
```bash
# Generate lock file
uv lock

# Sync dependencies (install everything)
uv sync --all-extras
```

**What this does:**
- `uv lock`: Resolves all dependencies and creates `uv.lock` with exact versions
- `uv sync`: Installs dependencies into `.venv` (or system if `--system` flag used)
- `--all-extras`: Installs dev dependencies (ruff, pytest) and optional dependencies

**Verification:**
- `uv.lock` file should be created
- `.venv/` directory should exist with installed packages
- `uv run python -c "import torchreid; print(torchreid.__version__)"` should work

---

### Step 6: Test Installation

**Test commands:**
```bash
# Test basic import
uv run python -c "import torchreid; print(torchreid.__version__)"

# Test Cython extension
uv run python torchreid/metrics/rank_cylib/test_cython.py

# Test main script
uv run python scripts/main.py --help
```

**Expected results:**
- Import succeeds, prints version 1.4.0
- Cython test passes (if test file exists)
- Main script shows help message

**If failures occur:**
- Check `uv.lock` for dependency resolution issues
- Verify Cython extension built correctly
- Check Python version compatibility

---

### Step 7: Test Ruff Linting

**Commands:**
```bash
# Check for linting issues (don't auto-fix yet)
uv run ruff check .

# Check import sorting
uv run ruff check . --select I

# Format code (optional, can do later)
uv run ruff format . --check
```

**Expected results:**
- Ruff should report issues (we'll fix in later phases)
- No crashes or configuration errors

**Action items:**
- Note any ruff configuration issues
- Compare ruff output with old flake8/isort output (if available)
- Adjust ruff config in pyproject.toml if needed

---

### Step 8: Update Documentation

**Files to update:**

#### README.md
- Replace installation section:
  ```bash
  # Old
  pip install -r requirements.txt
  python setup.py develop
  
  # New
  uv sync
  # or
  pip install -e .
  ```

- Update linting section:
  ```bash
  # Old
  bash linter.sh
  
  # New
  uv run ruff check .
  uv run ruff format .
  ```

#### CLAUDE.md
- Update "Installation" section with uv commands
- Update "Linting" section with ruff commands
- Add note about Python 3.10+ requirement

**Verification:**
- Documentation reflects new workflow
- All example commands work when copy-pasted

---

### Step 9: Update .gitignore

**Add to `.gitignore`:**
```
# uv
.venv/
uv.lock  # Optional: some projects commit this, some don't
```

**Decision needed:** Should `uv.lock` be committed?
- **Yes (recommended):** Ensures reproducible builds, all developers get same versions
- **No:** More flexible, but can lead to "works on my machine" issues

**Recommendation:** Commit `uv.lock` for reproducibility.

---

### Step 10: Final Verification

**Complete test suite:**
```bash
# 1. Clean install test
rm -rf .venv uv.lock
uv sync --all-extras
uv run python -c "import torchreid; print('OK')"

# 2. Cython extension test
uv run python torchreid/metrics/rank_cylib/test_cython.py

# 3. Linting test
uv run ruff check . --output-format=github

# 4. Script functionality test
uv run python scripts/main.py --help

# 5. Verify old files are gone
test ! -f requirements.txt && echo "OK: requirements.txt removed"
test ! -f linter.sh && echo "OK: linter.sh removed"
test ! -f .flake8 && echo "OK: .flake8 removed"
```

**Success criteria:**
- ✅ All imports work
- ✅ Cython extension builds and runs
- ✅ Ruff runs without errors
- ✅ Main scripts work
- ✅ Legacy files removed
- ✅ Documentation updated

---

## Rollback Plan

If something goes wrong:

1. **Restore from git:**
   ```bash
   git checkout -- requirements.txt .flake8 .isort.cfg .style.yapf linter.sh setup.py
   git restore setup.py  # if modified
   ```

2. **Remove new files:**
   ```bash
   rm -f pyproject.toml uv.lock
   rm -rf .venv
   ```

3. **Reinstall old way:**
   ```bash
   pip install -r requirements.txt
   python setup.py develop
   ```

---

## Potential Issues & Solutions

### Issue 1: Cython extension won't build
**Symptom:** `python setup.py build_ext` fails  
**Solution:** 
- Verify `numpy` is installed before building
- Check `include_dirs` path is correct
- May need to set `NUMPY_INCLUDE` environment variable

### Issue 2: Dependency conflicts
**Symptom:** `uv lock` fails or resolves incompatible versions  
**Solution:**
- Check version constraints in pyproject.toml
- May need to relax some constraints (e.g., `numpy>=1.21,<2.0`)
- Check for conflicting transitive dependencies

### Issue 3: Ruff config doesn't match old linting
**Symptom:** Ruff reports different issues than flake8/isort  
**Solution:**
- Adjust `tool.ruff.lint.select` and `tool.ruff.lint.ignore` in pyproject.toml
- Can run both tools side-by-side initially to compare
- Ruff is generally stricter, which is good for code quality

### Issue 4: Missing dependencies
**Symptom:** Import errors after migration  
**Solution:**
- Check `requirements.txt` for any dependencies not in pyproject.toml
- Verify optional dependencies (export extras) are included if needed
- Check for implicit dependencies (e.g., `torch` might need specific CUDA version)

---

## Post-Migration Checklist

- [ ] `pyproject.toml` created with all dependencies
- [ ] `setup.py` simplified to Cython-only
- [ ] `uv.lock` generated and committed
- [ ] Legacy linting files deleted
- [ ] Ruff configured and tested
- [ ] Documentation updated (README.md, CLAUDE.md)
- [ ] `.gitignore` updated
- [ ] All verification tests pass
- [ ] Cython extension builds successfully
- [ ] Main scripts work with `uv run`

---

## Next Steps After Phase 1

Once Phase 1 is complete:
1. **Phase 2:** Fix NumPy compatibility issues (can use ruff to find `np.int`, `np.float`, `np.bool`)
2. **Phase 3:** Fix PyTorch compatibility (`torch.utils.model_zoo` → `torch.hub`)
3. **Phase 4:** Remove Python 2 compatibility code
4. **Phase 5:** Add test suite

---

## Notes

- **Backward compatibility:** Keep `setup.py` for now (even if minimal) to support `pip install -e .` workflows
- **Gradual migration:** Can keep both `requirements.txt` and `pyproject.toml` temporarily if needed, but clean up before Phase 2
- **Team coordination:** If working with others, coordinate the migration to avoid merge conflicts
- **CI/CD:** Update any CI scripts to use `uv sync` instead of `pip install -r requirements.txt`
