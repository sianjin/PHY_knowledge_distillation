# Changelog

All notable changes, fixes, and issues for the Physical-Layer Knowledge Distillation project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed - 2026-02-03

#### MCS Indexing Correction
**Issue:** Code assumed MCS is 1-indexed (1-10), but actual data uses 0-indexed MCS (0-9)

**Root Cause:**
- Original implementation in `encoder.py` converted MCS by subtracting 1: `mcs_idx = config_dict['MCS'] + (-1)`
- Training examples generated random MCS in range [1, 11) instead of [0, 10)
- This mismatch would cause embedding lookup errors or incorrect conditioning

**Files Changed:**
- `pkd/model/encoder.py:162` - Updated docstring to specify 0-indexed MCS
- `pkd/model/encoder.py:184` - Removed `-1` offset: now uses `config_dict['MCS']` directly
- `pkd/model/pkd_model.py:32` - Clarified num_mcs parameter documentation
- `pkd/example.py:63,75` - Fixed random MCS generation: `randint(1,11)` → `randint(0,10)`
- `pkd/example.py:83,402` - Added clarifying comments
- `pkd/tests/test_components.py:87,95,107` - Added comments
- `pkd/per_lut.py:62` - Updated docstring

**Testing Required:**
- [ ] Verify embedding layer accepts MCS values 0-9 without errors
- [ ] Retrain model with corrected MCS indexing
- [ ] Validate that MCS=0 and MCS=9 produce different predictions

**Related Issues:** #N/A (discovered during data inspection)

**Prevention:** See `docs/COMMON_ISSUES.md` section on "Index Convention Mismatches"

---

### Added - 2026-02-03

#### Data Structure Documentation
**What:** Created comprehensive documentation for MAT file dataset structure

**Files Added:**
- `data/README.md` - Complete data structure specification including:
  - File naming conventions
  - All 6 dataset fields (config, gamma_eff, packet_error, packet_abs_error, runtime_sec, seed)
  - Python loading examples with h5py
  - Dataset statistics and coverage
  - Clear specification that MCS is 0-indexed, N_ss is 1-indexed

**Purpose:**
- Single source of truth for data format
- Prevents index confusion in future development
- Onboarding reference for new contributors

---

## Template for Future Entries

When documenting changes, use this format:

```markdown
### [Fixed/Added/Changed/Deprecated/Removed] - YYYY-MM-DD

#### Brief Title
**Issue/Feature:** One-line description

**Root Cause:** (for bugs)
Why did this happen? What was the misunderstanding?

**Files Changed:**
- `path/to/file.py:line` - What changed and why

**Testing Required:** (if applicable)
- [ ] Test case 1
- [ ] Test case 2

**Related Issues:** Link to GitHub issues or discussion

**Prevention:** Reference to documentation that prevents recurrence
```

---

## Guidelines for Changelog Maintenance

1. **Update immediately** when fixing bugs or making changes
2. **Be specific** - include file paths and line numbers
3. **Explain the "why"** - future you needs context
4. **Link to documentation** - point to prevention strategies
5. **Track testing** - checkbox what needs validation
6. **Date everything** - track when issues were discovered/fixed

---

## Previous Changes (Migrated from existing docs)

### Fixed - 2025-02-01

#### AR Process Stability Issues (CRITICAL)
**Root Cause:** PACF coefficients allowed to approach ±1 (up to ±0.9999), causing near unit-root behavior

**Solution:**
- Replaced `pacf_delta=1e-4` with `kappa_max=0.95` parameter
- Constrained PACF: `kappa = kappa_max * tanh(u)` ensures `|kappa| < 0.95`
- Added stability diagnostics: `|kappa| max`, `phi sum`, `|phi| sum`

**Files Changed:**
- `pkd/model/ld.py:71` - New kappa_max parameterization
- `pkd/model/pkd_model.py:26` - Added kappa_max parameter (default 0.95)
- `pkd/train.py:115-119` - Enhanced AR coefficient diagnostics
- `pkd/infer.py:84-106` - Added stability warnings
- `pkd/example.py:22,83,371` - Added kappa_max to configs

**Impact:** Prevents rollout divergence, ensures strongly stationary AR process

**Prevention:** See `docs/COMMON_ISSUES.md` - "AR Process Stability" section

---

#### Student Model Collapse (CRITICAL)
**Root Cause:** Three critical bugs caused degenerate two-valued sequences

**Bugs Fixed:**
1. **History Ordering Mismatch**
   - Training used reversed history: `[X_{t-1}, ..., X_{t-p}]`
   - Inference used forward history: `[X_{t-p}, ..., X_{t-1}]`
   - Fix: `pkd/infer.py:153,250` - Added `[::-1]` to reverse state array

2. **Hard Clipping Point Masses**
   - `np.clip(X_t, -5, 10)` created exact point masses at boundaries
   - Fix: `pkd/infer.py:167-173` - Replaced with soft saturation `B * tanh(X_t/B)`

3. **Missing Validation**
   - No checks for non-finite or degenerate sequences
   - Fix: `pkd/example.py:433-453` - Added comprehensive validation

**Files Changed:**
- `pkd/infer.py:153,250` - Fixed history ordering
- `pkd/infer.py:167-173` - Soft saturation instead of hard clip
- `pkd/example.py:433-453` - Added validation checks

**Impact:** Fixed degenerate sequences (2 unique values → ~2000 unique values)

**Prevention:** See `docs/COMMON_ISSUES.md` - "Sequence Generation Validation" section

---

#### Gaussian Innovation Implementation
**Root Cause:** Normalizing flows were overcomplicated for most use cases

**Solution:**
- Created modular `innovation.py` with `GaussianInnovation` and `FlowInnovation`
- Default to Gaussian: `innovation_type='gaussian'`, `min_sigma=0.1`
- Simpler (8K params vs 27K), faster (~3x), more stable

**Files Changed:**
- `pkd/model/innovation.py` - NEW: Modular innovation models
- `pkd/model/pkd_model.py:27` - Added `innovation_type` parameter
- `pkd/example.py:25,88,396` - Default to Gaussian innovation

**Impact:** Simpler, faster training with comparable performance

**Prevention:** See `docs/ARCHITECTURE.md` - "Why Gaussian Innovation" section

---

#### Training Process Fixes (CRITICAL)
**Root Cause:** Multiple training pipeline bugs

**Bugs Fixed:**
1. **Log Transform Without Clipping**
   - `np.log(seq)` could produce `-inf` for zero values
   - Fix: `pkd/train.py:10-31` - Added `np.maximum(seq, 1e-12)` before log

2. **Incomplete Non-Finite Detection**
   - Only checked `isnan`, not `isinf`
   - Fix: `pkd/train.py:95-125` - Changed to `isfinite` with full diagnostics

3. **Wrong Model Saved**
   - Saved last epoch instead of best epoch
   - Fix: `pkd/train.py:149-174` - Reload best checkpoint before returning

**Files Changed:**
- `pkd/train.py:10-31` - Log clipping with validation
- `pkd/train.py:95-125` - Comprehensive non-finite diagnostics
- `pkd/train.py:149-174` - Save model_config, reload best model

**Impact:** Prevents training crashes, ensures best model is saved

**Prevention:** See `docs/COMMON_ISSUES.md` - "Data Loading Pitfalls" section

---

#### FiLM Notation Consistency
**Root Cause:** Implementation used `gamma` instead of paper's `alpha` for FiLM parameters

**Solution:**
- Renamed `gamma_net` → `alpha_net` to match paper Equation 19
- Updated all documentation and comments

**Files Changed:**
- `pkd/model/encoder.py:128-133` - Renamed variables to match paper
- `pkd/README.md` - Updated FiLM equation

**Impact:** No functional change, improved paper-code consistency

**Reference:** PAPER_CONSISTENCY_REPORT.md (now archived)

---

**Note:** For detailed technical write-ups of major changes, create separate markdown files (e.g., `AR_STABILITY_FIXES.md`) and link them here. The CHANGELOG should be scannable, while detailed docs can be verbose.
