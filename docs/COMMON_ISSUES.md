# Common Issues and Pitfalls

This document catalogs known issues, common mistakes, and gotchas in the PKD codebase. Use this as a reference to avoid repeating past mistakes.

---

## Index Convention Mismatches

### MCS Indexing (FIXED 2026-02-03)

**Problem:** Confusion between 0-indexed and 1-indexed MCS values

**Symptoms:**
- Embedding lookup errors when MCS=0
- Model predictions inconsistent across MCS levels
- Training examples don't match data file ranges

**Root Cause:**
- Data files use **0-indexed MCS** (0-9)
- Original code assumed **1-indexed MCS** (1-10) and subtracted 1
- Mixed conventions between data generation and model code

**Solution:**
- MCS is **always 0-indexed** (0-9) throughout the codebase
- No conversion needed in `encoder.py`
- See `data/README.md` for canonical indexing specification

**Files to Check:**
- `pkd/model/encoder.py:184` - Should use `config_dict['MCS']` directly
- `pkd/example.py` - Random MCS should use `randint(0, 10)`
- Any new data loading code - verify MCS is used as-is from MAT files

**Prevention:**
```python
# ✓ CORRECT: MCS is 0-indexed (0-9)
config = {
    'MCS': torch.tensor(5)  # Valid: 0-9
}
mcs_idx = config['MCS']  # No conversion needed

# ✗ WRONG: Don't subtract 1
mcs_idx = config['MCS'] - 1  # This assumes 1-indexed data (incorrect!)
```

---

### N_ss Indexing (Current Behavior)

**Status:** Working as intended

**Convention:**
- N_ss is **1-indexed** (1-4) in data files (physical meaning: number of streams)
- Converted to **0-indexed** in `encoder.py` for embedding lookup
- This is correct because N_ss represents a count, not an ID

**Code Pattern:**
```python
# In encoder.py:
nss_idx = config_dict['N_ss'] + (-1)  # Convert 1-4 → 0-3 for embeddings
```

**Rule of Thumb:**
- **Counts/physical quantities**: Usually 1-indexed in data (N_t, N_r, N_ss)
- **IDs/categorical labels**: Usually 0-indexed in data (channel_model_id, MCS)
- **Always check** `data/README.md` for canonical specification

---

## AR Process Stability

### PACF Coefficient Explosion

**Problem:** AR process becomes unstable when PACF coefficients approach ±1

**Symptoms:**
- Generated sequences explode to infinity or collapse to zero
- NaN/Inf values in `gamma_eff` predictions
- Training divergence despite low training loss
- Student sequence has only 2 unique values (degenerate)
- AR coefficients sum to > 1.0

**Root Cause:**
- Previous implementation: `kappa = (1 - 1e-4) * tanh(u)` allowed `|kappa| < 0.9999`
- When `|kappa|` approaches 1:
  - Roots approach unit circle (near unit-root process)
  - Variance → ∞, autocorrelation decays slowly
  - Numerical errors compound exponentially in rollout
  - Small perturbations cause large deviations

**Solution:**
- Constrain PACF: `kappa = kappa_max * torch.tanh(κ_raw)`
- Recommended `kappa_max` values:
  - **0.90**: Very stable, conservative (use if rollout still unstable)
  - **0.95**: Balanced (default, good for most cases)
  - **0.98**: More flexible, allows longer memory (use carefully)

**Files:**
- `pkd/model/ld.py:71` - PACF parameterization with kappa_max
- `pkd/model/pkd_model.py:26` - Added kappa_max parameter
- `pkd/train.py:115-119` - Enhanced diagnostics

**Diagnostic Checks During Training:**
```
AR coefficients:
  phi sum: min=0.23, max=0.78       ← Should be < 1
  |phi| sum: min=0.56, max=0.89     ← Should be < 1
  |kappa| max: 0.94                 ← Should be < kappa_max (0.95)
```

**Good signs:**
- `|kappa| max < kappa_max` (constraint working)
- `phi sum` and `|phi| sum < 1` (stable AR)

**Bad signs:**
- `|kappa| max ≈ kappa_max` (hitting constraint, need more flexibility)
- `phi sum > 1` or `|phi| sum > 1` (UNSTABLE!)

**Tuning kappa_max:**

| Use Case | kappa_max | Reasoning |
|----------|-----------|-----------|
| First attempt | 0.95 | Good balance |
| Unstable rollout | 0.90 | Prioritize stability |
| Short memory data | 0.90 | Don't need long memory |
| Long memory data | 0.97 | Need flexibility |
| Production system | 0.93 | Conservative but flexible |

**Prevention:**
```python
# Always validate generated sequences
student_arr = np.asarray(student_seq)
assert np.all(np.isfinite(student_arr)), "Non-finite values detected"
assert np.all(student_arr > 0), "Non-positive SINR values"

# Check diversity (not degenerate)
unique_ratio = len(np.unique(np.round(student_arr, 6))) / len(student_arr)
assert unique_ratio > 0.9, f"Low diversity: {unique_ratio:.2%}"
```

---

## Innovation Distribution Issues

### Normalizing Flow Complexity (Replaced 2025-02-01)

**Problem:** Normalizing flows were overkill for innovation modeling

**Symptoms:**
- Slower training
- Numerical instability in flow inverse
- PIT histogram showed poor calibration

**Solution:**
- Replaced with learnable Gaussian: `ε ~ N(0, σ²(config))`
- Much simpler, faster, and more stable
- See `GAUSSIAN_INNOVATION.md` for rationale

**Current Best Practice:**
- Use `innovation_type='gaussian'` (default)
- Set `min_sigma=0.1` to prevent numerical issues
- Only consider flows if Gaussian innovation shows poor PIT calibration

---

## Data Loading Pitfalls

### HDF5 MAT File Reading

**Problem:** MATLAB's HDF5 format has quirks

**Common Issues:**

1. **Complex Numbers:**
   ```python
   # ✗ WRONG: Assuming complex array
   data = f['gamma_eff'][:]  # May be compound dtype

   # ✓ CORRECT: Check dtype first
   data = f['gamma_eff'][:]
   if data.dtype.names:  # Compound type
       complex_data = data['real'] + 1j * data['imag']
   ```

2. **Transpose Issues:**
   ```python
   # MATLAB stores column-major, may need transpose
   gamma_eff = f['gamma_eff'][:].T  # Check shape!
   ```

3. **Field Name Mismatches:**
   - Actual field: `packet_error` (not `error_packet`)
   - Actual field: `packet_abs_error` (not `error_abs_packet`)
   - Always check with `list(f.keys())` first

**Prevention:**
- Use `data/README.md` as single source of truth for field names
- Write validation script to check all files have expected structure
- Log loaded shapes during debugging

---

## Configuration Tensor Handling

### In-Place Modification Errors

**Problem:** PyTorch doesn't allow in-place modification of leaf tensors

**Symptoms:**
```python
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

**Root Cause:**
```python
# ✗ WRONG: In-place subtraction
mcs_idx = config_dict['MCS']
mcs_idx -= 1  # Modifies original tensor!
```

**Solution:**
```python
# ✓ CORRECT: Create new tensor
mcs_idx = config_dict['MCS'] + (-1)  # New tensor, not in-place
```

**Prevention:**
- Never use `*=`, `+=`, `-=` on config tensors
- Always create new tensors with binary operators
- See `TRAINING_FIXES.md` for more details

---

## History Ordering Mismatch (CRITICAL)

### AR Coefficients Applied to Wrong Lags

**Problem:** Training and inference used different history orderings

**Symptoms:**
- Student sequence becomes degenerate (only 2 values)
- Stable dynamics → unstable/oscillatory dynamics
- Process rapidly diverges even with stable PACF

**Root Cause:**
```python
# Training (train.py:40):
X_hist = X_seq[t-ar_order:t][::-1]  # [X_{t-1}, X_{t-2}, ..., X_{t-p}]

# Inference (OLD, BROKEN):
state_array = np.array(list(self.state_buffer))  # [X_{t-p}, ..., X_{t-1}]
mu = params.c + np.dot(params.phi, state_array)  # WRONG ORDER!
```

**Impact:**
- AR coefficient φ₁ applied to lag-p instead of lag-1
- φ_p applied to lag-1 instead of lag-p
- Completely changes dynamics, breaks stationarity guarantees

**Solution (infer.py:153):**
```python
# CRITICAL FIX: Reverse to match training order
state_array = np.array(list(self.state_buffer))[::-1]  # [X_{t-1}, ..., X_{t-p}]
mu = params.c + np.dot(params.phi, state_array)  # CORRECT!
```

**Files:**
- `pkd/infer.py:153` - Fixed in `step()`
- `pkd/infer.py:250` - Fixed in `run_with_time_skipping()`

**Prevention:**
```python
# Always verify history order matches training
from collections import deque
d = deque([1, 2, 3, 4], maxlen=4)  # Oldest to newest
# Training expects: [4, 3, 2, 1] (newest to oldest)
state_array = np.array(list(d))[::-1]
```

---

## Hard Clipping Creates Point Masses (CRITICAL)

### Degenerate Distribution from Boundary Saturation

**Problem:** Hard clipping at boundaries creates exact point masses, destroying continuous distribution

**Symptoms:**
- Student sequence has exactly 2 unique values
- Values are exp(-5) ≈ 0.0067 and exp(10) ≈ 22026.46
- CCDF, ACF, PSD all completely wrong
- KS test p-value ≈ 0 (distributions clearly different)

**Root Cause:**
```python
# OLD (BROKEN):
X_t = np.clip(X_t, -5, 10)  # Creates point masses at -5 and 10!
```

**Impact:**
- When process saturates → exact boundary values
- P(X_t = -5) = P(X_t < -5) > 0 (point mass)
- P(X_t = 10) = P(X_t > 10) > 0 (point mass)
- NOT a continuous distribution!
- Process oscillates deterministically between boundaries

**Solution (infer.py:167-173):**
```python
# Soft saturation using tanh (no point masses)
B = 10.0
X_t = B * np.tanh(X_t / B)

# Fail-fast check
if not np.isfinite(X_t):
    raise RuntimeError(f"X_t non-finite: {X_t}")
```

**Why soft saturation works:**
- `tanh(x)` smoothly saturates: `tanh(x) → ±1` as `x → ±∞`
- No exact boundary values → continuous distribution preserved
- Still provides numerical protection for extreme values
- Example: `tanh(100/10) * 10 = 9.99` (not exactly 10.0)

**Files:**
- `pkd/infer.py:167-173` - Soft saturation in `step()`
- `pkd/infer.py:171-173,262-264` - Fail-fast checks

**Test:**
```python
import numpy as np
B = 10.0
for x in [-100, -20, -5, 0, 5, 20, 100]:
    soft = B * np.tanh(x / B)
    hard = np.clip(x, -5, 10)
    print(f"x={x:4.0f}: soft={soft:6.2f}, hard={hard:6.2f}")
# Output:
# x=-100: soft= -9.99, hard= -5.00  ← No exact point mass!
# x= 100: soft=  9.99, hard= 10.00  ← No exact point mass!
```

**Prevention:**
- Never use hard `np.clip()` for continuous distributions
- Always use soft saturation (tanh, sigmoid, softplus)
- Validate sequence has high diversity (not 2 values!)

---

## Sequence Generation Validation

### Non-Finite Value Detection

**Problem:** Generated sequences can have NaN/Inf due to AR instability or numerical errors

**Symptoms:**
- Evaluation crashes with "Student sequence contains non-finite values"
- Loss becomes NaN during training
- Generated SINR values are all identical (degenerate)

**Validation Checklist:**
```python
# After generating student_seq:
student_arr = np.asarray(student_seq)

# 1. Check for non-finite values
assert np.all(np.isfinite(student_arr)), "Non-finite values detected"

# 2. Check for positive values (SINR must be > 0)
assert np.all(student_arr > 0), "Non-positive SINR values"

# 3. Check for diversity (not all identical)
unique_ratio = len(np.unique(np.round(student_arr, 6))) / len(student_arr)
assert unique_ratio > 0.9, f"Low diversity: {unique_ratio:.2%}"

# 4. Check reasonable range
assert np.min(student_arr) > 1e-6, "SINR too small"
assert np.max(student_arr) < 1e6, "SINR too large"
```

**Files:**
- `pkd/example.py:466-493` - Reference validation implementation

---

## Model Training Pitfalls

### Validation Loop Configuration Mismatch

**Problem:** Validation configs not properly batched or formatted

**Symptoms:**
- Training succeeds but validation fails
- Shape mismatches in validation loop
- Inconsistent loss between train/val

**Solution:**
- Ensure val_configs match train_configs format exactly
- Use same preprocessing for both train and val
- Validate config tensor shapes before forward pass

**Prevention:**
```python
# Create validation configs with same structure as training
val_configs = [{
    'channel_model_id': 0,
    'N_t': 4,
    'N_r': 4,
    'BW': 20.0,
    'SNR_bar': 15.0,
    'MCS': np.random.randint(0, 10),  # Same range as training!
    'N_ss': np.random.randint(1, 5)
} for _ in range(num_val)]
```

---

## Quick Reference: Before You Code

### Starting a New Feature?

1. **Check this document** - Is there a known pitfall?
2. **Check `CHANGELOG.md`** - Has this been fixed before?
3. **Check `data/README.md`** - Verify data format assumptions
4. **Check existing tests** - What edge cases are covered?

### Found a Bug?

1. **Document it here** - Add to relevant section
2. **Update `CHANGELOG.md`** - Record the fix with files/dates
3. **Add test case** - Prevent regression
4. **Update prevention guide** - Help future developers

### Code Review Checklist

- [ ] No in-place modifications of config tensors
- [ ] MCS used as 0-indexed (no -1 conversion)
- [ ] N_ss converted to 0-indexed for embeddings
- [ ] Generated sequences validated for finiteness
- [ ] AR stability constraints in place (kappa_max)
- [ ] MAT file field names match `data/README.md`

---

## Related Documentation

- [CHANGELOG.md](../CHANGELOG.md) - Chronological record of all changes
- [data/README.md](../data/README.md) - Canonical data format specification
- [AR_STABILITY_FIXES.md](../AR_STABILITY_FIXES.md) - Deep dive on AR stability
- [GAUSSIAN_INNOVATION.md](../GAUSSIAN_INNOVATION.md) - Innovation design decisions
- [TRAINING_FIXES.md](../TRAINING_FIXES.md) - Training process improvements

---

**Last Updated:** 2026-02-03
**Maintained By:** Project contributors
**How to Contribute:** When you fix a bug or discover a gotcha, add it here with context!
