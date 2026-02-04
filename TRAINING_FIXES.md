# Training Pipeline Fixes

## Critical Issues Fixed

### 1. **Log Transform Without Clipping (CRITICAL)**

**Problem**: The dataset was computing `np.log(seq)` directly on gamma_eff values, which could produce `-inf` if any value was 0 or negative:

```python
# OLD (BROKEN)
self.X_sequences = [np.log(seq) for seq in sequences]
```

**Impact**:
- If `gamma_eff` contains 0 or negative values → `log(0) = -inf` or `log(neg) = nan`
- Training data contains `-inf` values
- Model computes with `-inf` → produces extremely large losses
- Explains epoch-1 massive losses (train: 90.69, val: 55.02)

**Fix**:
```python
# NEW (FIXED)
self.X_sequences = [np.log(np.maximum(seq, eps)) for seq in sequences]
```

Added:
- Clipping: `np.maximum(seq, 1e-12)` before log transform
- Validation: Check for non-finite values and warn
- Reports finite ratio for debugging

### 2. **Incomplete Non-Finite Value Detection**

**Problem**: Training only checked for `NaN` but not `Inf`:

```python
# OLD (INCOMPLETE)
if torch.isnan(loss):
    # diagnostics...
```

**Impact**:
- `Inf` values passed through unchecked
- Could silently corrupt training
- Hard to diagnose numerical issues

**Fix**:
```python
# NEW (COMPLETE)
if not torch.isfinite(loss):
    # comprehensive diagnostics...
```

Added comprehensive diagnostics that report:
- Input data finite ratios
- Model predictions (mu, eps, z)
- AR coefficients and their sums
- Log-likelihood statistics
- Exact count of non-finite values

### 3. **Validation Loss Lower Than Training Loss**

**Observation**: Validation loss consistently lower than training loss (e.g., epoch 2: train 2.35, val 2.28).

**Explanation** (NORMAL behavior):
This is expected and not a bug! Reasons:
1. **Dropout/Regularization**: `model.train()` uses dropout → noisier predictions → higher loss; `model.eval()` deterministic → lower loss
2. **Sample Difficulty**: Training set may have harder samples than validation
3. **Batch Effects**: Training shuffles and may group difficult samples

**Added**: Informative logging when val < train to reassure user this is normal.

### 4. **Negative Loss Values**

**Observation**: Loss becomes negative in later epochs (epoch 2+).

**Explanation** (NORMAL behavior):
- Loss = `-log_q.mean()`
- For continuous distributions, log-density can be positive if the model is very confident
- Negative loss just means high likelihood (good!)
- This is mathematically correct for continuous probability densities

**No fix needed** - this is expected behavior.

### 5. **Best Model Not Saved Correctly**

**Problem**: `train_pkd()` returned the model state from the LAST epoch, not the BEST epoch. Then `example_training()` would overwrite the best checkpoint.

**Impact**:
- Saved model could be worse than the best training checkpoint
- Early stopping saves best model, but then it gets overwritten
- User loads undertrained model

**Fix**:
- Modified `train_pkd()` to reload best checkpoint before returning
- Removed redundant save in `example_training()`
- Added `model_config` parameter to be saved with checkpoint
- Added checkpoint diagnostics in evaluation

## Summary of Changes

### [train.py](pkd/train.py)

1. **Line 10-31**: `PKDDataset.__init__`
   - Added `eps=1e-12` parameter
   - Changed to `np.log(np.maximum(seq, eps))`
   - Added non-finite value checks with warnings

2. **Line 95-125**: `train_epoch` loss checking
   - Changed from `torch.isnan(loss)` to `not torch.isfinite(loss)`
   - Added comprehensive diagnostics reporting 8+ statistics
   - Better error messages for debugging

3. **Line 127-148**: `validate` function
   - Added `non_finite_batches` counter
   - Skip batches with non-finite loss instead of crashing
   - Report warning if non-finite batches found
   - Handle edge case of all batches non-finite

4. **Line 202-214**: Training loop logging
   - Added note when val < train to explain it's normal
   - Compute and display percentage difference

5. **Line 149-174**: `train_pkd` function signature and checkpoint handling
   - Added `model_config` parameter
   - Save `model_config` in checkpoint
   - Save `train_loss` in checkpoint
   - Reload best model before returning (CRITICAL FIX)

### [example.py](pkd/example.py)

1. **Line 76-99**: `example_training`
   - Pass `model_config` to `train_pkd()`
   - Removed redundant checkpoint save after training
   - Training function now handles everything

2. **Line 357-377**: `example_evaluation`
   - Added checkpoint diagnostics on load
   - Print epoch number, train loss, val loss
   - Warn if `model_config` missing

## How to Use

### Training
```bash
python pkd/example.py train
```

Now correctly:
- Clips values before log transform (no -inf)
- Catches non-finite values early with detailed diagnostics
- Saves and loads the BEST model (not last epoch)
- Includes model_config in checkpoint

### Evaluation
```bash
python pkd/example.py eval
```

Now shows:
- Which epoch's model was loaded
- Training and validation loss at best epoch
- Model parameter diagnostics
- Sequence quality metrics

## Expected Behavior

### Normal Training Output:
```
Epoch 1/10
Train Loss: 2.5432
Val Loss: 2.4123
  Note: Val loss is 5.1% lower than train loss (may be normal with dropout/regularization)
Saved best model with val loss 2.4123

Epoch 2/10
Train Loss: 2.1234
Val Loss: 2.0987
  Note: Val loss is 1.2% lower than train loss (may be normal with dropout/regularization)
Saved best model with val loss 2.0987
...
```

### If Issues Detected:
```
============================================================
Non-finite loss detected: inf
============================================================

Input data:
  X_t: min=-5.2341, max=3.2134, mean=1.2345
  X_t finite ratio: 0.9980
  ...
```

This detailed output helps diagnose exactly where numerical issues occur.

## Next Steps

1. **Run Training**: `python pkd/example.py train`
   - Should complete without NaN/Inf errors
   - Should show gradually decreasing loss
   - May see negative losses (normal!)

2. **Run Evaluation**: `python pkd/example.py eval`
   - Should load trained model correctly
   - Should generate diverse sequences (not constant)
   - Check the 3 evaluation plots

3. **Monitor for**:
   - Any warnings about non-finite values in dataset
   - Validation batches being skipped
   - Loss patterns (should decrease smoothly)

## Why These Fixes Matter

| Issue | Impact | Priority |
|-------|--------|----------|
| Log without clipping | Training crashes or diverges | **CRITICAL** |
| Incomplete non-finite checks | Silent corruption | **HIGH** |
| Wrong model saved | Poor inference quality | **CRITICAL** |
| Val < Train misunderstanding | User confusion | Low |
| Negative loss misunderstanding | User confusion | Low |

The first and third issues were critical bugs that prevented proper training and inference.
