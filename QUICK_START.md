# PKD Quick Start Guide

## Installation & Setup

```bash
cd /path/to/PHY_knowledge_distillation_v2
```

---

## Training

```bash
python pkd/example.py train
```

**What happens**:
- Creates PKD model with:
  - AR(10) process with stable PACF parameterization (kappa_max=0.95)
  - Gaussian innovation with learnable σ
  - Compositional encoder for config variables
- Trains on 100 log-AR(1) sequences
- Saves best model to `pkd_model.pt`
- Early stopping with patience=3

**Expected output**:
```
PKDModel initialized with gaussian innovation
PACFToAR initialized with kappa_max=0.950 for stability

Epoch 1/10
Training: [progress bar]
Train Loss: 2.3456
Val Loss: 2.2987
Saved best model with val loss 2.2987

...

Early stopping triggered after X epochs
Training complete. Best model saved to pkd_model.pt
```

---

## Evaluation

```bash
python pkd/example.py eval
```

**What happens**:
- Loads trained model from `pkd_model.pt`
- Generates teacher sequence (log-AR(1))
- Generates student sequence (model rollout)
- Evaluates 3 levels of fidelity:
  1. Marginal distribution (CCDF, QQ plot)
  2. Temporal dependence (ACF, PSD)
  3. Innovation structure (PIT calibration)

**Expected output**:
```
PKD Model Fidelity Evaluation

Loaded checkpoint from pkd_model.pt
  Trained for X epochs
  Best validation loss: Y.YYYY

Model parameter diagnostics:
  Sample mean (m): 2.1234
  Sample AR coeffs (phi): [0.45 0.23 ...]
  Sample sigma: 0.67

Student sequence validation:
  ✓ All values finite and positive
  Unique values: 1998 / 2000
  Mean: 8.45, Std: 5.23

--- 1. Marginal Distribution Fidelity ---
Kolmogorov-Smirnov test: statistic=0.0523, p-value=0.4532

--- 2. Temporal Dependence ---
ACF RMSE (excluding lag 0): 0.0823

--- 3. Innovation Structure ---
=== Innovation Diagnostics ===
Ljung-Box test (ε_t): statistic=18.23, p-value=0.5678
PIT uniformity KS test: statistic=0.0634, p-value=0.7821

Evaluation complete! Check generated PNG files.
```

**Generated files**:
- `eval_marginal.png`: CCDF, QQ plot, quantile error
- `eval_temporal.png`: ACF, PSD comparison
- `eval_innovations.png`: PIT histogram and ACF

---

## Model Configuration

### Default (Gaussian Innovation)

```python
model = PKDModel(
    num_channel_models=5,
    num_mcs=10,
    num_nss=4,
    ar_order=10,
    hidden_dim=128,
    kappa_max=0.95,       # Keep PACF away from ±1
    innovation_type='gaussian',  # Simple, stable
    min_sigma=0.1         # Minimum sigma for stability
)
```

### Alternative (Flow Innovation)

```python
model = PKDModel(
    num_channel_models=5,
    num_mcs=10,
    num_nss=4,
    ar_order=10,
    hidden_dim=128,
    kappa_max=0.95,
    innovation_type='flow',  # More flexible
    num_flow_bins=16,
    flow_tail_bound=5.0
)
```

---

## Key Parameters

### AR Stability: `kappa_max`
- **Default**: 0.95
- **Range**: 0.90 (very stable) to 0.98 (more flexible)
- **Lower**: More stable rollout, less memory
- **Higher**: More flexible, can model longer memory

### Innovation: `innovation_type`
- **`'gaussian'`** (default): Simple, stable, learnable variance
- **`'flow'`**: Flexible, can model non-Gaussian distributions

### Gaussian: `min_sigma`
- **Default**: 0.1
- **Range**: 0.05 to 0.5
- **Lower**: Allows lower variance (may be unstable)
- **Higher**: More conservative (may underfit)

---

## Troubleshooting

### Training Issues

**NaN loss**:
```
Non-finite loss detected: nan
```
**Fix**: Check log transform clipping, increase `min_sigma`

**Unstable AR**:
```
AR coefficients sum to 1.234 > 1.0
```
**Fix**: Lower `kappa_max` to 0.90

**Early stopping too soon**:
```
Early stopping triggered after 4 epochs
```
**Fix**: Increase `early_stopping_patience` or `num_epochs`

### Evaluation Issues

**Degenerate sequence**:
```
Unique values: 2 / 2000
```
**Fix**: Retrain with latest fixes (history ordering, soft saturation, AR stability)

**Poor temporal correlation**:
```
ACF RMSE: 0.5432
```
**Fix**: Increase `kappa_max` for more flexibility, train longer

**Non-uniform PIT**:
```
PIT uniformity KS test: statistic=0.234, p-value=0.001
```
**Fix**: Consider `innovation_type='flow'` for non-Gaussian innovations

---

## Directory Structure

```
pkd/
├── model/
│   ├── pkd_model.py      # Main model
│   ├── encoder.py        # Compositional encoder
│   ├── heads.py          # Parameter heads
│   ├── ld.py             # PACF → AR conversion
│   ├── innovation.py     # Gaussian/Flow innovations (NEW)
│   └── flow.py           # Spline flow (kept for flow innovation)
├── train.py              # Training loop
├── infer.py              # Inference engine
├── example.py            # Training & evaluation
├── per_lut.py            # PER lookup table
└── ...

Outputs:
├── pkd_model.pt          # Trained model checkpoint
├── eval_marginal.png     # Marginal distribution plots
├── eval_temporal.png     # Temporal dependence plots
└── eval_innovations.png  # Innovation diagnostics plots
```

---

## Documentation

- **[GAUSSIAN_INNOVATION.md](GAUSSIAN_INNOVATION.md)**: Gaussian innovation model details
- **[AR_STABILITY_FIXES.md](AR_STABILITY_FIXES.md)**: AR stability improvements
- **[CRITICAL_FIXES.md](CRITICAL_FIXES.md)**: History ordering, clipping fixes
- **[TRAINING_FIXES.md](TRAINING_FIXES.md)**: Log transform, model saving fixes

---

## Common Workflows

### 1. Quick Test
```bash
# Train for 5 epochs
python pkd/example.py train  # edit num_epochs=5 in code

# Evaluate
python pkd/example.py eval
```

### 2. Production Training
```python
# In example.py, modify:
num_epochs=50
early_stopping_patience=10
train_sequences = [...]  # 1000+ sequences
```

### 3. Hyperparameter Tuning
```python
# Try different kappa_max
for kappa in [0.90, 0.93, 0.95, 0.97]:
    model = PKDModel(..., kappa_max=kappa)
    # train and evaluate
    # compare ACF RMSE

# Try different min_sigma
for min_sigma in [0.05, 0.1, 0.2]:
    model = PKDModel(..., min_sigma=min_sigma)
    # train and evaluate
```

### 4. Compare Gaussian vs. Flow
```python
# Train both
model_gaussian = PKDModel(..., innovation_type='gaussian')
model_flow = PKDModel(..., innovation_type='flow')

# Evaluate and compare marginal KS, ACF RMSE, PIT
```

---

## Performance Expectations

### Training Time (CPU, 100 sequences)
- **Gaussian**: ~5-8 minutes
- **Flow**: ~8-12 minutes

### Typical Metrics (After Proper Training)
- **Marginal KS statistic**: < 0.10
- **ACF RMSE**: < 0.15
- **PIT KS p-value**: > 0.05
- **Unique values**: ~2000 / 2000 (continuous)

### Signs of Good Training
```
✓ Train loss decreases steadily
✓ Val loss < Train loss (normal with regularization)
✓ |kappa| max < kappa_max (constraint working)
✓ phi sum < 1.0 (stable AR)
✓ sigma: 0.3-1.5 (reasonable variance)
```

---

## Next Steps

1. ✅ **Train**: `python pkd/example.py train`
2. ✅ **Evaluate**: `python pkd/example.py eval`
3. 📊 **Check plots**: Open `eval_*.png` files
4. 🔧 **Tune**: Adjust `kappa_max`, `min_sigma` if needed
5. 🚀 **Deploy**: Use `PKDInference` for packet-level simulation

---

## Quick Reference: All Fixes Applied

✅ **History ordering**: State array reversed to match training
✅ **Soft saturation**: Tanh instead of hard clip
✅ **AR stability**: PACF constrained to |κ| < 0.95
✅ **Log clipping**: Prevent -inf in training data
✅ **Best model saving**: Load best epoch, not last
✅ **Gaussian innovation**: Learnable σ, simpler than flow
✅ **Validation checks**: Fail-fast on non-finite values

**Result**: Stable, continuous-valued stochastic process with proper temporal correlations!
