# PKD Quick Start Guide

**Goal:** Get PKD running in under 5 minutes

---

## Step 1: Installation (1 minute)

```bash
cd /path/to/PHY_knowledge_distillation
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- tqdm ≥ 4.65.0

---

## Step 2: Run Tests (1 minute)

```bash
cd pkd/tests
python test_components.py
```

**Expected output:**
```
Testing PACF stability...
✓ PACF stability test passed
Testing flow invertibility...
✓ Flow invertibility test passed
Testing AR mean consistency...
✓ AR mean consistency test passed
Testing time-skipping correctness...
✓ Time-skipping correctness test passed

✅ All tests passed!
```

---

## Step 3: Train Model (2 minutes)

```bash
python pkd/example.py train
```

**What happens:**
- Creates PKD model with Gaussian innovation
- Trains on 100 synthetic AR(1) sequences
- Saves best model to `pkd_model.pt`

**Expected output:**
```
PKDModel initialized with gaussian innovation
PACFToAR initialized with kappa_max=0.950 for stability

Epoch 1/10
Train Loss: 2.3456
Val Loss: 2.2987
Saved best model

...

Training complete. Best model saved to pkd_model.pt
```

---

## Step 4: Evaluate Model (1 minute)

```bash
python pkd/example.py eval
```

**What happens:**
- Loads trained model
- Generates teacher and student sequences
- Compares marginal distribution, temporal dependence, innovation structure

**Expected output:**
```
Student sequence validation:
  ✓ All values finite and positive
  Unique values: 1998 / 2000
  Mean: 8.45, Std: 5.23

--- 1. Marginal Distribution Fidelity ---
Kolmogorov-Smirnov test: statistic=0.0523, p-value=0.4532

--- 2. Temporal Dependence ---
ACF RMSE (excluding lag 0): 0.0823

--- 3. Innovation Structure ---
PIT uniformity KS test: statistic=0.0634, p-value=0.7821

Evaluation complete!
```

**Generated files:**
- `eval_marginal.png`: CCDF, QQ plot, quantile error
- `eval_temporal.png`: ACF, PSD comparison
- `eval_innovations.png`: PIT histogram and ACF

---

## Next Steps

### Option A: Use PKD for Inference

```python
from pkd import PKDModel, PKDInference, AWGNPERLookup
import torch

# Load model
model = PKDModel(num_channel_models=5, num_mcs=10, num_nss=4)
checkpoint = torch.load('pkd_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Setup inference
per_lut = AWGNPERLookup.create_dummy_lut(num_mcs=10)
inference = PKDInference(model, per_lut)

# Run simulation
config = {
    'channel_model_id': torch.tensor(0),
    'N_t': torch.tensor(4),
    'N_r': torch.tensor(4),
    'BW': torch.tensor(20.0),
    'SNR_bar': torch.tensor(15.0),
    'MCS': torch.tensor(5),  # 0-indexed!
    'N_ss': torch.tensor(2)
}

config_trajectory = [config] * 1000
results = inference.run_sequence(config_trajectory)

print(f"Generated {len(results['gamma_eff'])} SINR values")
print(f"PER: {results['per']}")
```

### Option B: Train on Your Data

```python
from pkd import PKDModel, train_pkd
import numpy as np

# 1. Replace with your PHY simulator output
train_sequences = [...]  # List of gamma_eff numpy arrays
train_configs = [{
    'channel_model_id': 0,
    'N_t': 4, 'N_r': 4, 'BW': 20.0,
    'SNR_bar': 15.0,
    'MCS': 5,  # 0-9 (0-indexed!)
    'N_ss': 2   # 1-4 (1-indexed)
} for _ in range(len(train_sequences))]

# 2. Create and train model
model = PKDModel(
    num_channel_models=5,
    num_mcs=10,  # MCS 0-9
    num_nss=4,
    ar_order=10,
    kappa_max=0.95,
    innovation_type='gaussian'
)

trained_model = train_pkd(
    model,
    train_sequences,
    train_configs,
    val_sequences,
    val_configs,
    num_epochs=50,
    device='cuda'
)
```

---

## Troubleshooting

### "Unique values: 2 / 2000" (Degenerate sequence)
**Fix:** Retrain with latest code (all critical fixes applied)

### "AR coefficients sum to X > 1.0"
**Fix:** Lower `kappa_max` to 0.90 for more stability

### "NaN loss during training"
**Fix:** Increase `min_sigma` or check input data for zeros

### "ACF RMSE very high"
**Fix:** Increase `kappa_max` or train longer

---

## Key Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `kappa_max` | 0.95 | 0.90-0.98 | AR stability (lower=safer) |
| `innovation_type` | 'gaussian' | 'gaussian'/'flow' | Innovation model |
| `min_sigma` | 0.1 | 0.05-0.5 | Minimum variance |
| `ar_order` | 10 | 5-20 | Temporal memory |

---

## Documentation

- **Full Guide:** [QUICK_START.md](QUICK_START.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Common Issues:** [docs/COMMON_ISSUES.md](docs/COMMON_ISSUES.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Data Format:** [data/README.md](data/README.md)

---

## Performance Expectations

### Good Training Signs
```
✓ Train loss decreases steadily
✓ |kappa| max < 0.95 (constraint working)
✓ phi sum < 1.0 (stable AR)
✓ Unique values: ~2000 / 2000 (continuous)
```

### Typical Metrics
- **Marginal KS statistic:** < 0.10
- **ACF RMSE:** < 0.15
- **PIT KS p-value:** > 0.05

---

**You're ready to use PKD!** 🎉

For detailed documentation, see [docs/README.md](docs/README.md)
