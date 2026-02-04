# PKD Quick Start Guide

## Installation (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Tests (1 minute)

```bash
cd pkd/tests
python test_components.py
```

Expected output:
```
✓ PACF stability test passed
✓ Flow invertibility test passed
✓ AR mean consistency test passed
✓ Time-skipping correctness test passed
✅ All tests passed!
```

## Run Example (2 minutes)

```bash
cd pkd
python example.py
```

This will:
1. Create a PKD model
2. Generate dummy effective SINR sequences
3. Run inference simulation
4. Demonstrate time-skipping

## Basic Usage

### 1. Create Model

```python
from pkd import PKDModel

model = PKDModel(
    num_channel_models=5,  # Number of channel types
    num_mcs=10,            # Number of MCS levels
    num_nss=4,             # Max spatial streams
    ar_order=10,           # AR process order
    hidden_dim=128         # Network hidden dimension
)
```

### 2. Setup Inference

```python
from pkd import PKDInference, AWGNPERLookup

# Create PER lookup table
per_lut = AWGNPERLookup.create_dummy_lut(num_mcs=10)

# Create inference engine
inference = PKDInference(
    model=model,
    per_lut=per_lut,
    ar_order=10,
    device='cuda'  # or 'cpu'
)
```

### 3. Run Simulation

```python
import torch

# Define configuration
config = {
    'channel_model_id': torch.tensor(0),
    'N_t': torch.tensor(4),
    'N_r': torch.tensor(4),
    'BW': torch.tensor(20.0),
    'SNR_bar': torch.tensor(15.0),
    'MCS': torch.tensor(5),
    'N_ss': torch.tensor(2)
}

# Simulate 1000 packets
config_trajectory = [config] * 1000
results = inference.run_sequence(config_trajectory)

# Access results
gamma_eff = results['gamma_eff']  # Effective SINR values
per = results['per']              # Packet error rates
errors = results['errors']        # Error events (0/1)
```

### 4. Time-Skipping (Efficient)

```python
# Define windows with constant config
config_windows = [
    (config, 500),   # 500 packets
    (config2, 300),  # 300 packets with different config
    (config, 200),   # 200 packets
]

# Run efficiently - network evaluated only 2 times!
results = inference.run_with_time_skipping(config_windows)
```

## Training with Your PHY Simulator

```python
from pkd import train_pkd

# 1. Generate data from your PHY simulator
train_sequences = [...]  # List of gamma_eff arrays
train_configs = [...]    # List of config dicts

# 2. Train
trained_model = train_pkd(
    model,
    train_sequences,
    train_configs,
    val_sequences,
    val_configs,
    num_epochs=50,
    batch_size=256,
    lr=1e-3,
    device='cuda'
)

# 3. Save
torch.save({
    'model_state_dict': trained_model.state_dict()
}, 'my_pkd_model.pt')

# 4. Load later
model.load_state_dict(torch.load('my_pkd_model.pt')['model_state_dict'])
```

## Key Configuration Parameters

```python
# Configuration dict structure
config = {
    # Static parameters (fixed per simulation run)
    'channel_model_id': 0,    # Your channel model ID (0 to num_channel_models-1)
    'N_t': 4,                 # Number of transmit antennas
    'N_r': 4,                 # Number of receive antennas
    'BW': 20.0,               # Bandwidth in MHz

    # Dynamic parameters (can vary with rate adaptation)
    'SNR_bar': 15.0,          # Average SNR in dB
    'MCS': 5,                 # MCS index (0 to num_mcs-1)
    'N_ss': 2,                # Number of spatial streams (1 to num_nss)
}
```

## Performance Tips

1. **Use time-skipping**: When config is constant, 100-1000x speedup
2. **Batch configurations**: Group similar configs together
3. **GPU acceleration**: Use `device='cuda'` for large models
4. **Tune AR order**: Start with p=10, increase if needed
5. **Cache size**: Monitor `len(inference.cache)` - should be reasonable

## Troubleshooting

### NaN losses during training
- Reduce learning rate
- Increase gradient clipping
- Check input normalization

### Flow invertibility errors
- Increase `min_bin_size` (default 1e-3)
- Increase `tail_bound` (default 5.0)
- Check input ranges

### Slow inference
- Verify time-skipping is working: `len(inference.cache)` should be small
- Use GPU if available
- Reduce `ar_order` if possible

## File Locations

- **Documentation**: `pkd/README.md`
- **Examples**: `pkd/example.py`
- **Tests**: `pkd/tests/test_components.py`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`

## Next Steps

1. ✅ Run tests to verify installation
2. ✅ Run example to understand workflow
3. 📝 Replace dummy data with your PHY simulator output
4. 🎯 Train on real data
5. 📊 Validate distributional fidelity
6. 🚀 Deploy in your system-level simulator

## Getting Help

- Check docstrings: All functions have detailed documentation
- Read README: `pkd/README.md` has comprehensive info
- Review paper: `PHY_Knowledge_Distillation.pdf`
- Check tests: `pkd/tests/test_components.py` shows usage patterns
