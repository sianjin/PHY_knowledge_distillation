# PKD Usage Examples

This document provides practical examples for training and evaluating the PKD model.

## Training Examples

**Prerequisites**: Ensure you have a Python environment with PyTorch installed. See [requirements.txt](requirements.txt) for dependencies.

### 1. Quick Test with Dummy Data

Train the model with synthetically generated AR(1) sequences (fastest, for testing):

```bash
cd pkd
python example.py train
```

Note: Use `python` (or `python3`) depending on your environment setup.

This will:
- Generate 100 training sequences and 20 validation sequences
- Use random configurations
- Train for 10 epochs
- Save the best model to `pkd_model.pt`

### 2. Train with Real PHY Simulator Data

Train with the real data from the `data/` folder:

```bash
cd pkd
python example.py train-real
```

This will:
- Automatically locate the `data/` directory (relative to the script location)
- Load all .mat files from the data directory
- Randomly shuffle sequences (seed=42) before splitting
- Use 80% of sequences for training, 20% for validation
- Extract real configurations from the data files
- Train for 10 epochs
- Save the best model to `pkd_model.pt`

**Path Resolution**: The script automatically finds the `data/` directory regardless of where you run it from. The path is resolved relative to the script location (`pkd/example.py` → `../data/`).

### 3. Train with Subset of Real Data

To train with only a few files (useful for quick testing or debugging):

```bash
cd pkd
python example.py train-real 5
```

This loads only the first 5 .mat files (alphabetically sorted).

## Data Format

### Real Data Structure

The .mat files in `data/` contain:
- **gamma_eff**: Shape (1000, 100) - 100 sequences of 1000 time steps each
  - Stored in **LOG scale** in the file
  - Converted to **LINEAR scale** by `load_real_data()`
- **config**: Shape (7, 100) - Configuration for each sequence
  - `[0]` channel_model_id (4 = Model-D)
  - `[1]` N_t (number of transmit antennas)
  - `[2]` N_r (number of receive antennas)
  - `[3]` BW (bandwidth in MHz)
  - `[4]` SNR_bar (average SNR in dB)
  - `[5]` MCS (modulation and coding scheme, 0-indexed)
  - `[6]` N_ss (number of spatial streams)

### Loaded Data Format

After loading with `load_real_data()`:
- Each sequence is in **LINEAR scale** (positive values)
- Format: `List[np.ndarray]` where each array has shape `(1000,)`
- Each config is a dictionary with keys: `channel_model_id`, `N_t`, `N_r`, `BW`, `SNR_bar`, `MCS`, `N_ss`
- **Data is randomly shuffled** before train/val split (default `random_seed=42`)
  - Ensures both train and validation sets have representative samples from all configurations
  - Prevents bias from sequential ordering in files
  - Reproducible with the same random seed

## Evaluation

### 1. Evaluate with Synthetic Data

Evaluate the trained model using a synthetically generated teacher sequence:

```bash
cd pkd
python3 example.py eval
```

This will:
1. Load the trained model from `pkd_model.pt`
2. Generate a synthetic log-AR(1) teacher sequence
3. Generate a student sequence using the PKD model
4. Compare and generate evaluation plots:
   - `eval_marginal.png` - Marginal distribution fidelity (CCDF, QQ plot, quantile error)
   - `eval_temporal.png` - Temporal correlation (ACF, PSD)
   - `eval_innovations.png` - Innovation structure (PIT histogram, ACF)
5. Print statistical test results

### 2. Evaluate with Real Data

Evaluate the trained model using real PHY simulator data as the teacher:

```bash
cd pkd
python3 example.py eval-real
```

This uses the first sequence from the first .mat file.

To specify which file and sequence to use:

```bash
# Use file index 5, sequence index 10
cd pkd
python3 example.py eval-real 5 10
```

Parameters:
- **file_idx** (default: 0): Index of the .mat file to use (0-49 for 50 files)
- **seq_idx** (default: 0): Index of the sequence within the file (0-99 for 100 sequences per file)

This will:
1. Load the trained model from `pkd_model.pt`
2. Load the specified real teacher sequence from the data
3. Extract the actual configuration (SNR, MCS, etc.) from the data
4. Generate a student sequence using the same configuration
5. Compare teacher vs student and generate the same evaluation plots

## Python API Usage

### Loading Real Data

```python
from pkd.example import load_real_data

# Load all data with random shuffling (default seed=42)
train_seqs, train_configs, val_seqs, val_configs = load_real_data('data')

# Load first 10 files only
train_seqs, train_configs, val_seqs, val_configs = load_real_data(
    data_dir='data',
    train_ratio=0.8,
    max_files=10
)

# Use different random seed for different split
train_seqs, train_configs, val_seqs, val_configs = load_real_data(
    data_dir='data',
    train_ratio=0.8,
    random_seed=123  # Different seed = different train/val split
)

# No shuffling (sequential split - not recommended)
train_seqs, train_configs, val_seqs, val_configs = load_real_data(
    data_dir='data',
    train_ratio=0.8,
    random_seed=None  # None = no shuffling
)

print(f"Loaded {len(train_seqs)} training sequences")
print(f"First config: {train_configs[0]}")
print(f"First sequence shape: {train_seqs[0].shape}")
```

### Training Programmatically

```python
from pkd.example import example_training

# Train with dummy data
model = example_training(use_real_data=False)

# Train with real data
model = example_training(
    use_real_data=True,
    data_dir='data',
    max_files=10  # Optional: limit number of files
)
```

### Custom Training Loop

```python
import torch
from pkd.model.pkd_model import PKDModel
from pkd.train import train_pkd
from pkd.example import load_real_data

# Load data
train_seqs, train_configs, val_seqs, val_configs = load_real_data('data')

# Create model
model = PKDModel(
    num_channel_models=5,
    num_mcs=10,
    num_nss=4,
    ar_order=10,
    hidden_dim=128,
    kappa_max=0.95,
    innovation_type='gaussian',
    min_sigma=0.1
)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_config = {
    'num_channel_models': 5,
    'num_mcs': 10,
    'num_nss': 4,
    'ar_order': 10,
    'hidden_dim': 128,
    'kappa_max': 0.95,
    'innovation_type': 'gaussian',
    'min_sigma': 0.1
}

trained_model = train_pkd(
    model,
    train_seqs,
    train_configs,
    val_seqs,
    val_configs,
    num_epochs=20,
    batch_size=256,
    lr=1e-3,
    device=device,
    early_stopping_patience=5,
    model_config=model_config
)
```

### Custom Evaluation

```python
from pkd.example import example_evaluation

# Evaluate with synthetic data
example_evaluation(use_real_data=False)

# Evaluate with real data (file 0, sequence 0)
example_evaluation(use_real_data=True, data_dir='data', file_idx=0, seq_idx=0)

# Evaluate with different file and sequence
example_evaluation(use_real_data=True, data_dir='data', file_idx=10, seq_idx=50)
```

## Dataset Statistics

Current dataset (as of 2026-02-03):
- **50 .mat files** covering:
  - MCS 0-4
  - SNR ranges: 0-18 dB (MCS0), 4-22 dB (MCS1), 8-26 dB (MCS2), 12-30 dB (MCS3), 16-34 dB (MCS4)
- **5,000 total sequences** (100 per file)
- **5,000,000 time steps** (1000 per sequence)
- Fixed parameters: Model-D, 4x2x2, BW=20MHz

Using all files:
- Training: 4,000 sequences (80%)
- Validation: 1,000 sequences (20%)

## Quick Reference

| Task | Command |
|------|---------|
| **Train with dummy data** | `python3 example.py train` |
| **Train with real data (all files)** | `python3 example.py train-real` |
| **Train with real data (5 files)** | `python3 example.py train-real 5` |
| **Evaluate with synthetic data** | `python3 example.py eval` |
| **Evaluate with real data (default)** | `python3 example.py eval-real` |
| **Evaluate with specific file/sequence** | `python3 example.py eval-real 5 10` |

### Understanding File and Sequence Indices

When using `eval-real`:
- **file_idx**: Which .mat file to use (sorted alphabetically)
  - `0` = `CBW20_Model-D_4-by-2-by-2_MCS0_SNR0.mat`
  - `1` = `CBW20_Model-D_4-by-2-by-2_MCS0_SNR10.mat`
  - etc.
- **seq_idx**: Which of the 100 sequences in that file (0-99)

Example: `python3 example.py eval-real 10 50` evaluates using:
- The 11th .mat file (sorted alphabetically)
- The 51st sequence (out of 100) from that file

## Troubleshooting

### "Found 0 .mat files" Error

If you see this error:
```
Found 0 .mat files
Total loaded: 0 sequences
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**Solution**: The script uses automatic path resolution. Make sure:
1. Your `data/` directory is at the project root (same level as `pkd/`)
2. The data directory contains `.mat` files
3. You're running the script with: `cd pkd && python example.py train-real`

The script automatically resolves the path to `../data/` from its location at `pkd/example.py`.

### "ModuleNotFoundError: No module named 'torch'" Error

**Solution**: Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install PyTorch separately following the instructions at [pytorch.org](https://pytorch.org/).

### Memory Issues with Large Datasets

If you encounter out-of-memory errors when loading all files:

1. **Reduce batch size**: Edit the `batch_size` parameter in the training function
2. **Load fewer files**: Use `python example.py train-real 10` to limit to 10 files
3. **Use smaller model**: Reduce `hidden_dim` in the model configuration
