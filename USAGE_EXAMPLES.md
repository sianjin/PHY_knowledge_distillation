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
- Use 70% of sequences for training, 10% for validation, 20% for test
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

**Note**: With the 70/10/20 split, training uses only 70% of sequences, validation 10%, and test 20%.

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

### 1. Evaluate on Held-Out Test Set (Recommended)

**This is the proper way to evaluate your trained model quantitatively.**

After training with real data, evaluate on the held-out test set (20% of data, same split as training):

```bash
cd pkd
python3 example.py test
```

This will:
1. Load the trained model from `pkd_model.pt`
2. Load the **held-out test set** (20% of sequences, same 70/10/20 split as training)
3. Use PyTorch DataLoader to efficiently evaluate **all ~1,000 test sequences**
4. Compute aggregate metrics:
   - Average test loss (negative log-likelihood)
   - Average log-likelihood across all test samples
   - Total number of test sequences and samples evaluated

**Important Notes**:
- Uses the same random seed (42) as training to ensure consistent train/val/test split
- Test set is completely held-out data that the model never saw during training
- If you trained with a subset of files (e.g., `train-real 10`), use the same number for testing (e.g., `test 10`)

```bash
# If you trained with 10 files:
python3 example.py train-real 10
# Then test with the same 10 files:
python3 example.py test 10
```

### 2. Evaluate with Synthetic Data (Quick Test)

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

### 3. Evaluate Single Test Sequence (Qualitative Analysis)

Evaluate the trained model on a single sequence from the **held-out test set** for detailed qualitative analysis:

```bash
cd pkd
python3 example.py eval-real
```

This uses the first test sequence (index 0 from the test set).

To specify which test sequence to inspect:

```bash
# Inspect the 51st test sequence
cd pkd
python3 example.py eval-real 50
```

Parameters:
- **test_idx** (default: 0): Index within the test set
  - **Valid range: 0 to (num_test_sequences - 1)**
  - For all files: **0-999** (approximately 1,000 test sequences with 70/10/20 split)
  - For subset: depends on how many files were used in training
- **max_files** (optional): Should match the number used in training if you trained on a subset

**Examples**:
```bash
# Inspect different test sequences
python3 example.py eval-real 0      # First test sequence
python3 example.py eval-real 100    # 101st test sequence
python3 example.py eval-real 999    # Last test sequence (if using all files)

# If you trained with only 10 files
python3 example.py eval-real 50 10  # 51st test sequence from 10-file subset
```

**Important**: This mode now correctly uses the held-out test set (20% of data), ensuring you're inspecting sequences the model never saw during training. The test set is loaded using the same 70/10/20 split and random seed (42) as training.

This will:
1. Load the trained model from `pkd_model.pt`
2. Load the specified real teacher sequence from the data
3. Extract the actual configuration (SNR, MCS, etc.) from the data
4. Generate a student sequence using the same configuration
5. Compare teacher vs student and generate detailed evaluation plots:
   - `eval_marginal.png` - Marginal distribution fidelity
   - `eval_temporal.png` - Temporal correlation
   - `eval_innovations.png` - Innovation structure

## Python API Usage

### Loading Real Data

```python
from pkd.example import load_real_data

# Load all data with random shuffling (default seed=42)
# Returns: train, val, test sets (70/10/20 split)
train_seqs, train_configs, val_seqs, val_configs, test_seqs, test_configs = load_real_data('data')

# Load first 10 files only
train_seqs, train_configs, val_seqs, val_configs, test_seqs, test_configs = load_real_data(
    data_dir='data',
    train_ratio=0.7,
    val_ratio=0.1,
    max_files=10
)

# Use different random seed for different split
train_seqs, train_configs, val_seqs, val_configs, test_seqs, test_configs = load_real_data(
    data_dir='data',
    train_ratio=0.7,
    val_ratio=0.1,
    random_seed=123  # Different seed = different train/val/test split
)

# No shuffling (sequential split - not recommended)
train_seqs, train_configs, val_seqs, val_configs, test_seqs, test_configs = load_real_data(
    data_dir='data',
    train_ratio=0.7,
    val_ratio=0.1,
    random_seed=None  # None = no shuffling
)

print(f"Loaded {len(train_seqs)} training sequences")
print(f"Loaded {len(val_seqs)} validation sequences")
print(f"Loaded {len(test_seqs)} test sequences")
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

# Load data (returns train/val/test with 70/10/20 split)
train_seqs, train_configs, val_seqs, val_configs, test_seqs, test_configs = load_real_data('data')

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

### Test Set Evaluation

```python
from pkd.example import example_test_evaluation

# Evaluate on held-out test set (recommended for quantitative metrics)
test_metrics = example_test_evaluation(data_dir='data')

# With subset of files (must match training)
test_metrics = example_test_evaluation(data_dir='data', max_files=10)

print(f"Test loss: {test_metrics['test_loss']:.4f}")
print(f"Avg log-likelihood: {test_metrics['avg_log_likelihood']:.4f}")
print(f"Test sequences: {test_metrics['num_sequences']}")
```

### Single Test Sequence Evaluation (Qualitative)

```python
from pkd.example import example_evaluation

# Evaluate with synthetic data
example_evaluation(use_real_data=False)

# Evaluate first test sequence
example_evaluation(use_real_data=True, data_dir='data', test_idx=0)

# Evaluate different test sequences
example_evaluation(use_real_data=True, data_dir='data', test_idx=100)

# With subset of files (must match training)
example_evaluation(use_real_data=True, data_dir='data', test_idx=50, max_files=10)
```

## Dataset Statistics

Current dataset (as of 2026-02-03):
- **50 .mat files** covering:
  - MCS 0-4
  - SNR ranges: 0-18 dB (MCS0), 4-22 dB (MCS1), 8-26 dB (MCS2), 12-30 dB (MCS3), 16-34 dB (MCS4)
- **5,000 total sequences** (100 per file)
- **5,000,000 time steps** (1000 per sequence)
- Fixed parameters: Model-D, 4x2x2, BW=20MHz

Using all files with 70/10/20 split:
- Training: 3,500 sequences (70%)
- Validation: 500 sequences (10%)
- Test: 1,000 sequences (20%)

### Test Set Distribution

The test set (1,000 sequences) has the following distribution across MCS and SNR:

| MCS | SNR Range | # SNR Values | # Test Sequences | Avg per SNR |
|-----|-----------|--------------|------------------|-------------|
| 0   | 0-18 dB   | 10           | 189              | ~19         |
| 1   | 4-22 dB   | 10           | 191              | ~19         |
| 2   | 8-26 dB   | 10           | 213              | ~21         |
| 3   | 12-30 dB  | 10           | 204              | ~20         |
| 4   | 16-34 dB  | 10           | 203              | ~20         |

**Key points**:
- Each MCS has approximately **190-213 test sequences**
- Each (MCS, SNR) combination has approximately **14-31 test sequences**
- Distribution is balanced due to random shuffling with seed=42
- Each file originally contains 100 sequences with the same (MCS, SNR) pair
- After 70/10/20 split, each (MCS, SNR) gets roughly 20 test sequences

## Quick Reference

| Task | Command |
|------|---------|
| **Train with dummy data** | `python3 example.py train` |
| **Train with real data (all files)** | `python3 example.py train-real` |
| **Train with real data (5 files)** | `python3 example.py train-real 5` |
| **Evaluate on test set (recommended)** | `python3 example.py test` |
| **Evaluate on test set (subset)** | `python3 example.py test 5` |
| **Evaluate with synthetic data** | `python3 example.py eval` |
| **Evaluate single test sequence** | `python3 example.py eval-real 50` |
| **Evaluate test sequence (subset)** | `python3 example.py eval-real 50 10` |

### Understanding Test Sequence Indices

When using `eval-real`, you specify which test sequence to inspect from the held-out test set:

**Command format**: `python3 example.py eval-real [test_idx] [max_files]`

**Valid ranges**:
- **test_idx**: Index within the test set
  - For all files (50 .mat files): **0-999** (approximately 1,000 test sequences)
  - For subset (e.g., 10 files): **0-199** (approximately 200 test sequences)
  - The exact number depends on the 70/10/20 split of your data
- **max_files** (optional): Must match the number used during training

**Examples**:
- `python3 example.py eval-real` → First test sequence (test_idx=0)
- `python3 example.py eval-real 100` → 101st test sequence
- `python3 example.py eval-real 999` → Last test sequence (with all files)
- `python3 example.py eval-real 50 10` → 51st test sequence (when trained with 10 files)

**Important**:
- This now correctly uses the **held-out test set only** (20% of data)
- Uses the same random seed (42) and split as training
- Ensures you're inspecting data the model never saw during training
- For comprehensive quantitative evaluation, use `python3 example.py test` instead

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
