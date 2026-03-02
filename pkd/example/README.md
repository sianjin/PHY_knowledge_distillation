# PKD Example Package

This package contains refactored code from the original monolithic `example.py` (1473 lines), organized into logical modules for better maintainability and readability.

## Structure

```
example/
├── __init__.py           # Package exports
├── sgn_cdf.py           # SGN CDF implementation for PIT (121 lines)
├── data_loader.py       # Data loading from .mat files (118 lines)
├── utils.py             # Utility functions: ACF, PSD, Ljung-Box (32 lines)
├── evaluation.py        # Evaluation functions (269 lines)
├── plotting.py          # Plotting and figure generation (454 lines)
├── main.py              # Main entry points (503 lines)
└── README.md            # This file
```

## Module Descriptions

### sgn_cdf.py
- `sgn_pdf()`: Compute SGN probability density function
- `sgn_cdf()`: Compute SGN cumulative distribution function (numerical)
- Critical for PIT (Probability Integral Transform) computation with SGN innovations

### data_loader.py
- `load_real_data()`: Load PHY simulator data from .mat files
- Handles 70/10/20 train/val/test split
- Converts log-scale data to linear scale

### utils.py
- `compute_acf()`: Compute autocorrelation function
- `compute_psd()`: Compute power spectral density
- `ljung_box_test()`: Ljung-Box test for autocorrelation

### evaluation.py
- `evaluate_marginal_distribution()`: Compare marginal distributions
- `evaluate_temporal_dependence()`: Compare ACF and PSD
- `evaluate_innovation_structure()`: Evaluate PIT calibration

### plotting.py
- `evaluate_test_set()`: Compute test set metrics
- `generate_figure1_per_mcs_metrics()`: Per-MCS metrics vs SNR
- `generate_figure2_quantile_error()`: Quantile error analysis
- `generate_figure3_ccdf_error()`: CCDF error analysis

### main.py
- `example_training()`: Training with real PHY simulator data
- `example_evaluation()`: Qualitative evaluation on single test sequence
- `example_test_evaluation()`: Comprehensive test set evaluation

## Usage

**Prerequisites**: Ensure you have a Python environment with PyTorch installed. See [requirements.txt](../requirements.txt) for dependencies.

The main entry point is `example.py`. Run all commands from the `pkd/` directory:

```bash
cd pkd

# Training
python example.py train                   # Train with all real data files
python example.py train 5                 # Train with first 5 files
python example.py train --exclusion-config example/training_exclusions.yaml  # Train with configuration exclusions

# Testing
python example.py test                                      # Auto-select most common slice
python example.py test 5                                    # Test with 5 files
python example.py test --slice N_t:4 N_r:2                  # Specify configuration slice
python example.py test --slice channel_model_id:2 N_t:4 N_r:2 BW:20.0 N_ss:2  # Full slice spec
python example.py test --slice N_t:3 N_r:2 MCS:7            # Test held-out configuration

# Qualitative Evaluation
python example.py eval                    # Evaluate first test sequence
python example.py eval 50                 # Evaluate 51st test sequence
python example.py eval 50 10              # Evaluate 51st test seq (10-file subset)
```

## Training Examples

### 1. Train with Real PHY Simulator Data

Train with the real data from the `data/` folder:

```bash
cd pkd
python example.py train
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

### 2. Train with Subset of Real Data

To train with only a few files (useful for quick testing or debugging):

```bash
cd pkd
python example.py train 5
```

This loads only the first 5 .mat files (alphabetically sorted).

**Note**: With the 70/10/20 split, training uses only 70% of sequences, validation 10%, and test 20%.

### 3. Training with Configuration Exclusions (Generalization Testing)

PKD supports **selective exclusion** of training data based on specific (config_slice, MCS, SNR) combinations to test model generalization on held-out configurations.

#### Why Use Exclusions?

When training on all available data, the model learns parameters for all configurations. To test whether the model can **generalize** to unseen configurations, you can:
1. Exclude specific configurations from training (e.g., MCS 7 for a particular antenna setup)
2. Train the model without seeing those configurations
3. Evaluate on the held-out configurations to measure generalization performance

#### Quick Start

1. **Create an exclusion configuration file** (or use the provided example):

```bash
# Use the provided example
cp pkd/example/training_exclusions.yaml my_exclusions.yaml

# Edit to specify your exclusions
# (see YAML Schema section below for details)
```

2. **Train with exclusions** (run from `pkd/` directory):

```bash
cd pkd
python example.py train --exclusion-config example/training_exclusions.yaml
```

3. **Test on held-out configuration**:

```bash
# Test only on the excluded configuration
python example.py test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:1 MCS:7
```

#### YAML Configuration Schema

See [training_exclusions.yaml](training_exclusions.yaml) for a fully documented example.

**Basic Structure**:
```yaml
version: "1.0"

settings:
  apply_to_train: true    # Exclude from training
  apply_to_val: true      # Exclude from validation
  apply_to_test: false    # Keep test untouched

exclusions:
  - name: "Descriptive name"
    description: "Why this exclusion?"
    config:
      channel_model_id: 2   # Config slice to match
      N_t: 3
      N_r: 2
      BW: 20.0
      N_ss: 1
    mcs_list: [7]           # MCS values to exclude
    snr_filter:             # Optional SNR filtering
      type: "range"
      min: 10.0
      max: 25.0
```

**SNR Filtering Options**:

1. **No SNR filter** (default) - Exclude at ALL SNR values:
   ```yaml
   mcs_list: [7]
   # No snr_filter → applies to all SNRs
   ```

2. **Specific SNR values** - Exclude only at certain SNR points:
   ```yaml
   snr_filter:
     type: "values"
     values: [15, 20, 25]  # Only these SNR values (dB)
   ```

3. **SNR range** - Exclude in a range [min, max]:
   ```yaml
   snr_filter:
     type: "range"
     min: 10.0   # SNR >= 10
     max: 25.0   # SNR <= 25
   ```

4. **SNR threshold** - Exclude above/below a threshold:
   ```yaml
   snr_filter:
     type: "threshold"
     operator: "gte"  # >= (or "lte", "gt", "lt")
     value: 20.0
   ```

#### Example Use Cases

**Example 1**: Test generalization to high MCS
```yaml
exclusions:
  - name: "Exclude MCS 8-9 for 2x2 MIMO"
    config:
      N_t: 2
      N_r: 2
      BW: 20.0
      N_ss: 2
    mcs_list: [8, 9]
```

**Example 2**: Test generalization at specific SNR
```yaml
exclusions:
  - name: "Exclude MCS 5 at high SNR"
    config:
      channel_model_id: 2
      N_t: 4
      N_r: 2
      BW: 20.0
      N_ss: 2
    mcs_list: [5]
    snr_filter:
      type: "values"
      values: [25, 30]  # Only at 25 and 30 dB
```

**Example 3**: Test generalization in operating range
```yaml
exclusions:
  - name: "Exclude MCS 7 in typical SNR range"
    config:
      channel_model_id: 2
      N_t: 3
      N_r: 2
      BW: 20.0
      N_ss: 1
    mcs_list: [7]
    snr_filter:
      type: "range"
      min: 15.0
      max: 25.0
```

#### CLI Examples

Note: Run these commands from the `pkd/` directory.

```bash
# Train with exclusions
cd pkd
python example.py train --exclusion-config example/training_exclusions.yaml

# Train subset of files with exclusions
python example.py train 10 --exclusion-config my_exclusions.yaml

# Train normally (no exclusions - backward compatible)
python example.py train

# Test on held-out configuration (note: MCS now supported in --slice!)
python example.py test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:1 MCS:7

# Test with auto-selected slice
python example.py test
```

#### What Happens During Training?

When you train with exclusions:

1. **Data is loaded** normally (all .mat files)
2. **Exclusion rules are applied** to training and validation sets
3. **Statistics are reported**:
   ```
   EXCLUSION FILTER REPORT
   =====================================
   TRAIN SET:
     Total sequences before: 3500
     Total sequences after:  3450
     Excluded sequences:     50 (1.43%)

     Exclusion rules applied (1):
       [  50 seqs] Exclude MCS7 for Model-B 3x2:1 config
                   Config: channel_model_id:2, N_t:3, N_r:2, BW:20.0, N_ss:1
                   MCS:    [7]
   ```
4. **Model is trained** on filtered data
5. **Test set remains untouched** (for unbiased evaluation)

#### Best Practices

1. **Start simple**: Exclude one configuration first to verify the approach
2. **Check statistics**: Ensure you're not excluding too much data (>50% triggers warning)
3. **Document your exclusions**: Use descriptive names and descriptions in YAML
4. **Version control**: Commit your exclusion configs alongside code
5. **Test generalization**: Use `--slice MCS:X` to evaluate on held-out configurations

#### Configuration Parameter Reference

- `channel_model_id`: int (1-6)
  1=Model-A, 2=Model-B, 3=Model-C, 4=Model-D, 5=Model-E, 6=Model-F

- `N_t`: int (1-8) - Number of transmit antennas

- `N_r`: int (1-8) - Number of receive antennas

- `BW`: float - Bandwidth in MHz (typically 20.0, 40.0, 80.0, 160.0)

- `N_ss`: int (1-4) - Number of spatial streams (≤ min(N_t, N_r))

- `MCS`: int (0-9) - Modulation and Coding Scheme
  0=lowest rate (most robust), 9=highest rate (least robust)

- `SNR_bar`: float - Average SNR in dB (typical range: -10 to +63)

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
python example.py test
```

This will:
1. Load the trained model from `pkd_model.pt`
2. Load the **held-out test set** (20% of sequences, same 70/10/20 split as training)
3. **Filter to a configuration slice** to ensure clean regime behavior without mixing heterogeneous configs
4. Evaluate all test sequences in the filtered slice
5. Generate three comprehensive evaluation figures:
   - `fig1_per_mcs_metrics.png` - Per-MCS metrics vs SNR
   - `test_quantile_error.png` - Quantile error analysis
   - `test_ccdf_error.png` - CCDF error analysis

#### Configuration Slicing

**PKD v1** filters test evaluation to a single configuration slice (fixed antenna config, bandwidth, channel model, etc.) to ensure figures show clean regime behavior without mixing heterogeneous configurations. Only MCS and SNR_bar vary within the slice.

**Auto-selection (default)**:
```bash
python example.py test
```
Automatically selects the most common configuration slice in your test set.

**Manual slice specification**:
```bash
# Specify partial slice (other params auto-selected)
python example.py test --slice N_t:4 N_r:2

# Full slice specification
python example.py test --slice channel_model_id:2 N_t:4 N_r:2 BW:20.0 N_ss:2 packet_length:1000
```

**Available slice parameters**:
- `channel_model_id` (int): Channel model ID (e.g., 0-4, where 4=Model-D)
- `N_t` (int): Number of transmit antennas (e.g., 2, 4, 8)
- `N_r` (int): Number of receive antennas (e.g., 2, 4, 8)
- `BW` (float): Bandwidth in MHz (e.g., 20.0, 40.0, 80.0)
- `N_ss` (int): Number of spatial streams (e.g., 1, 2, 4)
- `MCS` (int): ✨ **NEW** - Modulation and Coding Scheme (0-9) for testing held-out configurations

**Notes**:
- `MCS` parameter is now supported for filtering to specific MCS values (useful for generalization testing)
- SNR_bar is typically the varying dimension for analysis within a slice
- `packet_length` is always 1000 bytes in the current dataset and does not need to be specified

**Examples**:
```bash
# Evaluate 4x2 MIMO configuration
python example.py test --slice N_t:4 N_r:2

# Evaluate Model-D channel with 20 MHz bandwidth
python example.py test --slice channel_model_id:4 BW:20.0

# Evaluate 2 spatial streams on 4x4 MIMO
python example.py test --slice N_t:4 N_r:4 N_ss:2

# Combine with file limit
python example.py test 10 --slice N_t:4 N_r:2
```

**Important Notes**:
- Uses the same random seed (42) as training to ensure consistent train/val/test split
- Test set is completely held-out data that the model never saw during training
- If you trained with a subset of files (e.g., `train 10`), use the same number for testing (e.g., `test 10`)
- The slice filter ensures sufficient (MCS, SNR) coverage for meaningful analysis

```bash
# If you trained with 10 files:
python example.py train 10
# Then test with the same 10 files:
python example.py test 10
# Or with specific slice:
python example.py test 10 --slice N_t:4 N_r:2
```

### 2. Evaluate Single Test Sequence (Qualitative Analysis)

Evaluate the trained model on a single sequence from the **held-out test set** for detailed qualitative analysis:

```bash
cd pkd
python example.py eval
```

This uses the first test sequence (index 0 from the test set).

To specify which test sequence to inspect:

```bash
# Inspect the 51st test sequence
cd pkd
python example.py eval 50
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
python example.py eval 0      # First test sequence
python example.py eval 100    # 101st test sequence
python example.py eval 999    # Last test sequence (if using all files)

# If you trained with only 10 files
python example.py eval 50 10  # 51st test sequence from 10-file subset
```

**Important**: This mode correctly uses the **held-out test set only** (20% of data), ensuring you're inspecting sequences the model never saw during training. The test set is loaded using the same 70/10/20 split and random seed (42) as training.

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

# Train with real data
model = example_training(
    data_dir='data',
    max_files=None  # None = load all files
)

# Train with subset of files
model = example_training(
    data_dir='data',
    max_files=10
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

# Auto-select most common slice (recommended)
results = example_test_evaluation(data_dir='data')

# With manual slice specification
slice_spec = {
    'channel_model_id': 2,  # int: 0-4 (4=Model-D)
    'N_t': 4,               # int: number of transmit antennas
    'N_r': 2,               # int: number of receive antennas
    'BW': 20.0,             # float: bandwidth in MHz
    'N_ss': 2               # int: number of spatial streams
}
results = example_test_evaluation(data_dir='data', slice_spec=slice_spec)

# With subset of files (must match training)
results = example_test_evaluation(data_dir='data', max_files=10, slice_spec=slice_spec)

# Access results
print(f"Slice label: {results['slice_label']}")
print(f"Evaluated sequences: {len(results['test_metrics'])}")
print(f"Figure 1 results: {results['fig1_results'].keys()}")
```

### Single Test Sequence Evaluation (Qualitative)

```python
from pkd.example import example_evaluation

# Evaluate first test sequence
example_evaluation(data_dir='data', test_idx=0)

# Evaluate different test sequences
example_evaluation(data_dir='data', test_idx=100)

# With subset of files (must match training)
example_evaluation(data_dir='data', test_idx=50, max_files=10)
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

Note: Run all commands from the `pkd/` directory (`cd pkd`).

| Task | Command |
|------|---------|
| **Train with real data (all files)** | `python example.py train` |
| **Train with real data (5 files)** | `python example.py train 5` |
| **Train with exclusions** | `python example.py train --exclusion-config example/training_exclusions.yaml` |
| **Evaluate on test set (auto-slice)** | `python example.py test` |
| **Evaluate with specific slice** | `python example.py test --slice N_t:4 N_r:2` |
| **Evaluate with full slice spec** | `python example.py test --slice channel_model_id:2 N_t:4 N_r:2 BW:20.0 N_ss:2` |
| **Evaluate held-out MCS (NEW)** | `python example.py test --slice N_t:3 N_r:2 MCS:7` |
| **Evaluate on test set (subset)** | `python example.py test 5 --slice N_t:4 N_r:2` |
| **Evaluate single test sequence** | `python example.py eval 50` |
| **Evaluate test sequence (subset)** | `python example.py eval 50 10` |

### Understanding Test Sequence Indices

When using `eval`, you specify which test sequence to inspect from the held-out test set:

**Command format**: `python example.py eval [test_idx] [max_files]`

**Valid ranges**:
- **test_idx**: Index within the test set
  - For all files (50 .mat files): **0-999** (approximately 1,000 test sequences)
  - For subset (e.g., 10 files): **0-199** (approximately 200 test sequences)
  - The exact number depends on the 70/10/20 split of your data
- **max_files** (optional): Must match the number used during training

**Examples**:
- `python example.py eval` → First test sequence (test_idx=0)
- `python example.py eval 100` → 101st test sequence
- `python example.py eval 999` → Last test sequence (with all files)
- `python example.py eval 50 10` → 51st test sequence (when trained with 10 files)

**Important**:
- This now correctly uses the **held-out test set only** (20% of data)
- Uses the same random seed (42) and split as training
- Ensures you're inspecting data the model never saw during training
- For comprehensive quantitative evaluation, use `python example.py test` instead

## Benefits of Refactoring

1. **Better Organization**: Related functions grouped into logical modules
2. **Easier Navigation**: Each module is 100-500 lines instead of 1473 lines
3. **Improved Maintainability**: Changes to evaluation don't affect plotting, etc.
4. **Reusability**: Modules can be imported independently
5. **Clear Dependencies**: Import statements show module relationships

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
3. You're running the script with: `cd pkd && python example.py train`

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
2. **Load fewer files**: Use `python example.py train 10` to limit to 10 files
3. **Use smaller model**: Reduce `hidden_dim` in the model configuration

## Migration from Original

The original `example.py` has been backed up as `example.py.backup`. The new `example.py` is a thin entry point that delegates to the refactored modules.

All functionality is preserved - the refactoring only changes organization, not behavior.
