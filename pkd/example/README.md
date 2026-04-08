# PKD Example Package

Refactored modules for training and evaluating the PKD (Physical Layer Knowledge Distillation) model with real PHY simulator data.

## Quick Start

Run all commands from the **project root** (`PHY_knowledge_distillation/`) using `-m` syntax:

```bash
cd PHY_knowledge_distillation

# Basic training
python -m pkd.example train

# Training with exclusions
python -m pkd.example train --exclusion-mcs 30%                    # Random 30% MCS exclusion
python -m pkd.example train --exclusion-config pkd/example/training_exclusions.yaml  # Manual exclusions

# Testing
python -m pkd.example test                                         # Auto-select slice
python -m pkd.example test --slice N_t:4 N_r:2 MCS:7              # Specific config

# Evaluation
python -m pkd.example eval --slice N_t:3 N_r:2 MCS:7              # Slice-based selection
```

## Module Structure

```
example/
├── data_loader.py       # Load .mat files, 70/10/20 split
├── exclusion_filter.py  # Training data exclusion (manual + random)
├── evaluation.py        # Marginal, temporal, innovation metrics
├── plotting.py          # Test set figures generation
├── utils.py             # ACF, PSD, Ljung-Box test
├── sgn_cdf.py          # SGN CDF for PIT computation
└── main.py             # Training & evaluation entry points
```

---

## Training

### Basic Training

```bash
# Train with all data
python -m pkd.example train

# Train with subset (first N files)
python -m pkd.example train 10
```

**What happens:**
- Loads .mat files from `../data/` directory
- 70/10/20 train/val/test split (random seed=42)
- Trains for 10 epochs with early stopping
- Saves best model to `pkd_model.pt`

### Training with Exclusions

PKD supports excluding specific configurations from training to test generalization on held-out data.

#### Method 1: Random Percentage Exclusion (NEW)

Randomly exclude X% of MCS values per configuration slice:

```bash
# Basic: exclude 30% of MCS randomly
python -m pkd.example train --exclusion-mcs 30%

# Custom seed for reproducibility
python -m pkd.example train --exclusion-mcs 25% --exclusion-seed 12345

# Combine random + manual exclusions
python -m pkd.example train --exclusion-mcs 30% --exclusion-config pkd/example/training_exclusions.yaml
```

**How it works:**
1. Groups data by config slice: (channel_model_id, N_t, N_r, BW, N_ss)
2. For each slice, randomly selects X% of MCS values to exclude
3. Generates YAML + JSON manifest in `pkd/example/exclusions/`
4. Applies exclusions to train/val sets (test set untouched)

**Output files:**
- `exclusions_random_30pct_seed42_TIMESTAMP.yaml` - Reusable config
- `exclusions_random_30pct_seed42_TIMESTAMP.json` - Machine-readable manifest

**Console output:**
```
Generating random 30% MCS exclusions (seed=42)
  Found 15 unique configuration slices

  Slice 1/15: Model-B 3x2:1, BW=20.0MHz
    Available MCS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Excluding 3 MCS (30%): [2, 5, 7]
    Training on: [0, 1, 3, 4, 6, 8, 9]

Saved to: pkd/example/exclusions/exclusions_random_30pct_seed42_20240315_103045.yaml
```

**Reproducibility:**
```bash
# Later, reuse exact same exclusions
python -m pkd.example train --exclusion-config pkd/example/exclusions/exclusions_random_30pct_seed42_20240315_103045.yaml
```

#### Method 2: Manual YAML Exclusions

Define specific exclusion rules in YAML format:

```bash
python -m pkd.example train --exclusion-config pkd/example/training_exclusions.yaml
```

**YAML format:**
```yaml
version: "1.0"

settings:
  apply_to_train: true
  apply_to_val: true
  apply_to_test: false

exclusions:
  - name: "Exclude MCS7 for Model-B 3x2:1"
    config:
      channel_model_id: 2
      N_t: 3
      N_r: 2
      BW: 20.0
      N_ss: 1
    mcs_list: [7]
    snr_filter:  # Optional: restrict to specific SNR values/ranges
      type: "range"
      min: 15.0
      max: 25.0
```

**SNR filter options:**
- **No filter** (default): Applies to all SNR values
- **Exact values**: `type: "values"`, `values: [15, 20, 25]`
- **Range**: `type: "range"`, `min: 10.0`, `max: 25.0`
- **Threshold**: `type: "threshold"`, `operator: "gte"`, `value: 20.0`

See [training_exclusions.yaml](training_exclusions.yaml) for full examples.

---

## Testing & Evaluation

### Test Set Evaluation (Quantitative)

Evaluate on the held-out test set (20% of data):

```bash
# Auto-select most common config slice
python -m pkd.example test

# Specify config slice manually
python -m pkd.example test --slice N_t:4 N_r:2

# Test on held-out MCS (from random exclusion)
python -m pkd.example test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:1 MCS:7
```

**Generates 3 comprehensive figures:**
- `figures/test_metrics_*.png` - Per-MCS metrics vs SNR (6 files)
- `figures/test_quantile_error.png` - Quantile error analysis
- `figures/test_ccdf_error.png` - CCDF error analysis

**Slice parameters:**
- `channel_model_id`: int (1-6 for Models A-F)
- `N_t`, `N_r`: int (transmit/receive antennas)
- `BW`: float (bandwidth in MHz)
- `N_ss`: int (spatial streams)
- `MCS`: int (0-9, useful for testing held-out configs)

### Single Sequence Evaluation (Qualitative)

Detailed evaluation on a single test sequence:

```bash
# Slice-based selection (recommended)
python -m pkd.example eval --slice N_t:3 N_r:2 MCS:7           # Random selection
python -m pkd.example eval --slice N_t:3 N_r:2 MCS:7 --idx 5   # Deterministic (6th match)

# Direct indexing (legacy)
python -m pkd.example eval 50                                   # 51st test sequence
```

**Generates detailed plots:**
- `figures/eval_marginal_*.png` - Marginal distribution fidelity (3 files)
- `figures/eval_temporal_*.png` - Temporal correlation (2 files)
- `figures/eval_innovations_*.png` - Innovation structure (2 files)
- `figures/eval_teacher_baseline_*.png` - Teacher AR baseline (4 files)

---

## Testing Held-Out Configurations

After training with exclusions, test on the held-out configs:

```bash
# View which MCS were excluded
cat pkd/example/exclusions/exclusions_random_30pct_seed42_*.yaml

# Test on specific held-out MCS (from YAML or console output)
python -m pkd.example test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:1 MCS:7

# Qualitative evaluation
python -m pkd.example eval --slice channel_model_id:2 N_t:3 N_r:2 MCS:7
```

### Programmatic Testing

Parse the JSON manifest to test all held-out configs:

```python
import json

with open('pkd/example/exclusions/exclusions_random_30pct_seed42_*.json') as f:
    manifest = json.load(f)

for slice_info in manifest['held_out_slices']:
    for mcs in slice_info['excluded_mcs']:
        slice_cfg = slice_info['slice']
        cmd = f"python -m pkd.example test --slice " + \
              " ".join(f"{k}:{v}" for k, v in slice_cfg.items()) + f" MCS:{mcs}"
        print(cmd)
```

---

## Command Reference

### Training Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example train` | Train with all data |
| `python -m pkd.example train 10` | Train with first 10 files |
| `python -m pkd.example train --exclusion-mcs 30%` | Random 30% MCS exclusion |
| `python -m pkd.example train --exclusion-mcs 25% --exclusion-seed 123` | Custom seed |
| `python -m pkd.example train --exclusion-config FILE.yaml` | Manual YAML exclusions |
| `python -m pkd.example train --exclusion-mcs 30% --exclusion-config FILE.yaml` | Hybrid (random + manual) |

### Testing Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example test` | Test with auto-selected slice |
| `python -m pkd.example test --slice N_t:4 N_r:2` | Test specific antenna config |
| `python -m pkd.example test --slice N_t:3 N_r:2 MCS:7` | Test held-out MCS |
| `python -m pkd.example test 10` | Test with 10-file subset |

### Evaluation Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example eval` | Evaluate first test sequence |
| `python -m pkd.example eval 50` | Evaluate 51st test sequence |
| `python -m pkd.example eval --slice N_t:4 N_r:2 MCS:7` | Random matching sequence |
| `python -m pkd.example eval --slice N_t:4 N_r:2 MCS:7 --idx 3` | 4th matching sequence |

### Exclusion Analysis Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example exclusion` | Analyze all exclusion percentages (auto-select slice) |
| `python -m pkd.example exclusion --slice channel_model_id:2 N_t:3 N_r:2 N_ss:1 BW:20.0` | Specify config slice |
| `python -m pkd.example exclusion --slice channel_model_id:2 N_t:3 N_r:2 N_ss:1 BW:20.0 --percentages 0 30 60` | Subset of percentages |

### Runtime Evaluation Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7` | Measure runtime (random SNR) |
| `python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7 --snr 20.0` | Measure runtime (specific SNR) |

**What it does:**
- Evaluates models from `pkd/trained_models/exclude {0,30,60,70,80,90}/`
- Computes 4 key metrics across exclusion percentages
- Generates 4 plots in `figures/`:
  - `exclusion_overall_ks.png` - Overall accuracy degradation
  - `exclusion_generalization_gap.png` - Seen vs unseen MCS performance
  - `exclusion_acf_rmse.png` - Temporal correlation accuracy
  - `exclusion_pit_passrate.png` - Calibration quality

**Command-line Options:**

```bash
# Auto-select most common slice (default)
python -m pkd.example exclusion

# Specify custom configuration slice
python -m pkd.example exclusion --slice channel_model_id:2 N_t:3 N_r:2 N_ss:1 BW:20.0

# Evaluate only specific percentages
python -m pkd.example exclusion --slice channel_model_id:2 N_t:3 N_r:2 N_ss:1 BW:20.0 --percentages 0 30 60

# Custom device
python -m pkd.example exclusion --slice channel_model_id:2 N_t:3 N_r:2 N_ss:1 BW:20.0 --device cuda

# Custom data directory
python -m pkd.example exclusion --data-dir path/to/data
```

**Slice Specification (key:value pairs after `--slice`):**
- `channel_model_id`: Channel model ID (int, 0-4)
- `N_t`: Number of transmit antennas (int)
- `N_r`: Number of receive antennas (int)
- `N_ss`: Number of spatial streams (int)
- `BW`: Bandwidth in MHz (float, e.g., 20.0, 40.0)

**Note:** All slice keys must be provided together. If `--slice` is omitted, the script automatically selects the most common configuration slice from test data.

### Runtime Evaluation Commands

| Command | Description |
|---------|-------------|
| `python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7` | Runtime averaged over 10 SNRs |
| `python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7 --snr 20.0` | Runtime at specific SNR |
| `python -m pkd.example.evaluate_runtime --channel-model 2 --N-t 3 --N-r 2 --BW 20.0 --N-ss 1 --MCS 7 --num-sequences 100` | Custom sequence count |

**What it does:**
- Measures PKD model inference runtime performance
- By default: Generates 50 sequences of length 1000 at each of 10 SNR values (10-55 dB)
- Reports **average total runtime** across SNRs for fair comparison (runtime varies with SNR)
- With `--snr`: Generates sequences at single specified SNR value
- Uses a single fixed configuration (no variation except SNR)

**Command-line Arguments:**

Required:
- `--channel-model`: Channel model ID (e.g., 2 for Model B, matches data format)
- `--N-t`: Number of transmit antennas
- `--N-r`: Number of receive antennas
- `--BW`: Bandwidth in MHz
- `--N-ss`: Number of spatial streams (1-4)
- `--MCS`: Modulation & Coding Scheme (0-9)

Optional:
- `--snr`: Average SNR in dB (default: average over 10 SNRs from 10 to 55 dB)
- `--num-sequences`: Number of sequences to generate per SNR (default: 50)
- `--sequence-length`: Length of each sequence (default: 1000)
- `--model-path`: Path to trained model (default: pkd_model.pt)

**Note on Runtime Measurement:**
- Default mode (no `--snr`): Runs at 10 different SNR values (10, 15, 20, ..., 55 dB) and reports **average total runtime**
- This provides fair comparison since runtime can vary significantly with SNR
- Single SNR mode (`--snr X`): Runs only at specified SNR for targeted testing

**Example Output (Default - Average over SNRs):**
```
========================================
PKD Model Runtime Evaluation
========================================

Configuration:
  Channel Model: B (id=2)
  Antennas: N_t=3, N_r=2
  Bandwidth: 20.0 MHz
  MCS: 7, N_ss: 1
  SNR values: [10. 15. 20. 25. 30. 35. 40. 45. 50. 55.] dB
  SNR mode: average over 10 SNRs (10-55 dB)

Runtime Settings:
  Sequences per SNR: 50
  Length per sequence: 1000
  Number of SNR values: 10
  Total sequences: 500
  Device: cpu

========================================
Results (Averaged over SNRs):
========================================
  Average total runtime: 2.3456 seconds
  Std dev runtime: 0.0234 seconds
  Min runtime: 2.3001 seconds
  Max runtime: 2.4012 seconds
  Total wall-clock time: 23.5678 seconds

  Average time per sequence: 0.0469 seconds
  Average time per sample: 0.000047 seconds
  Average throughput: 21321.23 samples/second
========================================
```

**Requirements:**
- All trained models must exist in `pkd/trained_models/exclude X/pkd_model.pt`
- Exclusion manifests in `pkd/example/exclusions/exclusions_random_X%_seed42_*.json`
- Test data in `../data/` directory

**Note:** This may take several minutes to complete as it evaluates all 7 models across the full test set.

---

## Data Format

### .mat File Structure

Each .mat file contains:
- **gamma_eff**: (1000, 100) - 100 sequences × 1000 time steps
  - Stored in LOG scale, converted to LINEAR scale by data loader
- **config**: (7, 100) - Configuration per sequence
  - [0] channel_model_id (1-6 for Models A-F)
  - [1] N_t (transmit antennas)
  - [2] N_r (receive antennas)
  - [3] BW (bandwidth in MHz)
  - [4] SNR_bar (average SNR in dB)
  - [5] MCS (0-9)
  - [6] N_ss (spatial streams)

### Data Loading

```python
from pkd.example import load_real_data

train_seqs, train_cfgs, val_seqs, val_cfgs, test_seqs, test_cfgs = load_real_data(
    data_dir='data',
    train_ratio=0.7,
    val_ratio=0.1,
    max_files=None,      # None = all files
    random_seed=42       # For reproducible split
)
```

**Returns:**
- Sequences in LINEAR scale as `List[np.ndarray]`, shape (1000,)
- Configs as `List[dict]` with keys: `channel_model_id`, `N_t`, `N_r`, `BW`, `SNR_bar`, `MCS`, `N_ss`, `packet_length`
- 70/10/20 train/val/test split with random shuffling (seed=42)

---

## Configuration Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `channel_model_id` | int | 1-6 | 802.11n TGn channel models (1=A, ..., 6=F) |
| `N_t` | int | 1-8 | Number of transmit antennas |
| `N_r` | int | 1-8 | Number of receive antennas |
| `BW` | float | 20, 40, 80, 160 | Bandwidth in MHz |
| `N_ss` | int | 1-4 | Spatial streams (≤ min(N_t, N_r)) |
| `MCS` | int | 0-9 | Modulation & Coding Scheme (0=robust, 9=high rate) |
| `SNR_bar` | float | -10 to 63 | Average SNR in dB |
| `packet_length` | int | - | Data packet length in bytes (typically 1000) |

---

## Python API

### Training

```python
from pkd.example import example_training

model = example_training(
    data_dir='data',
    max_files=None,                     # None = all files
    exclusion_config=None,              # Path to YAML config
    exclusion_mcs_percentage=30.0,      # Random % exclusion
    exclusion_seed=42                   # Random seed
)
```

### Test Evaluation

```python
from pkd.example import example_test_evaluation

results = example_test_evaluation(
    data_dir='data',
    max_files=None,
    slice_spec={'N_t': 4, 'N_r': 2, 'MCS': 7}  # Optional slice filter
)
```

### Single Sequence Evaluation

```python
from pkd.example import example_evaluation

metrics = example_evaluation(
    data_dir='data',
    test_idx=50,                            # Direct indexing
    slice_spec={'N_t': 4, 'N_r': 2},       # Or slice-based
    slice_idx=3                             # Index within slice
)
```

---

## Exclusion Filter Implementation

### Random Exclusion Algorithm

The `--exclusion-mcs` feature:

1. **Groups configs** by slice: (channel_model_id, N_t, N_r, BW, N_ss)
2. **Collects MCS values** per slice from train+val sets
3. **Randomly selects** X% of MCS to exclude using numpy with seed
4. **Generates YAML** config in standard exclusion format
5. **Saves manifest** JSON for programmatic testing
6. **Applies filtering** using existing `filter_dataset_with_exclusions()` pipeline

**Key features:**
- Reproducible (fixed seed, default=42)
- Combines with manual YAML exclusions
- Zero code duplication (reuses existing filter)
- Transparent (detailed console output)
- Testable (generated YAML/JSON files)

### Modified Files

1. **pkd/example.py** - Added CLI parsing (`parse_exclusion_mcs`, `parse_exclusion_seed`)
2. **pkd/example/main.py** - Updated `example_training()` to orchestrate random exclusions
3. **pkd/example/exclusion_filter.py** - Added 5 new functions:
   - `generate_random_mcs_exclusions()` - Core algorithm
   - `save_exclusion_config()` - YAML saving
   - `merge_exclusion_configs()` - Combine random + manual
   - `save_exclusion_manifest()` - JSON manifest
   - `_get_channel_name()` - Helper (1→A, 2→B, etc.)

---

## Troubleshooting

### "Found 0 .mat files"

**Solution:** Ensure `data/` directory exists at project root with .mat files.

```bash
# Project structure
PHY_knowledge_distillation/
├── data/               # .mat files here
│   └── *.mat
└── pkd/
    └── example.py      # Run from pkd/ directory
```

### "Percentage must be between 0 and 100"

Use valid range for `--exclusion-mcs`:
```bash
# Valid
python -m pkd.example train --exclusion-mcs 30%

# Invalid
python -m pkd.example train --exclusion-mcs 150%
```

### "All training data was excluded"

Reduce exclusion percentage:
```bash
python -m pkd.example train --exclusion-mcs 20%  # Lower percentage
```

### Memory Issues

1. Reduce batch size in training
2. Load fewer files: `python -m pkd.example train 10`
3. Reduce `hidden_dim` in model config

---

## Examples

### Example 1: Train with Random Exclusion

```bash
python -m pkd.example train --exclusion-mcs 30%
```

Output: `pkd/example/exclusions/exclusions_random_30pct_seed42_*.yaml`

### Example 2: Different Seeds for Multiple Experiments

```bash
python -m pkd.example train --exclusion-mcs 30% --exclusion-seed 1
python -m pkd.example train --exclusion-mcs 30% --exclusion-seed 2
python -m pkd.example train --exclusion-mcs 30% --exclusion-seed 3
```

Each produces different held-out configurations.

### Example 3: Combine Random + Manual

```bash
python -m pkd.example train \
  --exclusion-mcs 30% \
  --exclusion-config pkd/example/training_exclusions.yaml
```

Merges both exclusion sets.

### Example 4: Test on Held-Out Config

```bash
# Train with exclusion
python -m pkd.example train --exclusion-mcs 30%
# Output shows: Excluding MCS [2, 5, 7] for Model-B 3x2:1

# Test on that held-out MCS
python -m pkd.example test --slice channel_model_id:2 N_t:3 N_r:2 BW:20.0 N_ss:1 MCS:7
```

---

## Benefits of Refactoring

- **Organized**: 100-500 lines per module (vs 1473 lines monolithic)
- **Maintainable**: Changes isolated to relevant modules
- **Reusable**: Import modules independently
- **Clear**: Explicit dependencies via imports
