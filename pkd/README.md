# PHY Knowledge Distillation (PKD) - PyTorch Implementation

Paper-faithful implementation of "PHY Knowledge Distillation for Scalable Wireless Network Simulations" by Sian Jin.

## Overview

This implementation provides a complete, production-ready framework for replacing expensive PHY simulators with a learned surrogate model that preserves the conditional distribution of effective SINR.

### Key Features

- **Log-domain AR(p) process** with guaranteed stability via PACF parameterization
- **Non-Gaussian innovations** using conditional monotone spline flows
- **Compositional conditioning** for generalization across configurations
- **Time-skipping via configuration caching** for efficient event-driven simulation
- **Teacher-forced training** with sequence negative log-likelihood
- **Seamless integration** with AWGN PER lookup tables

## Installation

```bash
pip install torch numpy scipy tqdm
```

## Project Structure

```
pkd/
├── config.py              # Configuration structures
├── model/
│   ├── __init__.py
│   ├── encoder.py         # Compositional conditioning encoder
│   ├── heads.py           # Parameter prediction heads
│   ├── ld.py              # Levinson-Durbin (PACF -> AR)
│   ├── flow.py            # 1D monotone spline flow
│   └── pkd_model.py       # Complete PKD model
├── train.py               # Training pipeline
├── infer.py               # Inference engine with time-skipping
├── per_lut.py             # AWGN PER lookup table
├── example.py             # Usage examples
└── tests/
    └── test_components.py # Unit tests
```

## Quick Start

### 1. Training

```python
from pkd.model.pkd_model import PKDModel
from pkd.train import train_pkd

# Create model
model = PKDModel(
    num_channel_models=5,
    num_mcs=10,
    num_nss=4,
    ar_order=10,
    hidden_dim=128
)

# Train with teacher-generated sequences
trained_model = train_pkd(
    model,
    train_sequences,  # List of gamma_eff arrays from PHY simulator
    train_configs,    # List of config dicts
    val_sequences,
    val_configs,
    num_epochs=50,
    batch_size=256,
    device='cuda'
)
```

### 2. Inference

```python
from pkd.infer import PKDInference
from pkd.per_lut import AWGNPERLookup

# Create PER lookup table
per_lut = AWGNPERLookup.load_ldpc_lut()  # Loads embedded LDPC PER table (MCS 0-11)

# Create inference engine
inference = PKDInference(
    model,
    per_lut,
    ar_order=10,
    device='cuda'
)

# Run simulation
config = {
    'channel_model_id': 0,
    'N_t': 4,
    'N_r': 4,
    'BW': 20.0,
    'SNR_bar': 15.0,
    'MCS': 5,
    'N_ss': 2
}

results = inference.run_sequence([config] * 1000)
# Returns: gamma_eff, per, errors arrays
```

### 3. Time-Skipping for Efficiency

```python
# Define windows with constant configuration
config_windows = [
    (config1, 500),  # 500 packets with config1
    (config2, 300),  # 300 packets with config2
    (config1, 200),  # 200 packets back to config1
]

# Network evaluated only 2 times (once per unique config)
results = inference.run_with_time_skipping(config_windows)
```

## Configuration Format

Each configuration dict should contain:

```python
config = {
    # Static (unchanged during run)
    'channel_model_id': int,  # Channel model ID
    'N_t': int,               # Transmit antennas
    'N_r': int,               # Receive antennas
    'BW': float,              # Bandwidth (MHz)

    # Dynamic (may vary with rate adaptation)
    'SNR_bar': float,         # Average SNR (dB)
    'MCS': int,               # Modulation and coding scheme
    'N_ss': int,              # Number of spatial streams
}
```

## Running Tests

```bash
cd pkd/tests
python test_components.py
```

Tests verify:
- PACF → AR stability
- Flow invertibility
- AR mean consistency
- Time-skipping correctness

## Data Flow & Scale Management

### Critical Distinction: X_t vs gamma_eff

The PKD model makes an important distinction between:

- **`X_t`**: The internal AR(p) process variable in natural log scale
- **`gamma_eff`**: The effective SINR that undergoes scale conversions for I/O

#### Scale Conversion Pipeline

1. **Input (.mat files)**: `gamma_eff` in **dB scale** (10*log10 of linear SINR)
   ```
   gamma_dB = 10 * log10(gamma_linear)
   ```

2. **Data Loading** (`data_loader.py`): Converts to natural log scale
   ```python
   X_t = gamma_dB * np.log(10) / 10  # Convert dB to natural log
   ```

3. **AR Process Model**: `X_t` follows AR(p) with Gaussian innovation
   ```
   X_t = μ_t + ε_t  where ε_t ~ N(0, σ²)
   μ_t = c + Σ(φ_i * X_{t-i})  # AR mean
   ```

4. **Training/Inference**: All internal computations use `X_t` in natural log scale

5. **Output** (`infer.py`): Converts back to dB scale for consistency
   ```python
   gamma_dB = X_t * 10 / np.log(10)  # Convert natural log to dB
   ```

6. **PER Lookup**: Uses dB scale directly (table expects dB inputs)

#### Why Natural Log for X_t?

- **Additivity**: AR process becomes additive rather than multiplicative
- **Stability**: Gaussian innovations in log space → log-normal in linear space
- **Positivity**: Ensures gamma_eff > 0 without explicit constraints
- **Correlation**: Better captures temporal correlation structure

#### Mathematical Relationship

```
gamma_linear = exp(X_t)           # Natural log → Linear
gamma_dB = 10*log10(gamma_linear) # Linear → dB
gamma_dB = X_t * 10/ln(10)        # Direct conversion: Natural log → dB
```

## Key Implementation Details

### 1. Stability Guarantee

AR coefficients are guaranteed stable by:
- Parameterizing via partial autocorrelation (PACF)
- Mapping: `kappa_i = (1 - delta) * tanh(u_i)` ensures `|kappa_i| < 1`
- Levinson-Durbin recursion converts PACF → stable AR coefficients

### 2. Non-Gaussian Innovations

Conditional monotone spline flow:
- Rational quadratic splines with configurable bins
- Epsilon-mixed softmax ensures minimum bin size (numerical stability)
- Exact likelihood via change of variables

### 3. Time-Skipping Mechanism

Configuration-driven parameter caching:
- When config unchanged: reuse cached `(m, phi, c, psi)`
- Network evaluation: O(1) per configuration change
- AR evolution: O(p) per packet (cheap)
- Typical speedup: 100-1000x vs per-packet network calls

### 4. Compositional Conditioning

Separates static and dynamic factors:
- Static: `h_static = MLP([channel_emb, N_t, N_r, BW])`
- SNR: `h_snr = MLP(SNR_bar)`
- Base: `h_base = h_static + h_snr`
- FiLM: `h_t = alpha(MCS, N_ss) * h_base + beta(MCS, N_ss)`

This structure enables smooth generalization across rate adaptation.

## Model Hyperparameters

Default values (tune based on your data):

```python
ar_order = 10              # AR process order
hidden_dim = 128           # Conditioning representation dimension
num_flow_bins = 16         # Spline bins for innovation flow
flow_tail_bound = 5.0      # Tail bound for spline domain
min_bin_size = 1e-3        # Minimum bin size (stability)
pacf_delta = 1e-4          # PACF margin from ±1
burn_in = 50               # Cold start burn-in packets
snr_quant = 0.1            # SNR quantization for caching (dB)
```

## Performance

Compared to full PHY simulation:
- **Runtime**: 100-1000x faster (with time-skipping)
- **Memory**: O(p) state + cached parameters (KB vs GB)
- **Fidelity**: Preserves temporal correlation and burst errors
- **Scalability**: Cost independent of MIMO dimension and bandwidth

## Citation

```bibtex
@article{jin2025phy,
  title={PHY Knowledge Distillation for Scalable Wireless Network Simulations},
  author={Jin, Sian},
  year={2025}
}
```

## License

See paper for usage terms.
