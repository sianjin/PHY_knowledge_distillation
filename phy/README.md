# PHY Knowledge Distillation (PKD)

**"Addressing Configuration Scalability for Wireless System Simulations using Knowledge Distillation"**
by Sian Jin

---

## Overview

High-fidelity physical-layer (PHY) simulation is essential for accurate system-level evaluation of modern wireless systems, but becomes computationally prohibitive for long simulation horizons and adaptive configurations. Existing PHY abstraction techniques reduce complexity, yet either neglect temporal correlation or rely on configuration-specific calibration and parameter tables that do not scale to realistic adaptive systems.

PHY Knowledge Distillation (PKD) replaces per-configuration calibration with learning-based parameter inference. A neural student model learns to predict the parameters of an interpretable stochastic effective-SINR process directly from the PHY configuration, treating the high-fidelity simulator as a stochastic teacher. The learned model generates time-correlated effective SINR samples without channel realization, waveform-level processing, or parameter lookup tables, while naturally supporting configuration adaptation and efficient time skipping.

### PHY Abstraction Paradigm Comparison

| Approach | Accuracy | Runtime Efficiency | Config Scalability |
|---|---|---|---|
| Full PHY Simulation | High | Low | High (but slow) |
| Traditional PHY Abstraction (EESM) | High | Medium | Requires per-config calibration |
| Stochastic PHY Abstraction (EESM-log-AR) | High | High | Limited — per-config parameter tables |
| **PKD (This Work)** | **High** | **High** | **High — inference from configuration** |

---

## Repository Structure

```
.
├── phy/                           # MATLAB PHY workflows
│   ├── eesm-accuracy-validation/  # Calibrate EESM beta and visualize accuracy
│   │   ├── box0Simulation.m       # Main simulation script
│   │   ├── calculateSINR.m        # EESM effective SINR computation
│   │   └── ...                    # Channel modeling, beamforming, spatial correlation
│   ├── teacher-data-generation/   # Generate effective-SINR training datasets
│   └── runtime-benchmark/         # Measure abstraction sequence-generation runtime
│
├── pkd/                           # Python: PKD student model
│   ├── config.py                  # Configuration structures
│   ├── train.py                   # Training pipeline
│   ├── infer.py                   # Inference with time-skipping
│   ├── per_lut.py                 # AWGN PER lookup tables
│   ├── example.py                 # Usage examples
│   ├── model/
│   │   ├── encoder.py             # Compositional FiLM conditioning encoder
│   │   ├── heads.py               # Parameter prediction heads (mean, PACF, innovation)
│   │   ├── ld.py                  # Levinson-Durbin recursion (PACF -> AR coefficients)
│   │   ├── flow.py                # 1D monotone spline normalizing flow
│   │   └── pkd_model.py           # Complete PKD model
│   └── tests/
│       └── test_components.py     # Unit tests
│
└── data/                          # .mat files from MATLAB simulation (teacher data)
```

---

## Part 1: MATLAB PHY Simulation (Teacher)

### PHY Folder Responsibilities

- `eesm-accuracy-validation/` calibrates the EESM beta parameter against
  full-PHY packet outcomes, reruns the abstraction, and plots PER comparisons.
  Its primary purpose is visual accuracy validation.
- `teacher-data-generation/` recalibrates beta for each selected PHY
  configuration, then generates and saves effective-SINR sequences for PKD
  training.
- `runtime-benchmark/` recalibrates beta before timing, then measures the
  lightweight abstraction path. Beta calibration is outside the timed section.

The folders are independent workflows: teacher-data generation and runtime
benchmarking call their own copies of `corrPHYVal.m`; they do not load a beta
artifact produced by `eesm-accuracy-validation/`.

### Requirements

- **MATLAB R2026a or later**
- **WLAN Toolbox** (required for 802.11ax / Wi-Fi 6 simulation)
- **Communications Toolbox**

### Simulation Setup

The simulator generates effective SINR sequences for Wi-Fi 6 (802.11ax) under the following setup (see Table II of the paper):

| Parameter | Value |
|---|---|
| Communication system | Wi-Fi 6 (802.11ax) |
| Channel type | TGax channels B and D |
| Doppler spectrum | Jakes' model |
| Maximum speed | 0.089 km/h |
| Coherence time | T_c = 0.978 s |
| Sample period | T_s = T_c / 4 |
| Channel coding | LDPC |
| Payload length | 1000 bytes |
| MIMO decoding | MMSE |
| Sequences per configuration | 50 |
| Packets per sequence | 1000 |

### Configuration Space

Each simulation run is parameterized by:

| Variable | Type | Values |
|---|---|---|
| Channel model (CH) | Static | TGax B, D |
| Transmit antennas (N_t) | Static | 1–4 |
| Receive antennas (N_r) | Static | 1–2 |
| Bandwidth (BW) | Static | 20 MHz, 40 MHz |
| Average SNR | Dynamic | 10 points along PER-SNR waterfall |
| MCS | Dynamic | 0–9 |
| Spatial streams (N_ss) | Dynamic | 1–2 |
| Resource allocation (R_t) | Dynamic | Full-band (R_t = 0) |

MIMO configurations evaluated: 1×1:1, 2×1:1, 2×2:1, 2×2:2, 3×1:1, 3×2:1, 3×2:2, 4×1:1, 4×2:1, 4×2:2.

The total configuration space is **4000 distinct configurations** (2 × 10 × 10 × 2 × 10 MIMO configs × 1 resource type).

### Running the Simulation

```matlab
% Navigate to the teacher-data generation folder
cd phy/teacher-data-generation

% Run the main simulation script
box0Simulation

% Output: .mat files in data/ with effective SINR sequences
```

### Output Format

Each `.mat` file contains:

| Variable | Shape | Description |
|---|---|---|
| `gamma_eff` | (1000 × 50) | Effective SINR sequences in **log scale** (natural log of linear SINR: X_t = ln(gamma_eff_linear)) |
| `config` | (7 × 50) | Configuration matrix: [CH; N_t; N_r; BW; SNR_bar; MCS; N_ss] per sequence |

> **Scale note:** `gamma_eff` is stored as X_t = ln(gamma_eff_linear). The PKD model operates internally in this natural log domain. For display or PER lookup, convert to dB via: gamma_dB = X_t * 10 / ln(10).

### Validation

```matlab
% Navigate to the EESM accuracy-validation folder
cd phy/eesm-accuracy-validation

% Run validation scripts to compare PHY abstraction methods
% (EESM, MIESM, RBIR, etc. vs full PHY simulation)
```

---

## Part 2: Python PKD Student Model

### Requirements

```
pip install -r requirements.txt
```

Dependencies: PyTorch >= 2.0.0, NumPy >= 1.24.0, SciPy >= 1.10.0, tqdm >= 4.65.0

### Model Architecture

The PKD student model consists of two components:

1. **Log-domain AR(p) stochastic process:**
   ```
   X_t = c_t + sum_{i=1}^{p} phi_{i,t} * X_{t-i} + epsilon_t
   ```
   where X_t = ln(gamma_eff_t) is the log-domain effective SINR.

2. **Neural parameter generator** that maps configuration C_t -> (m_t, kappa_{1:p,t}, sigma_t):
   - **Compositional conditioning encoder:** separates static link properties (CH, N_t, N_r, BW) from dynamic packet-level parameters (SNR, MCS, N_ss, R_t) via FiLM modulation.
   - **Mean head:** predicts conditional mean m_t = E[X_t | C_t].
   - **Autoregressive dynamics head:** predicts PACF coefficients kappa_{1:p,t} in (-1, 1), then applies Levinson-Durbin recursion to obtain stable AR coefficients phi_{1:p,t}.
   - **Innovation head:** predicts scale sigma_t > 0 for Gaussian innovation epsilon_t ~ N(0, sigma_t^2).

The Levinson-Durbin mapping from PACF to AR coefficients **guarantees stability** of the AR(p) process for any network output.

### Quick Start

```bash
# Run tests
cd pkd/tests && python test_components.py

# Run example (synthetic data)
python -m pkd.example

# Train on real MATLAB data
python -m pkd.example train-real
```

### Training

Place `.mat` files from `phy/teacher-data-generation/` into `data/`, then:

```python
from pkd import train_pkd

model = train_pkd(
    model, train_sequences, train_configs,
    val_sequences, val_configs,
    num_epochs=50, batch_size=256, device='cuda'
)
```

Training minimizes the conditional negative log-likelihood (NLL):

```
L(theta) = -sum_n sum_{t=p+1}^{T} ln q_theta(X_t^(n) | X_{t-p:t-1}^(n), C_t^(n))
```

Early stopping with patience=3 based on validation NLL.

### Inference

```python
from pkd import PKDModel, PKDInference, AWGNPERLookup

model = PKDModel(num_channel_models=5, num_mcs=10, num_nss=4)
per_lut = AWGNPERLookup.load_ldpc_lut()
inference = PKDInference(model, per_lut, device='cuda')

config = {
    'channel_model_id': 0, 'N_t': 4, 'N_r': 2, 'BW': 40.0,
    'SNR_bar': 15.0, 'MCS': 7, 'N_ss': 2
}

results = inference.run_sequence([config] * 1000)
print(f"Mean SINR: {results['gamma_eff'].mean():.2f} dB")
print(f"Packet errors: {results['errors'].sum()}/{len(results['errors'])}")
```

### Time-Skipping (100-1000x Speedup)

```python
# Network evaluated only once per unique configuration window
config_windows = [
    (config_mcs5, 500),   # 500 packets with MCS 5
    (config_mcs7, 300),   # 300 packets with MCS 7 (rate adaptation)
]
results = inference.run_with_time_skipping(config_windows)
```

---

## Runtime Performance

From paper Table III — generating 50 sequences of 1000 packets (TGax channel B, MCS 7):

| MIMO Config | BW | Traditional PHY Abstraction | PKD |
|---|---|---|---|
| 1×1:1 | 20 MHz | 15 min | **1.2 sec** |
| 1×1:1 | 40 MHz | 21 min | **1.2 sec** |
| 3×2:1 | 20 MHz | 21 min | **1.2 sec** |
| 3×2:1 | 40 MHz | 25 min | **1.2 sec** |
| 4×2:2 | 20 MHz | 22 min | **1.2 sec** |
| 4×2:2 | 40 MHz | 31 min | **1.2 sec** |

PKD runtime is **independent of MIMO dimension and bandwidth**, since it replaces the full PHY simulation chain with scalar AR recursion operations.

### Generalization Under Sparse Configuration Coverage

PKD maintains high accuracy even when up to ~70% of MCS configurations are excluded from training:
- **Marginal distribution** (KS statistic): near-constant up to 70% MCS exclusion.
- **Temporal correlation** (ACF RMSE): robust across all exclusion levels, since temporal dynamics are governed by the Jakes Doppler spectrum rather than MCS.

---

## Data Flow and Scale Conversions

```
MATLAB simulation
    gamma_eff stored as: X_t = ln(gamma_eff_linear)   [natural log scale]
          |
          v
PKD training/inference
    Internal: X_t in natural log domain (AR process operates here)
    Output gamma_eff in dB: gamma_dB = X_t * 10 / ln(10)
          |
          v
PER lookup
    Uses gamma_dB for AWGN table lookup -> per -> Bernoulli error sampling
```

---

## Citation

```bibtex
@article{jin2025phy,
  title={Addressing Configuration Scalability for Wireless System Simulations using Knowledge Distillation},
  author={Jin, Sian},
  year={2025}
}
```

---

## References

1. J. Francis and N. B. Mehta, "EESM-based link adaptation in point-to-point and multi-cell OFDM systems," *IEEE Trans. Wireless Commun.*, vol. 13, no. 1, pp. 407–417, 2014.
2. R. Patidar et al., "Link-to-system mapping for ns-3 Wi-Fi OFDM error models," *WNS3 '17*, ACM, 2017.
3. S. Jin, S. Roy, and T. R. Henderson, "Efficient PHY layer abstraction for fast simulations in complex system environments," *IEEE Trans. Commun.*, vol. 69, no. 8, pp. 5649–5660, 2021.
4. S. Jin, S. Roy, and T. R. Henderson, "EESM-log-AR: an efficient error model for OFDM MIMO systems over time-varying channels," *WNS3 '21*, ACM, 2021.
