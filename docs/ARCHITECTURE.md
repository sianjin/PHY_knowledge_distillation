# Architecture Documentation

This document describes key architectural decisions, design patterns, and the rationale behind major components of the PKD system.

---

## System Overview

**Physical-Layer Knowledge Distillation (PKD)** is a framework for learning fast neural approximations of expensive wireless PHY simulators.

```
┌─────────────────────────────────────────────────────┐
│                    PKD System                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐      ┌──────────────┐           │
│  │ PHY Simulator│ ───> │ MAT Files    │           │
│  │  (Teacher)   │      │ (Training    │           │
│  └──────────────┘      │  Data)       │           │
│                        └──────┬───────┘           │
│                               │                    │
│                               ▼                    │
│                        ┌──────────────┐           │
│                        │  PKD Model   │           │
│                        │  (Student)   │           │
│                        └──────┬───────┘           │
│                               │                    │
│                               ▼                    │
│                        ┌──────────────┐           │
│                        │  Inference   │           │
│                        │  Engine      │           │
│                        └──────────────┘           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. PKD Model (`pkd/model/pkd_model.py`)

**Purpose:** Conditional autoregressive model for log(gamma_eff) sequences

**Architecture:**
```
Input: config (7 params)
  │
  ▼
┌─────────────────────────┐
│ Compositional Encoder   │
│  - Static features      │
│  - SNR encoding         │
│  - FiLM modulation      │
└──────────┬──────────────┘
           │
           ▼
      context h_t
           │
           ├──────────────────────────────┐
           ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│ AR Parameter Gen │          │ Innovation Parameter│
│  - Mean (m)      │          │  - Gaussian: σ(h_t) │
│  - PACF (κ)      │          │  - Flow: ψ(h_t)     │
│  - Offset (c)    │          │                      │
└─────────┬────────┘          └──────────┬───────────┘
          │                              │
          └──────────┬───────────────────┘
                     ▼
              AR(p) Process
      log(γ_t) = c + Σφ_i·log(γ_{t-i}) + ε_t
                     │
                     ▼
              exp(log(γ_t))
                     │
                     ▼
                 gamma_eff
```

**Key Design Decisions:**

#### Why Log-Domain AR?
- **Multiplicative dynamics:** Wireless channels have multiplicative fading
- **Positivity:** exp(·) ensures γ > 0 without explicit constraints
- **Stability:** Log-domain is more numerically stable for large dynamic ranges
- **Gaussian innovations:** More appropriate in log-domain (log-normal distribution)

**Decision Record:**
- **Date:** Initial design
- **Alternative Considered:** Linear-domain AR
- **Rationale:** Log-domain better matches wireless channel physics

---

#### Why PACF Parameterization?
- **Stability Guarantee:** κ_i ∈ (-1, 1) ⟹ stable AR process (Levinson-Durbin)
- **Interpretability:** PACF has direct physical meaning in time series
- **Flexibility:** Can learn complex correlation structures

**Decision Record:**
- **Date:** 2025-02-01 (stability fixes)
- **Alternative Considered:** Direct AR coefficient prediction
- **Rationale:** PACF guarantees stability, direct φ does not
- **Implementation:** `kappa_max * tanh(κ_raw)` constrains to safe range

**Reference:** See `AR_STABILITY_FIXES.md`

---

#### Why Gaussian Innovation (Not Flow)?
- **Simplicity:** One parameter (σ) vs. many (flow bins, bounds)
- **Speed:** ~3x faster than normalizing flows
- **Stability:** No flow inverse numerical issues
- **Empirical Performance:** Gaussian is sufficient for most scenarios

**Decision Record:**
- **Date:** 2025-02-01
- **Alternative Considered:** Normalizing flows (originally implemented)
- **Rationale:** Gaussian is simpler and works well empirically
- **When to Reconsider:** If PIT uniformity test fails (p < 0.05)

**Reference:** See `GAUSSIAN_INNOVATION.md`

---

### 2. Compositional Encoder (`pkd/model/encoder.py`)

**Purpose:** Embed configuration parameters into context vector h_t

**Architecture:**
```
config = {
  channel_model_id,  ─┐
  N_t, N_r, BW       ─┼─> Static Encoder ──┐
}                     │                     │
                      │                     ├─> h_base
SNR_bar ─────────────┴──> SNR Encoder ─────┘
                                             │
MCS ──────> Embedding ──┐                   │
                        ├──> FiLM ─────> h_t
N_ss ─────> Embedding ──┘      ▲
                                │
                            h_base
```

**Design Decisions:**

#### Why Compositional Design?
- **Modularity:** Static vs. dynamic features separated
- **Rate Adaptation:** MCS/N_ss change frequently, modeled via FiLM
- **Physical Meaning:** Reflects how wireless systems actually work

**FiLM (Feature-wise Linear Modulation):**
```python
γ, β = FiLM(MCS, N_ss)
h_t = γ ⊙ h_base + β
```
- **Why:** MCS/N_ss act as "control signals" that modulate base representation
- **Benefit:** Captures interaction between rate parameters and channel conditions

---

#### Why These Encodings?

| Feature | Encoding | Rationale |
|---------|----------|-----------|
| `channel_model_id` | One-hot | Categorical, no ordering |
| `N_t`, `N_r` | log(N + 1) | Logarithmic effect on capacity |
| `BW` | log(BW + 1) | Logarithmic effect on capacity |
| `SNR_bar` | Linear / 30.0 | Already in dB (log scale) |
| `MCS` | Embedding | Categorical ID (0-indexed) |
| `N_ss` | Embedding | Count, but treated categorically |

**Decision Record:**
- **Date:** Initial design
- **Alternatives:** Could use learned embeddings for all features
- **Rationale:** Leverage known physical relationships (Shannon capacity)

---

### 3. Inference Engine (`pkd/infer.py`)

**Purpose:** Generate sequences and compute PER using trained model

**Key Methods:**

1. **`run_sequence(config_traj)`**
   - Generates gamma_eff sequence autoregressively
   - Handles burn-in period for initialization
   - Returns sequence + packet errors

2. **`predict_per(config_dict, num_realizations)`**
   - Monte Carlo PER estimation
   - Generates multiple sequences, averages error rate

3. **PER Lookup Integration:**
   - Uses AWGN PER LUT: `PER = f(gamma_eff_mean, MCS, N_ss)`
   - Maps effective SINR → packet error probability

**Design Decisions:**

#### Why Autoregressive Generation?
```python
for t in range(T):
    log_gamma[t] = sample from p(log_γ_t | log_γ_{t-p:t-1}, config)
```
- **Captures Temporal Dependence:** Essential for wireless channels
- **Sequential Nature:** Matches how channels actually evolve

#### Why Burn-In Period?
- **Problem:** Initial AR states are random, not from stationary distribution
- **Solution:** Generate extra samples, discard first `burn_in` steps
- **Typical Value:** `burn_in = 100` for `ar_order = 10`

---

## Data Pipeline

### Input: MAT Files (`data/*.mat`)

**Structure:** See `data/README.md` for complete specification

**Loading Pattern:**
```python
with h5py.File(mat_file, 'r') as f:
    config = f['config'][:]              # (7,) or (7,1)
    gamma_eff = f['gamma_eff'][:]        # (100, 1000)
    packet_error = f['packet_error'][:]  # (100, 1000)
    # ... other fields
```

**Critical Convention:**
- **MCS is 0-indexed** (0-9) in data files
- **N_ss is 1-indexed** (1-4) in data files
- Encoder handles N_ss conversion to 0-indexed for embeddings

---

### Training Data Format (`pkd/train.py`)

**Input:**
- `sequences`: List of numpy arrays, each shape `(T,)` in linear scale
- `configs`: List of dicts with 7 config parameters

**Preprocessing:**
```python
# Convert to log-domain
log_gamma = np.log(sequences)

# Create AR windows
for t in range(ar_order, T):
    X_hist = log_gamma[t-ar_order:t][::-1]  # Reversed history
    X_t = log_gamma[t]                      # Current value
    # Compute loss via model.compute_log_likelihood(X_t, X_hist, config)
```

**Why Reversed History?**
- φ₁ corresponds to lag-1, φ₂ to lag-2, etc.
- Reversing `[t-p, ..., t-1]` → `[t-1, ..., t-p]` matches coefficient order

---

## Training Procedure

**Objective:** Maximum likelihood estimation

```
θ* = argmax Σ log p(log(γ_t) | log(γ_{t-p:t-1}), config; θ)
            t,seq
```

**Loss Function:**
```python
log_likelihood = model.compute_log_likelihood(X_t, X_hist, config_dict)
loss = -log_likelihood.mean()  # Negative log-likelihood
```

**Optimizer:** Adam with learning rate scheduling

**Early Stopping:** Based on validation loss (patience = 3-5 epochs)

**Decision Record:**
- **Date:** Initial implementation
- **Alternative:** Could add PER prediction loss
- **Current Status:** Pure likelihood training works well
- **Future:** May add multi-task loss if PER accuracy insufficient

---

## Evaluation Metrics

### 1. Marginal Distribution Fidelity

**Metrics:**
- **CCDF Comparison:** Visual check of tail behavior
- **QQ Plot:** Quantile-quantile alignment
- **KS Test:** Kolmogorov-Smirnov statistic (p > 0.05 is good)

**Why Important:** Ensures generated SINR values have correct distribution

---

### 2. Temporal Dependence

**Metrics:**
- **ACF Comparison:** Autocorrelation function matching
- **PSD Comparison:** Power spectral density matching
- **ACF RMSE:** Root mean squared error of ACF (lower is better)

**Why Important:** Captures correlation structure, critical for burst errors

---

### 3. Innovation Structure (Calibration)

**Metrics:**
- **Ljung-Box Test:** Checks if innovations are uncorrelated (p > 0.05 is good)
- **PIT Uniformity:** Probability Integral Transform should be U(0,1)
- **PIT KS Test:** Uniformity test (p > 0.05 is good)

**Why Important:**
- Proper calibration ensures uncertainty quantification is correct
- Uniform PIT means predictive distribution is well-specified

---

## Key Invariants and Assumptions

### Data Conventions
1. **MCS is 0-indexed** throughout (0-9)
2. **N_ss is 1-indexed** in data (1-4), 0-indexed in embeddings
3. **gamma_eff is in linear scale** in data, log-scale in model
4. **Sequences are stationary** (no non-stationary trends)

### Model Constraints
1. **PACF bounded:** |κ_i| ≤ kappa_max < 1 (stability)
2. **Positive SINR:** Enforced via exp(log(γ))
3. **Finite variance:** min_sigma > 0 prevents σ → 0

### Numerical Stability
1. **Log-domain computation** for gamma_eff
2. **Softplus for positive parameters:** σ = softplus(σ_raw) + min_sigma
3. **Gradient clipping** during training (max_norm = 1.0)

---

## Extension Points

### Adding New Features

**To add a new configuration parameter:**

1. Update `data/README.md` with new config field
2. Modify `encoder.py` to include new feature:
   ```python
   # In StaticEncoder or SNREncoder
   new_feature_encoded = self.encode_new_feature(config_dict['new_param'])
   ```
3. Update all example configs in `example.py`, `test_components.py`
4. Retrain model with expanded embedding dimension

**To change innovation distribution:**

1. Implement new `InnovationModel` in `pkd/model/innovation.py`
2. Add to `PKDModel.__init__` switch statement:
   ```python
   if innovation_type == 'your_new_type':
       self.innovation = YourInnovationModel(...)
   ```
3. Update `CHANGELOG.md` with rationale

---

### Adding New Metrics

**To add evaluation metric:**

1. Create function in `pkd/example.py`:
   ```python
   def evaluate_your_metric(teacher_seq, student_seq):
       # Compute metric
       return {'metric_name': value}
   ```
2. Call in `example_evaluation()` workflow
3. Document in this file under "Evaluation Metrics"

---

## Performance Considerations

### Speed
- **Gaussian innovation:** ~3x faster than flows
- **AR order:** Higher p = more parameters, slower inference
- **Batch size:** Larger batches = better GPU utilization

### Memory
- **Sequence length:** Longer sequences require more memory
- **Batch size:** Limited by GPU memory
- **Number of realizations:** For PER estimation, trade-off accuracy vs. speed

---

## References to Other Documentation

- [CHANGELOG.md](../CHANGELOG.md) - What changed and when
- [COMMON_ISSUES.md](./COMMON_ISSUES.md) - Known pitfalls and solutions
- [data/README.md](../data/README.md) - Data format specification
- [AR_STABILITY_FIXES.md](../AR_STABILITY_FIXES.md) - AR process design
- [GAUSSIAN_INNOVATION.md](../GAUSSIAN_INNOVATION.md) - Innovation choice
- [QUICK_START.md](../QUICK_START.md) - Getting started guide

---

## Decision Log Template

When making major architectural changes, document here:

```markdown
### Component/Feature Name

**Date:** YYYY-MM-DD
**Alternatives Considered:**
- Option A: Brief description
- Option B: Brief description

**Decision:** Chose Option X

**Rationale:**
- Reason 1
- Reason 2

**Trade-offs:**
- Pro: ...
- Con: ...

**When to Reconsider:**
- Condition 1
- Condition 2

**Reference:** Link to detailed write-up if exists
```

---

**Last Updated:** 2026-02-03
**Maintained By:** Project contributors
