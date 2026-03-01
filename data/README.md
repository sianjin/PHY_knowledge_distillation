# Data Directory

This folder contains the `.mat` data files used to train and evaluate the PKD (Physical Knowledge Distillation) model.

## File Naming Convention

Files follow the pattern:

```
CBW{BW}_Model-{ChannelModel}_{Nt}-by-{Nr}-by-{Nss}_MCS{MCS}_SNR{SNR}.mat
```

For example:
- `CBW20_Model-B_2-by-2-by-1_MCS5_SNR15.mat`
  - Channel bandwidth: 20 MHz
  - Channel model: B (indoor office)
  - Antenna configuration: 2 Tx × 2 Rx, 1 spatial stream
  - MCS index: 5
  - Average SNR: 15 dB

## File Content

Each `.mat` file contains:
- `gamma_eff`: Effective SINR sequences of shape `(1000, 100)` — 100 sequences of 1000 packets each, stored in **log scale** (natural log of linear SINR)
- `config`: Configuration matrix of shape `(7, 100)` with parameters for each sequence:
  1. Channel model ID (0=Model-B, 1=Model-D, …)
  2. Number of transmit antennas (N_t)
  3. Number of receive antennas (N_r)
  4. Bandwidth in MHz (BW)
  5. Average SNR in dB (SNR_bar)
  6. MCS index (0–9)
  7. Number of spatial streams (N_ss)

## Generating New Data

The `.mat` files are produced by the MATLAB simulation scripts in `phy/simulation/`. See [`phy/README.md`](../phy/README.md) for instructions on running the simulations.

## Using the Data

Pass the path to this directory when running training or evaluation:

```bash
# Train with all files in this directory
python pkd/example.py train

# Evaluate on the test split
python pkd/example.py test

# Qualitative evaluation for a specific sequence
python pkd/example.py eval 0
```

All scripts resolve the data directory automatically as `<project_root>/data/`.
