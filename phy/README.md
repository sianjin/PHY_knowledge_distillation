# PHY Simulation and Validation

This folder contains MATLAB code for wireless PHY layer simulation and traditional PHY abstraction validation.

## Requirements

- **MATLAB R2026a or later**
- **WLAN Toolbox** (required for 802.11ax simulation)

## Folder Structure

### `simulation/`
Contains MATLAB scripts to simulate effective SINR sequences for different network configurations.

- Generates 100 sequences × 1000 packets per configuration
- Each sequence corresponds to a specific set of wireless parameters:
  - Channel model ID
  - Number of transmit/receive antennas (N_t, N_r)
  - Bandwidth (BW)
  - Average SNR (SNR_bar)
  - Modulation and Coding Scheme (MCS)
  - Number of spatial streams (N_ss)

**Key Files:**
- `box0Simulation.m`: Main simulation script for generating effective SINR data
- `calculateSINR.m`: Computes effective SINR per packet
- Additional helper functions for channel modeling, beamforming, and spatial correlation

### `validation/`
Contains MATLAB scripts to validate traditional PHY abstraction methods.

- Compares different PHY abstraction techniques (ESM, MIESM, RBIR, etc.)
- Evaluates accuracy against full PHY simulation
- Generates validation metrics for abstraction quality

## Usage

### Running Simulations

```matlab
% Navigate to simulation folder
cd phy/simulation

% Run the main simulation script
box0Simulation

% Output: .mat files with effective SINR sequences
```

### Validation

```matlab
% Navigate to validation folder
cd phy/validation

% Run validation scripts to compare PHY abstraction methods
% (specific script names depend on your validation setup)
```

## Output Format

The simulation generates `.mat` files containing:
- `gamma_eff`: Effective SINR sequences (1000 × 100) in **log scale**
- `config`: Configuration matrix (7 × 100) with parameters for each sequence

## Integration with PKD

The generated `.mat` files from `simulation/` are used as teacher data for training the PKD (Physical Knowledge Distillation) model:

1. Place `.mat` files in the `data/` folder at project root
2. Run PKD training: `python pkd/example.py train-real`
3. The PKD model learns to reproduce the effective SINR distributions

## Notes

- All effective SINR values are stored in **log scale** (log of linear SINR)
- The PKD model converts between log and linear scales as needed
- Packet length is 1000 bytes by default in the simulation
- PER lookup tables are adjusted from L0=1458 bytes reference using TGax methodology
