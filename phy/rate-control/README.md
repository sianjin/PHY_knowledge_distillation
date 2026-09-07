# rate-control/ — Closed-loop rate adaptation under a time-varying configuration (Fig. 15)

This workflow addresses Reviewer 1's comment that the framework is motivated by
time-varying configurations, yet all other experiments fix `C_t = C`. Here the
average SNR follows a common deterministic trajectory and the MCS is selected
packet-by-packet by a rate controller, so `C_t != C`.

It is a copy of `runtime-benchmark/` with a rate-control driver added. The
shared PHY helpers (`box0Validation.m`, `calculateSINR.m`, `getBox0*Params.m`,
`spatialCorrelation.m`, `tgaxLinkPerformanceModel.m`, `mcs2beta.m`,
`betaOptimization.m`, `awgnPerSnrFittingMse.m`, ...) are unchanged.
`main.m`, `corrPHY.m`, `corrPHYSim.m` are the original runtime-benchmark
entry points and are **not** used here.

## What Fig. 15 tests

Whether replacing the PHY simulator with the PKD student preserves *closed-loop
rate-adaptation behaviour* under a time-varying configuration — **statistically**,
not by reproducing the same stochastic realization. Teacher and student share:

- the same time-varying SNR trajectory `{SNR_t}` (`snrTrajectory.m`), and
- the same rate-control policy (`rateController.m`).

The only difference is the source of effective SINR:

| System  | Effective-SINR source                         |
|---------|-----------------------------------------------|
| Teacher | MATLAB PHY simulator (this folder)            |
| Student | PKD log-AR process (`pkd/`, Python)           |

Fidelity hierarchy: effective-SINR → PER → **closed-loop rate-adaptation**.

## Fixed static slice

Model-B, `3x2:2`, CBW40 (same slice as the Fig. 14 PER–SNR waterfall).
MCS 0–9 adapted dynamically. Payload 1000 bytes, LDPC.

## Files

| File | Role |
|---|---|
| `snrTrajectory.m` | Deterministic common `{SNR_t}` (sinusoid + fixed-seed AR(1) jitter), swept through low/med/high SNR. Saved to `snr_trajectory.mat`. |
| `rateController.m` | Shared EWMA-PER dual-threshold controller. **Byte-for-byte twin of `pkd/rate_control.py`.** |
| `betaTable.m` | Calibrates EESM `beta` for MCS 0–9 on the slice via `corrPHYVal`. Cached to `beta_table.mat`. Outside the closed-loop path. |
| `box0RateControl.m` | One closed-loop realization: one TGax channel realization, per-packet effective SINR from EESM at the time-varying `N0_t`, coin flip vs. AWGN-LUT PER, `rateController` picks `MCS_{t+1}`. |
| `corrPHYRateControl.m` | `N_real` independent realizations (`parfor`, independent channel seeds, common trajectory). Saves `teacher_rate_control.mat`. |
| `main_rate_control.m` | Top-level driver. |

## How to run

```matlab
cd phy/rate-control

% 1. Smoke test (fast): edit main_rate_control.m to call
%    corrPHYRateControl(cbw, chan, numTxRx, numSs, 4, 200);
%    or just:
corrPHYRateControl("CBW40", "Model-B", [3 2], 2, 4, 200);

% 2. Full run
main_rate_control            % N_real = 100, T = 1000
```

First run calibrates `beta` for all 10 MCS (slow, cached afterwards in
`beta_table.mat`). Delete `beta_table.mat` to force recalibration.

## Outputs (consumed by `pkd/example/evaluate_rate_control.py`)

- **`snr_trajectory.mat`** — `snrTraj` (T×1), plus slice metadata.
- **`beta_table.mat`** — `betaVec` (1×10), `mcsList`.
- **`teacher_rate_control.mat`** (v7.3 / HDF5):
  - `mcsAll` (N×T, uint8) — MCS used per packet
  - `effSINRAll` (N×T) — EESM effective SINR (dB)
  - `perInstAll` (N×T) — AWGN-LUT PER at (effSINR, MCS)
  - `errorAll` (N×T, uint8) — sampled packet-error flags
  - `throughputMbps` (N×1) — achieved throughput per run
    (`successful_packets * payloadBits / (T * txPeriod)`)
  - `snrTraj`, `txPeriod`, `payloadBits`, `seed`, `meta`

## Validation checklist (before the Python side)

- [ ] `snrTrajectory(1000)` sweeps roughly 2–36 dB, ~1.5 periods, smooth.
- [ ] Smoke run completes; `mean MCS` tracks the SNR trajectory
      (high MCS at SNR peaks, MCS 0–1 at the trough).
- [ ] `effSINRAll` correlates with `snrTraj` (broadcast across runs).
- [ ] Mean sampled PER is near the controller's operating point
      (between `PER_LOW=0.02` and `PER_HIGH=0.1`, typically ~0.03–0.08).
- [ ] `throughputMbps` is a plausible spread (tens of Mbps for this slice).
