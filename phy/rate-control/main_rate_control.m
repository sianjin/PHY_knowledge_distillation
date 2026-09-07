% main_rate_control  Top-level driver for the Fig. 15 closed-loop
% rate-adaptation experiment (MATLAB PHY teacher).
%
% Produces:
%   snr_trajectory.mat     - common deterministic time-varying SNR input
%   beta_table.mat         - calibrated EESM beta for MCS 0-9 on the slice
%   teacher_rate_control.mat - N closed-loop realizations (per-packet MCS,
%                              effective SINR, PER, errors, throughput)
%
% The Python PKD student reads snr_trajectory.mat and teacher_rate_control.mat
% via pkd/example/evaluate_rate_control.py to build Fig. 15.
%
% Fixed static slice: Model-B, 3x2:2, CBW40 (same slice as the Fig. 14
% PER-SNR waterfall). Only MCS is time-varying, so C_t ~= C.

clear; clc;

cbw     = "CBW40";
chan    = "Model-B";
numTxRx = [3 2];
numSs   = 2;

% ---- Smoke test first: uncomment for a quick sanity run ----
% corrPHYRateControl(cbw, chan, numTxRx, numSs, 4, 200);

% ---- Full run ----
N_real = 100;
T      = 1000;
corrPHYRateControl(cbw, chan, numTxRx, numSs, N_real, T);
