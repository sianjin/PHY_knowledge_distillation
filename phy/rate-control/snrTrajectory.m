function snr = snrTrajectory(T)
%snrTrajectory Deterministic common time-varying SNR trajectory for Fig. 15.
%
%   snr = snrTrajectory(T) returns a T-by-1 vector of average-SNR values
%   (dB) that sweeps through low-, medium-, and high-SNR regions. The
%   trajectory is fully deterministic (fixed RNG seed for the jitter) so
%   that the MATLAB teacher and the Python PKD student are driven by
%   EXACTLY the same input. Other configuration dimensions stay fixed;
%   only MCS is adapted by rate control, so C_t ~= C.
%
%   Model:
%       snr_t = mid + amp * sin(2*pi*f*t/T) + jitter_t
%   with a small fixed-seed AR(1) jitter (std ~1 dB) and a final clamp to
%   [snrMin, snrMax].
%
%   Defaults: T = 1000.

if nargin < 1 || isempty(T)
    T = 1000;
end

% --- Deterministic parameters (must match pkd/rate_control side) ---
mid    = 19;    % dB, midpoint of the sinusoid
amp    = 17;    % dB, amplitude -> nominal range [2, 36] dB
f      = 1.5;   % number of full sinusoid periods over the run
snrMin = 1;     % dB, hard floor
snrMax = 38;    % dB, hard ceiling

% AR(1) jitter, fixed seed for reproducibility
jitterStd  = 1.0;    % dB (marginal std of the jitter)
jitterRho  = 0.9;    % AR(1) coefficient
jitterSeed = 20260907;

t = (1:T).';
base = mid + amp * sin(2*pi*f*t/T);

s = RandStream('twister', 'Seed', jitterSeed);
innovStd = jitterStd * sqrt(1 - jitterRho^2);
jitter = zeros(T, 1);
for k = 2:T
    jitter(k) = jitterRho * jitter(k-1) + innovStd * randn(s);
end

snr = base + jitter;
snr = min(max(snr, snrMin), snrMax);
end
