function snr = snrTrajectory(T, snrMin, snrMax)
%snrTrajectory Deterministic common time-varying SNR trajectory for Fig. 15.
%
%   snr = snrTrajectory(T, snrMin, snrMax) returns a T-by-1 vector of
%   average-SNR values
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
if nargin < 2 || isempty(snrMin), snrMin = 12; end
if nargin < 3 || isempty(snrMax), snrMax = 58; end
validateattributes(snrMin, {'numeric'}, {'real', 'finite', 'scalar'});
validateattributes(snrMax, {'numeric'}, {'real', 'finite', 'scalar', '>', snrMin});

% --- Deterministic parameters (must match pkd/rate_control side) ---
mid    = (snrMin + snrMax) / 2;
amp    = (snrMax - snrMin) / 2;
f      = 2;     % complete cycles keep low- and high-SNR exposure balanced

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
