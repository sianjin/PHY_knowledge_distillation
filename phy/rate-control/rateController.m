function [mcsNext, state] = rateController(perInst, mcsCur, state)
%rateController Shared EWMA-PER dual-threshold rate-control policy (Fig. 15).
%
%   [mcsNext, state] = rateController(perInst, mcsCur, state) implements the
%   rate-adaptation policy used identically by the MATLAB teacher and the
%   Python PKD student. The ONLY difference between the two closed-loop
%   systems is the source of the effective SINR that produces perInst; the
%   controller logic below is byte-for-byte equivalent to
%   pkd/rate_control.py (RateController).
%
%   Inputs
%     perInst : instantaneous estimated PER for the current packet, i.e.
%               AWGN-LUT PER at (effective SINR, current MCS).
%     mcsCur  : MCS index used for the current packet (0..MCS_MAX).
%     state   : controller state struct. Pass [] on the first call to
%               initialise. Fields:
%                 .perEwma   - EWMA of perInst
%                 .goodCount - consecutive packets with perEwma < PER_LOW
%
%   Output
%     mcsNext : MCS index to use for the next packet (0..MCS_MAX).
%     state   : updated controller state.
%
%   Policy
%     perEwma_t = (1-ALPHA)*perEwma_{t-1} + ALPHA*perInst_t
%     if perEwma_t > PER_HIGH        -> MCS down by 1, reset goodCount
%     elseif perEwma_t < PER_LOW     -> goodCount++
%                                       if goodCount >= UP_COUNT ->
%                                           MCS up by 1, reset goodCount
%     else                            -> hold, reset goodCount
%
%   Constants (keep in sync with pkd/rate_control.py):
%   Fast-EWMA policy with a dead-band: raise MCS only when EWMA PER is
%   clearly low (< PER_LOW), lower when high (> PER_HIGH), hold in between.
%   UP_COUNT = 1 (step up on a single good EWMA sample) is needed for the
%   controller to track even the slow (f = 1) Fig. 15 SNR sweep. The
%   PER_LOW/PER_HIGH gap (0.03 vs 0.10) damps the up/down oscillation that
%   UP_COUNT = 1 would otherwise cause -- with PER_LOW = 0.05 the mid-SNR
%   operating point sat above PER_HIGH and the controller hunted. Down-
%   steps stay immediate, which keeps the loop stable.
ALPHA    = 0.2;
PER_LOW  = 0.03;
PER_HIGH = 0.10;
UP_COUNT = 1;
MCS_MIN  = 0;
MCS_MAX  = 9;

if isempty(state)
    state = struct('perEwma', perInst, 'goodCount', 0);
else
    state.perEwma = (1 - ALPHA) * state.perEwma + ALPHA * perInst;
end

mcsNext = mcsCur;

if state.perEwma > PER_HIGH
    mcsNext = max(mcsCur - 1, MCS_MIN);
    state.goodCount = 0;
elseif state.perEwma < PER_LOW
    state.goodCount = state.goodCount + 1;
    if state.goodCount >= UP_COUNT
        mcsNext = min(mcsCur + 1, MCS_MAX);
        state.goodCount = 0;
    end
else
    state.goodCount = 0;
end
end
