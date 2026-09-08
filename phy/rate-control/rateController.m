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
%                 .badCount  - consecutive packets with perEwma > PER_HIGH
%
%   Output
%     mcsNext : MCS index to use for the next packet (0..MCS_MAX).
%     state   : updated controller state.
%
%   Policy
%     perEwma_t = (1-ALPHA)*perEwma_{t-1} + ALPHA*perInst_t
%     if perEwma_t > PER_HIGH    -> badCount++; goodCount = 0
%                                   if badCount  >= DOWN_COUNT ->
%                                       MCS down by 1, reset badCount
%     elseif perEwma_t < PER_LOW -> goodCount++; badCount = 0
%                                   if goodCount >= UP_COUNT ->
%                                       MCS up by 1, reset goodCount
%     else                       -> hold, reset both counters
%
%   Constants (keep in sync with pkd/rate_control.py):
%   Fast-EWMA policy with a dead-band and symmetric confirmation counts:
%   raise MCS when EWMA PER is clearly low (< PER_LOW), lower when clearly
%   high (> PER_HIGH), hold in between. UP_COUNT = 1 keeps the ascent quick
%   enough to track the slow (f = 1) Fig. 15 SNR sweep. DOWN_COUNT = 2 (a
%   single bad EWMA sample no longer forces a step down) stops one unlucky
%   Bernoulli packet error from knocking the controller down a level --
%   the cause of the wide across-run MCS spread and the stubborn low-MCS
%   tail at high SNR in earlier runs. ALPHA = 0.1 (was 0.2) further limits
%   how much a single error moves perEwma.
ALPHA      = 0.1;
PER_LOW    = 0.03;
PER_HIGH   = 0.10;
UP_COUNT   = 1;
DOWN_COUNT = 2;
MCS_MIN    = 0;
MCS_MAX    = 9;

if isempty(state)
    state = struct('perEwma', perInst, 'goodCount', 0, 'badCount', 0);
else
    state.perEwma = (1 - ALPHA) * state.perEwma + ALPHA * perInst;
end

mcsNext = mcsCur;

if state.perEwma > PER_HIGH
    state.goodCount = 0;
    state.badCount = state.badCount + 1;
    if state.badCount >= DOWN_COUNT
        mcsNext = max(mcsCur - 1, MCS_MIN);
        state.badCount = 0;
    end
elseif state.perEwma < PER_LOW
    state.badCount = 0;
    state.goodCount = state.goodCount + 1;
    if state.goodCount >= UP_COUNT
        mcsNext = min(mcsCur + 1, MCS_MAX);
        state.goodCount = 0;
    end
else
    state.goodCount = 0;
    state.badCount = 0;
end
end
