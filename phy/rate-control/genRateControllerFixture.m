function genRateControllerFixture(outFile)
%genRateControllerFixture Reference MCS traces from the MATLAB controller.
%
%   Runs rateController.m over a few crafted PER sequences and saves the
%   resulting per-packet MCS traces so the Python twin
%   (pkd/tests/test_rate_control_parity.py) can assert byte-for-byte
%   agreement. Run this once (and again whenever rateController.m changes).
%
%   Default outFile: phy/rate-control/rate_controller_fixture.mat

if nargin < 1 || isempty(outFile)
    outFile = fullfile(fileparts(mfilename('fullpath')), 'rate_controller_fixture.mat');
end

% Recover the controller constants by probing rateController with a known
% state (they are local to the function, so we document them here and the
% Python side cross-checks against its own module).
constants = struct('ALPHA', 0.1, 'PER_LOW', 0.03, 'PER_HIGH', 0.10, ...
    'UP_COUNT', 1, 'DOWN_COUNT', 2, 'MCS_MIN', 0, 'MCS_MAX', 9);

perSeqs = {
    % 1: climb to ceiling, hold in dead-band, alternating bad/good,
    %    sustained bad drop, small recovery, noisy tail
    [zeros(1,40), 0.06*ones(1,20), repmat([0.30 0.0],1,10), ...
     0.5*ones(1,40), zeros(1,10), local_noise(50, 0, 0.2, 12345)];
    % 2: pure climb from the floor
    zeros(1,60);
    % 3: pure drop from the ceiling
    0.9*ones(1,60);
    % 4: single bad spike around a settled operating point (must NOT drop)
    [0.06*ones(1,5), 0.5, 0.06*ones(1,5)];
    % 5: long noisy run near the operating point
    local_noise(300, 0, 0.15, 777);
    };
mcsInits = [4, 0, 9, 5, 4];

cases = struct('per', {}, 'mcsInit', {}, 'mcsTrace', {});
for k = 1:numel(perSeqs)
    per = perSeqs{k}(:).';
    mcsInit = mcsInits(k);
    trace = zeros(1, numel(per));
    st = [];
    mcs = mcsInit;
    for t = 1:numel(per)
        trace(t) = mcs;
        [mcs, st] = rateController(per(t), mcs, st);
    end
    cases(k) = struct('per', per, 'mcsInit', mcsInit, 'mcsTrace', trace); %#ok<AGROW>
end

save(outFile, 'constants', 'cases', '-v7');
fprintf('genRateControllerFixture: saved %d cases to %s\n', numel(cases), outFile);
end

function x = local_noise(n, lo, hi, seed)
s = RandStream('twister', 'Seed', seed);
x = lo + (hi - lo) * rand(s, 1, n);
end
