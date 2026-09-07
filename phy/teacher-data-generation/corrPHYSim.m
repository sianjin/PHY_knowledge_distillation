function corrPHYSim(cbw,chan,mcs,isnr,numTxRx,numSs,betaOpt)
maxNumErrors = 1e3;  % The maximum number of packet errors at an SNR point
maxNumPackets = 1e3; % The maximum number of packets at an SNR point
N_seq = 50; % Number of sequences
T = maxNumPackets; % Sequence length

% Fixed PHY configuration for all simulations
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = cbw; % Channel bandwidth
bandwidth = cfgHE.ChannelBandwidth;
cfgHE.APEPLength = 1000;          % Payload length in bytes
cfgHE.ChannelCoding = 'LDPC';     % Channel coding

% Generate a structure array of simulation configurations. Each element is
% one SNR point to simulate.
simParams = getBox0SimParams(chan,numTxRx,numSs,mcs,cfgHE,maxNumErrors,maxNumPackets,isnr);
snr = simParams.SNR;

% Preallocate
gamma_eff   = zeros(N_seq, T, 'single');
packet_error  = zeros(N_seq, T, 'uint8');
packet_abs_error = zeros(N_seq, T, 'uint8');
config      = zeros(N_seq, 7, 'single');
runtime_sec = zeros(N_seq, 1, 'single');

% Create deterministic per-iteration seeds
baseSeed = 12345; 
seed = int64( baseSeed + (0:N_seq-1)' );

parfor n = 1:N_seq
    % Independent RNG per iteration (reproducible)
    rng(double(seed(n)), 'twister'); 

    tStart = tic;

    % Run correlated box 0 PHY simulator
    results = box0Simulation(simParams,betaOpt);

    % Validate expected sizes once (asserts are cheap vs debugging time)
    gamma_eff(n, :)  = single(results.snreffStore(1:T));
    packet_error(n, :) = uint8(results.perStore(1:T));
    packet_abs_error(n, :) = uint8(results.perAbsStore(1:T));

    runtime_sec(n) = single(toc(tStart));

    % Per-sequence configuration (constant within iteration)
    config(n, :) = single([simParams.ChannelModelID, numTxRx(1), numTxRx(2), simParams.BW, snr, mcs, numSs]);
end

% Save as HDF5-backed MAT
folder = '/local-ssd/sianjin/EESM-log-AR/data'; % your folder here
fname_I = sprintf('%s_%s_%s-by-%s-by-%s_MCS%s_SNR%s.mat', ...
    bandwidth, char(chan), num2str(numTxRx(1)), num2str(numTxRx(2)), ...
    num2str(numSs), num2str(mcs), num2str(snr));
save(fullfile(folder,fname_I), 'gamma_eff', 'packet_error', 'packet_abs_error', 'config', 'runtime_sec', 'seed', '-v7.3');
% save(fname_I, 'gamma_eff', 'packet_error', 'packet_abs_error', 'config', 'runtime_sec', 'seed', '-v7.3');
end


