function corrPHYSim(cbw,chan,mcs,isnr,numTxRx,numSs,betaOpt)
maxNumErrors = 1e3;  % The maximum number of packet errors at an SNR point
maxNumPackets = 1e3; % The maximum number of packets at an SNR point
N_seq = 50; % Number of sequences
T = maxNumPackets; % Sequence length

% Fixed PHY configuration for all simulations
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = cbw; % Channel bandwidth
cfgHE.APEPLength = 1000;          % Payload length in bytes
cfgHE.ChannelCoding = 'LDPC';     % Channel coding

% Generate a structure array of simulation configurations. Each element is
% one SNR point to simulate.
simParams = getBox0SimParams(chan,numTxRx,numSs,mcs,cfgHE,maxNumErrors,maxNumPackets,isnr);
snr = simParams.SNR;

% Preallocate
gamma_eff   = zeros(N_seq, T, 'single');
packet_abs_error = zeros(N_seq, T, 'uint8');
config      = zeros(N_seq, 7, 'single');

% Create deterministic per-iteration seeds
baseSeed = 12345; 
seed = int64( baseSeed + (0:N_seq-1)' );

for n = 1:N_seq
    % Independent RNG per iteration (reproducible)
    rng(double(seed(n)), 'twister'); 

    % Run correlated box 0 PHY simulator
    results = box0Simulation(simParams,betaOpt);

    % Validate expected sizes once (asserts are cheap vs debugging time)
    gamma_eff(n, :)  = single(results.snreffStore(1:T));
    packet_abs_error(n, :) = uint8(results.perAbsStore(1:T));

    % Per-sequence configuration (constant within iteration)
    config(n, :) = single([simParams.ChannelModelID, numTxRx(1), numTxRx(2), simParams.BW, snr, mcs, numSs]);
end
end


