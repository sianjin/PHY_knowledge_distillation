function out = box0RateControl(simParams, betaVec, mcsList, snrTraj, mcsInit)
%box0RateControl One closed-loop rate-adaptation realization (Fig. 15 teacher).
%
%   out = box0RateControl(simParams, betaVec, mcsList, snrTraj, mcsInit)
%   runs ONE independent realization of the closed-loop rate-adaptation
%   experiment for the MATLAB PHY teacher:
%
%       gamma_eff,t  ->  PER_hat,t  ->  MCS_{t+1}
%
%   The static configuration (channel model, N_t, N_r, N_ss, BW) is fixed
%   for the whole run. The average SNR follows the common deterministic
%   trajectory snrTraj (same for teacher and student), and the MCS is
%   selected packet-by-packet by rateController(). Hence C_t ~= C.
%
%   This is the abstraction-only path (no waveform decode): a single TGax
%   channel realization is propagated, per-packet effective SINR is
%   obtained from the EESM mapping of the per-subcarrier SINR at the
%   time-varying noise power N0_t, and the packet outcome is a coin flip
%   against the AWGN-LUT PER at the current MCS.
%
%   Inputs
%     simParams : struct from getBox0SimParams (fixed slice; .Config has
%                 NumSpaceTimeStreams etc. Its .MCS / .SNR are ignored here).
%     betaVec   : 1-by-numel(mcsList) calibrated EESM beta (see betaTable).
%     mcsList   : MCS indices addressed by betaVec, e.g. 0:9.
%     snrTraj   : T-by-1 common average-SNR trajectory (dB).
%     mcsInit   : starting MCS index (default: middle of mcsList).
%
%   Output struct fields (all length T unless noted)
%     mcs       : MCS index used for each packet
%     effSINR   : EESM effective SINR (dB) for each packet
%     perInst   : AWGN-LUT PER at (effSINR, mcs) for each packet
%     errorFlag : sampled packet-error indicator (uint8)
%     txPeriod  : packet transmit period (s), scalar
%     numPkt    : T

cfgHE = simParams.Config;
T = numel(snrTraj);

if nargin < 5 || isempty(mcsInit)
    mcsInit = mcsList(ceil(numel(mcsList)/2));
end

betaFor = @(m) betaVec(mcsList == m);

% --- OFDM / precoding setup (fixed for the whole run: MCS-independent) ---
ofdmInfo = wlanHEOFDMInfo('HE-Data', cfgHE);
ruIndex = 1;
sig = struct('Config', cfgHE, 'Field', 'data', 'OFDMConfig', ofdmInfo, 'RUIndex', ruIndex);
Wtx = wlan.internal.phy.l2sm.getPrecodingMatrix(sig);
Wtx = Wtx * sqrt(ofdmInfo.NumTones);
Ptxrx = 1; % 0 dBW transmit power

ind = wlanFieldIndices(cfgHE);

% --- Channel setup (same construction as box0Simulation) ---
tgaxChannel = simParams.Channel;
tgaxChannelInfo = info(tgaxChannel);
pathFilters = tgaxChannelInfo.ChannelFilterCoefficients;
chInfo = getChanInfoParams(tgaxChannel);

mimoChan = comm.MIMOChannel;
mimoChan.FadingTechnique = 'Sum of sinusoids';
mimoChan.SampleRate = wlanSampleRate(cfgHE);
mimoChan.AveragePathGains = tgaxChannelInfo.AveragePathGains;
mimoChan.PathDelays = tgaxChannelInfo.PathDelays;
mimoChan.SpatialCorrelationSpecification = 'Separate Tx Rx';
mimoChan.TransmitCorrelationMatrix = permute(chInfo.TxCorrelationMatrix, [3 2 1]);
mimoChan.ReceiveCorrelationMatrix = permute(chInfo.RxCorrelationMatrix, [3 2 1]);
wavelength = 3e8 / tgaxChannel.CarrierFrequency;
tgaxDopplerShift = tgaxChannel.EnvironmentalSpeed * (5/18) / wavelength;
mimoChan.MaximumDopplerShift = tgaxDopplerShift;
mimoChan.PathGainsOutputPort = true;
mimoChan.InitialTimeSource = 'Input port';
mimoChan.RandomStream = 'Global stream';

coherenceTime = 0.423 / tgaxDopplerShift;
txPeriod = coherenceTime / 4 * 1e6; % microseconds

% --- MCS-independent dummy TX frame ---------------------------------------
% The abstraction path never decodes; comm.MIMOChannel only needs an input
% of a representative length to advance the fading process. Using a fixed
% waveform (independent of the controller's MCS choices) guarantees the
% channel realization is identical regardless of rate-control decisions.
cfgDummy = cfgHE;
cfgDummy.MCS = mcsList(1);
psduLenDummy = getPSDULength(cfgDummy);
txDummy = wlanWaveformGenerator(randi([0 1], psduLenDummy*8, 1, 'int8'), ...
    cfgDummy, 'IdleTime', 0, 'WindowTransitionTime', 0);
txPad = [txDummy; zeros(50, cfgHE.NumTransmitAntennas)];

% --- Preallocate outputs ---
mcs       = zeros(T, 1, 'uint8');
effSINR   = nan(T, 1);
perInst   = nan(T, 1);
errorFlag = zeros(T, 1, 'uint8');

% --- Closed loop ---
ctrlState = [];
mcsCur = mcsInit;
txStartTime = 0;

for t = 1:T
    % Propagate channel for this packet
    [~, pathGains] = mimoChan(txPad, txStartTime*1e-6);
    txStartTime = txStartTime + txPeriod;

    heltfPathGains = pathGains(ind.HELTF(1):ind.HELTF(2), :, :, :, :);
    pktOffset = channelDelay(heltfPathGains, pathFilters);
    chan = helperPerfectChannelEstimate(heltfPathGains, pathFilters, ofdmInfo, pktOffset);
    Htxrx = permute(mean(chan, 2), [1 3 4 2]); % Nst-by-Nt-by-Nr

    % Per-subcarrier SINR at the time-varying noise power (MCS-independent)
    N0 = 10^(-snrTraj(t)/10);
    sinr = calculateSINR(Htxrx, Ptxrx, Wtx, N0);

    % EESM effective SINR for the CURRENT MCS
    beta = betaFor(mcsCur);
    es = tgaxLinkPerformanceModel.effectiveSINR(sinr, beta);

    % AWGN-LUT PER at (effective SINR, current MCS), payload length 1000 B
    per = tgaxLinkPerformanceModel.estimatePER(es, cfgHE.APEPLength, 'HE_SU', mcsCur, 'LDPC');

    % Sample packet outcome
    err = rand(1) <= per;

    mcs(t)       = mcsCur;
    effSINR(t)   = es;
    perInst(t)   = per;
    errorFlag(t) = err;

    % Rate-control update for the next packet
    [mcsCur, ctrlState] = rateController(per, mcsCur, ctrlState);
end

out = struct;
out.mcs = mcs;
out.effSINR = effSINR;
out.perInst = perInst;
out.errorFlag = errorFlag;
out.txPeriod = txPeriod / 1e6; % seconds
out.numPkt = T;

disp([char(cfgHE.ChannelBandwidth) ', ' ...
      char(simParams.DelayProfile) ', ' ...
      num2str(simParams.NumTransmitAntennas) 'x' ...
      num2str(simParams.NumReceiveAntennas) ':' ...
      num2str(cfgHE.NumSpaceTimeStreams) ', ' ...
      'rate-control run completed, ' num2str(T) ' packets, ' ...
      'mean MCS ' num2str(mean(double(mcs)), '%.2f') ', ' ...
      'PER ' num2str(mean(double(errorFlag)), '%.3f')]);
end

%% Get spatial correlation parameters
function chInfo = getChanInfoParams(tgaxChannel)
   modelConfig = struct( ...
              'NumTransmitAntennas', tgaxChannel.NumTransmitAntennas, ...
              'NumReceiveAntennas', tgaxChannel.NumReceiveAntennas, ...
              'TransmitAntennaSpacing', tgaxChannel.TransmitAntennaSpacing, ...
              'ReceiveAntennaSpacing', tgaxChannel.ReceiveAntennaSpacing, ...
              'DelayProfile', tgaxChannel.DelayProfile, ...
              'UserIndex', tgaxChannel.UserIndex, ...
              'ChannelBandwidth', tgaxChannel.ChannelBandwidth, ...
              'TransmitReceiveDistance', tgaxChannel.TransmitReceiveDistance, ...
              'CarrierFrequency', tgaxChannel.CarrierFrequency, ...
              'TransmissionDirection', tgaxChannel.TransmissionDirection, ...
              'NumPenetratedFloors', tgaxChannel.NumPenetratedFloors, ...
              'NumPenetratedWalls', tgaxChannel.NumPenetratedWalls, ...
              'WallPenetrationLoss', tgaxChannel.WallPenetrationLoss, ...
              'FormatType', class(tgaxChannel), ...
              'InputDataType', 'double');
    chInfo = spatialCorrelation(modelConfig);
end
