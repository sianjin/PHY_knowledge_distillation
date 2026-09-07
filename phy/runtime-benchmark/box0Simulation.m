function out = box0Simulation(simParams,beta)
% box0Simulation Example helper function

% Extract configuration
cfgHE = simParams.Config;
maxNumPackets = simParams.MaxNumPackets;
% maxNumErrors = simParams.MaxNumErrors;
snr = simParams.SNR;
% Create an NDP packet with the correct number of space-time streams to
% generate enough LTF symbols
cfgNDP = wlanHESUConfig('APEPLength',0,'GuardInterval',0.8); % No data in an NDP
cfgNDP.ChannelBandwidth = cfgHE.ChannelBandwidth;
cfgNDP.NumTransmitAntennas = cfgHE.NumTransmitAntennas;
cfgNDP.NumSpaceTimeStreams = cfgHE.NumTransmitAntennas;

% Indices to extract fields from the PPDU
ind = wlanFieldIndices(cfgHE);

% Get occupied subcarrier indices and OFDM parameters
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHE);

% Create an instance of the AWGN channel per SNR point simulated
awgnChannel = comm.AWGNChannel;
awgnChannel.NoiseMethod = 'Signal to noise ratio (SNR)';
% Account for noise energy in nulls so the SNR is defined per
% active subcarrier
% snr here refers to the SNR on active subcarriers
awgnChannel.SNR = snr-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones);

% For abstraction
ruIndex = 1;
sig = struct('Config',cfgHE,'Field','data','OFDMConfig',ofdmInfo,'RUIndex',ruIndex);
Wtx = wlan.internal.phy.l2sm.getPrecodingMatrix(sig); % Include cyclic shift and scaling applied per STS
Wtx = Wtx*sqrt(ofdmInfo.NumTones); % Scale back to unit power per subcarrier
N0 = (10^(-snr/10));

% Get path filers for last channel (same for all channels)
tgaxChannel = simParams.Channel;
tgaxChannelInfo = info(tgaxChannel);
pathFilters = tgaxChannelInfo.ChannelFilterCoefficients; % [NP numChannelTaps]
chInfo = getChanInfoParams(tgaxChannel); % Get Tx and Rx antenna correlation matrices

% Create and configure comm.MIMOChannel same as wlanTGaxChannel
% Doppler spectrum: Jakes; This is a must if FadingTechnique is set to SOS
mimoChan = comm.MIMOChannel;
mimoChan.FadingTechnique = 'Sum of sinusoids'; % Set FadingTechnique to SOS
mimoChan.SampleRate = wlanSampleRate(cfgHE);
mimoChan.AveragePathGains = tgaxChannelInfo.AveragePathGains;
mimoChan.PathDelays = tgaxChannelInfo.PathDelays;
mimoChan.SpatialCorrelationSpecification = 'Separate Tx Rx';
mimoChan.TransmitCorrelationMatrix  = permute(chInfo.TxCorrelationMatrix,[3 2 1]); 
mimoChan.ReceiveCorrelationMatrix  = permute(chInfo.RxCorrelationMatrix,[3 2 1]);
wavelength = 3e8/tgaxChannel.CarrierFrequency;
tgaxDopplerShift = tgaxChannel.EnvironmentalSpeed*(5/18)/wavelength; % Change km/h to m/s
mimoChan.MaximumDopplerShift = tgaxDopplerShift; % For Jakes model
mimoChan.PathGainsOutputPort = true; 
mimoChan.InitialTimeSource = 'Input port'; % Set comm.MIMOChannel property to define the InitialTime for each packet
mimoChan.RandomStream = 'Global stream';

% Transmit period in microseconds
coherenceTime = 0.423/tgaxDopplerShift;
txPeriod = coherenceTime/4*1e6; 

% Doppler-Symbol Period product
prodDoppPeriod = txPeriod*tgaxDopplerShift/10^6;

% Loop to simulate multiple packets
% perStore = nan(maxNumPackets,1);
perAbsStore = nan(maxNumPackets,1);
perAbsRawStore = nan(maxNumPackets,1);
snreffStore = nan(maxNumPackets,1);
% sinrStore = nan(ofdmInfo.NumTones,cfgHE.NumSpaceTimeStreams,maxNumPackets); % Nsc-by-Nsts-by-maxNumPackets
% numPacketErrors = 0;
numPacketErrorsAbs = 0;
numPkt = 1; % Index of packet transmitted
% Assume different packets separated by packet TX time and SIFS
% 1st Packet---SIFS---2nd  Packet---SIFS---3rd Packet---SIFS ...
txStartTime = 0; % Initial transmit time for the 1st packet (reference time 0)
while numPkt<=maxNumPackets
    % Generate a packet with random PSDU
    psduLength = getPSDULength(cfgHE); % PSDU length in bytes
    txPSDU = randi([0 1],psduLength*8,1,'int8');
    tx = wlanWaveformGenerator(txPSDU,cfgHE,'IdleTime',0,'WindowTransitionTime',0);

    % Add trailing zeros to allow for channel delay
    txPad = [tx; zeros(50,cfgHE.NumTransmitAntennas)];
    
    % Pass through comm.MIMOChannel
    [~,pathGains] = mimoChan(txPad,txStartTime*1e-6);  
     
    % Update txStartTime for the next sample instant
    txStartTime = txStartTime+txPeriod;
        
    % Get perfect timing offset and channel matrix for HE-LTF field
    heltfPathGains = pathGains(ind.HELTF(1):ind.HELTF(2),:,:,:,:);
    pktOffset = channelDelay(heltfPathGains,pathFilters);
    chan = helperPerfectChannelEstimate(heltfPathGains,pathFilters,ofdmInfo,pktOffset);
    
    % Calculate SINR using abstraction
    % As multiple symbols returned average over symbols and permute
    % for calculations
    Htxrx = permute(mean(chan,2),[1 3 4 2]); % Nst-by-Nt-by-Nr
    Ptxrx = 1; % Assume transmit power is 0dBW
    sinr = calculateSINR(Htxrx,Ptxrx,Wtx,N0);
    % sinrStore(:,:,numPkt) = sinr;
    
    % Link performance model - estimate PER using abstraction
    effSINR = tgaxLinkPerformanceModel.effectiveSINR(sinr,beta);
    perAbs = tgaxLinkPerformanceModel.estimatePER(effSINR,cfgHE);

    % Flip a coin for the abstracted PHY
    packetErrorAbs = rand(1)<=perAbs;
    numPacketErrorsAbs = numPacketErrorsAbs+packetErrorAbs;

    % Store outputs for analysis
    perAbsRawStore(numPkt) = perAbs;
    perAbsStore(numPkt) = packetErrorAbs;
    snreffStore(numPkt) = effSINR;

    numPkt = numPkt+1;
end

% Remove last increment
numPkt = numPkt-1;

% Calculate packet error rate (PER) at SNR point
% packetErrorRate = numPacketErrors/numPkt;
packetErrorRateAbs = numPacketErrorsAbs/numPkt;

% Return results
out = struct;
out.packetErrorRateAbs = packetErrorRateAbs;
% out.packetErrorRate = packetErrorRate;
% out.perStore = perStore;
% out.sinrStore = sinrStore;
out.numPkt = numPkt;
out.snreffStore = snreffStore;
out.perAbsRawStore = perAbsRawStore;
out.perAbsStore = perAbsStore;
out.txPeriod = txPeriod/10^6; % Tx period in sec
out.tgaxDopplerShift = tgaxDopplerShift;
out.prodDoppPeriod = prodDoppPeriod;

disp([char(cfgHE.ChannelBandwidth) ', '...
      char(simParams.DelayProfile) ', '...
      num2str(simParams.NumTransmitAntennas) 'x' ...
      num2str(simParams.NumReceiveAntennas) ':'...
      num2str(simParams.Config.NumSpaceTimeStreams) ','...
      ' MCS ' num2str(simParams.MCS) ','...
      ' SNR ' num2str(simParams.SNR) ','...
      ' completed after ' num2str(out.numPkt) ' packets']);
  
end

%% Get spatial correlation prameters
function chInfo = getChanInfoParams(tgaxChannel)
%getChanInfoParams Get TGax spatial parameters

   modelConfig = struct( ...
              'NumTransmitAntennas',tgaxChannel.NumTransmitAntennas, ...
              'NumReceiveAntennas',tgaxChannel.NumReceiveAntennas, ...
              'TransmitAntennaSpacing',tgaxChannel.TransmitAntennaSpacing, ...
              'ReceiveAntennaSpacing',tgaxChannel.ReceiveAntennaSpacing, ...
              'DelayProfile',tgaxChannel.DelayProfile, ...
              'UserIndex',tgaxChannel.UserIndex, ...
              'ChannelBandwidth',tgaxChannel.ChannelBandwidth, ...
              'TransmitReceiveDistance',tgaxChannel.TransmitReceiveDistance, ...
              'CarrierFrequency',tgaxChannel.CarrierFrequency, ...
              'TransmissionDirection',tgaxChannel.TransmissionDirection, ...
              'NumPenetratedFloors',tgaxChannel.NumPenetratedFloors, ...
              'NumPenetratedWalls',tgaxChannel.NumPenetratedWalls, ...
              'WallPenetrationLoss',tgaxChannel.WallPenetrationLoss, ...
              'FormatType',class(tgaxChannel), ...
              'InputDataType','double');
          
    chInfo = spatialCorrelation(modelConfig);

end
