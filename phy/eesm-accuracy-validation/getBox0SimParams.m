function simParams = getBox0SimParams(chans,numTxRx,numSs,mcs,cfgHE,maxNumErrors,maxNumPackets)
% getBox0SimParams Example helper function

% These arrays define the value and order SNRs are defined
channelConfigs = ["Model-B","Model-D"];
anteannaSNRConfigs = [1 1; 4 2];

snr = {
    % Model-B
    [ ...
    {... % 1x1
    [-10:4:2,5:3:20], ...           % Nss = 1, BW20/40 -> [-10:4:2,5:3:20], ...              % MCS 0
    [-6:4:6,9:3:24], ...            % Nss = 1, BW20/40 -> [-6:4:6,9:3:24], ...               % MCS 1
    [-4:4:8,11:3:26], ...           % Nss = 1, BW20/40 -> [-4:4:8,11:3:26], ...              % MCS 2
    [-2:4:10,13:3:28], ...          % Nss = 1, BW20/40 -> [-2:4:10,13:3:28], ...             % MCS 3
    [2:4:14,17:3:32], ...           % Nss = 1, BW20/40 -> [2:4:14,17:3:32], ...              % MCS 4
    [6:4:18,21:3:36], ...           % Nss = 1, BW20/40 -> [6:4:18,21:3:36], ...              % MCS 5
    [8:4:20,23:3:38], ...           % Nss = 1, BW20/40 -> [8:4:20,23:3:38], ...              % MCS 6
    [10:4:22,25:3:40], ...          % Nss = 1, BW20/40 -> [10:4:22,25:3:40], ...             % MCS 7
    [12:4:24,27:3:42], ...          % Nss = 1, BW20/40 -> [12:4:24,27:3:42], ...             % MCS 8
    [14:4:26,29:3:44], ...          % Nss = 1, BW20/40 -> [14:4:26,29:3:44], ...             % MCS 9
    }; ...
    {... % 4x2
    [-4:4:8,10:2:20], ...           % Nss = 1, BW20/40 -> [-4:4:8,10:2:20]; Nss = 2, BW20/40 -> 0:3:27, ...             % MCS 0
    [0:4:12,14:2:24], ...           % Nss = 1, BW20/40 -> [0:4:12,14:2:24]; Nss = 2, BW20/40 -> 4:3:31, ...             % MCS 1
    [4:4:16,18:2:28], ...           % Nss = 1, BW20/40 -> [4:4:16,18:2:28]; Nss = 2, BW20/40 -> 8:3:35, ...             % MCS 2
    [8:4:20,22:2:32], ...           % Nss = 1, BW20/40 -> [8:4:20,22:2:32]; Nss = 2, BW20/40 -> 12:3:39, ...            % MCS 3
    [12:4:24,26:2:36], ...          % Nss = 1, BW20/40 -> [12:4:24,26:2:36]; Nss = 2, BW20/40 -> 16:3:43, ...           % MCS 4
    [14:4:26,28:2:38], ...          % Nss = 1, BW20/40 -> [14:4:26,28:2:38]; Nss = 2, BW20/40 -> 20:3:47, ...           % MCS 5
    [16:4:28,30:2:40], ...          % Nss = 1, BW20/40 -> [16:4:28,30:2:40]; Nss = 2, BW20/40 -> 24:3:51, ...           % MCS 6
    [18:4:30,32:2:42], ...          % Nss = 1, BW20/40 -> [18:4:30,32:2:42]; Nss = 2, BW20/40 -> 28:3:55, ...           % MCS 7
    [20:4:32,34:2:44], ...          % Nss = 1, BW20/40 -> [20:4:32,34:2:44]; Nss = 2, BW20/40 -> 32:3:59, ...           % MCS 8
    [22:4:34,36:2:46], ...          % Nss = 1, BW20/40 -> [22:4:34,36:2:46]; Nss = 2, BW20/40 -> 36:3:63, ...           % MCS 9
    }; ...
    ];

    % Model-D
    [ ...
    {... % 1x1
    [-10:4:2,5:3:20], ...           % Nss = 1, BW20/40 -> [-10:4:2,5:3:20], ...              % MCS 0
    [-6:4:6,9:3:24], ...            % Nss = 1, BW20/40 -> [-6:4:6,9:3:24], ...               % MCS 1
    [-4:4:8,11:3:26], ...           % Nss = 1, BW20/40 -> [-4:4:8,11:3:26], ...              % MCS 2
    [-2:4:10,13:3:28], ...          % Nss = 1, BW20/40 -> [-2:4:10,13:3:28], ...             % MCS 3
    [2:4:14,17:3:32], ...           % Nss = 1, BW20/40 -> [2:4:14,17:3:32], ...              % MCS 4
    [6:4:18,21:3:36], ...           % Nss = 1, BW20/40 -> [6:4:18,21:3:36], ...              % MCS 5
    [8:4:20,23:3:38], ...           % Nss = 1, BW20/40 -> [8:4:20,23:3:38], ...              % MCS 6
    [10:4:22,25:3:40], ...          % Nss = 1, BW20/40 -> [10:4:22,25:3:40], ...             % MCS 7
    [12:4:24,27:3:42], ...          % Nss = 1, BW20/40 -> [12:4:24,27:3:42], ...             % MCS 8
    [14:4:26,29:3:44], ...          % Nss = 1, BW20/40 -> [14:4:26,29:3:44], ...             % MCS 9
    }; ...
    {... % 4x2
    -8:2:10, ...                    % Nss = 1, BW20/40 -> -8:2:10; Nss = 2, BW20/40 -> 0:2:18, ...            % MCS 0
    -4:2:14, ...                    % Nss = 1, BW20/40 -> -4:2:14; Nss = 2, BW20/40 -> 4:2:22, ...            % MCS 1
    0:2:18, ...                     % Nss = 1, BW20/40 -> 0:2:18; Nss = 2, BW20/40 -> 8:2:26, ...             % MCS 2
    4:2:22, ...                     % Nss = 1, BW20/40 -> 4:2:22; Nss = 2, BW20/40 -> 12:2:30, ...            % MCS 3
    8:2:26, ...                     % Nss = 1, BW20/40 -> 8:2:26; Nss = 2, BW20/40 -> 16:2:34, ...            % MCS 4
    12:2:30, ...                    % Nss = 1, BW20/40 -> 12:2:30; Nss = 2, BW20/40 -> 20:2:38, ...           % MCS 5
    14:2:32, ...                    % Nss = 1, BW20/40 -> 14:2:32; Nss = 2, BW20/40 -> 22:2:40, ...           % MCS 6
    16:2:34, ...                    % Nss = 1, BW20/40 -> 16:2:34; Nss = 2, BW20/40 -> 24:2:42, ...           % MCS 7
    18:2:36, ...                    % Nss = 1, BW20/40 -> 18:2:36; Nss = 2, BW20/40 -> 26:2:44, ...           % MCS 8
    20:2:38, ...                    % Nss = 1, BW20/40 -> 20:2:38; Nss = 2, BW20/40 -> 28:2:46, ...           % MCS 9
    }; ...
    ] ...
    };

% Create channel configuration
tgaxChannel = wlanTGaxChannel;
tgaxChannel.DelayProfile = 'Model-D';
tgaxChannel.NumTransmitAntennas = cfgHE.NumTransmitAntennas;
tgaxChannel.NumReceiveAntennas = 1;
tgaxChannel.TransmitReceiveDistance = 15; % Distance in meters for NLOS
tgaxChannel.ChannelBandwidth = cfgHE.ChannelBandwidth;
tgaxChannel.LargeScaleFadingEffect = 'None';
fs = wlanSampleRate(cfgHE);
tgaxChannel.SampleRate = fs;
tgaxChannel.PathGainsOutputPort = true;
tgaxChannel.NormalizeChannelOutputs = false;

% Generate a structure array containing the simulation parameters,
% simParams. Each element contains the parameters for a simulation.
simParamsRef = struct('MCS',0,'SNR',0,'RandomSubstream',0,'Config',cfgHE, ...
    'MaxNumPackets',maxNumPackets,'MaxNumErrors',maxNumErrors, ...
    'NumTransmitAntennas',0,'NumReceiveAntennas',0,'DelayProfile',"Model-B",...
    'Channel',tgaxChannel);
simParams = repmat(simParamsRef,0,0);
% There must be a SNR cell for each channel
assert(all(numel(channelConfigs)==numel(snr)))
% There must be a SNR cell element for each MIMO configuration
assert(all(size(anteannaSNRConfigs,1)==cellfun(@(x)size(x,1),snr)))
for ichan = 1:numel(chans)
    channelIdx = chans(ichan)==channelConfigs;
    for itxrx = 1:size(numTxRx,1)
        numTxRxIdx = all(numTxRx(itxrx,:)==anteannaSNRConfigs,2);
        for imcs = 1:numel(mcs)
            snrIdx = mcs(imcs)+1;
            for isnr = 1:numel([snr{channelIdx}{numTxRxIdx,snrIdx}])
                % Set simulation specific parameters
                sp = simParamsRef;
                sp.MCS = mcs(imcs);
                sp.NumTransmitAntennas = numTxRx(itxrx,1);
                sp.NumReceiveAntennas = numTxRx(itxrx,2);
                sp.DelayProfile = chans(ichan);

                % Set random substream for reproducible results
                sp.RandomSubstream = isnr;
                
                % Setup PHY configuration
                sp.Config.MCS = mcs(imcs);
                sp.Config.NumTransmitAntennas = numTxRx(itxrx,1);
                sp.Config.NumSpaceTimeStreams = numSs;
                sp.Config.SpatialMapping = 'Fourier';

                % Configure channel model
                sp.Channel = clone(tgaxChannel);
                sp.Channel.DelayProfile = chans(ichan);
                sp.Channel.NumTransmitAntennas = numTxRx(itxrx,1);
                sp.Channel.NumReceiveAntennas = numTxRx(itxrx,2);

                % Lookup SNR to simulate
                sp.SNR = snr{channelIdx}{numTxRxIdx,snrIdx}(isnr);

                % Append to other tests
                simParams = [simParams sp]; %#ok<AGROW>
            end
        end
    end
end

end