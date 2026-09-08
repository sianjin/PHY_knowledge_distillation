function simParams = getBox0ValParams(chans,numTxRx,numSs,mcs,cfgHE,maxNumErrors,maxNumPackets)
% getBox0ValParams Helper function
%
% SNR now depends on (channel model, numTx, numRx, numSs, MCS).
% We encode the antenna+spatial-stream configuration via a 3-column matrix:
%   [numTx  numRx  numSs] (represented as numTx x numRx : numSs)

chans = reshape(string(chans), 1, []);

% These arrays define the value and order SNRs are defined
channelConfigs     = ["Model-B","Model-D"];
anteannaSNRConfigs = [ ...
    1 1 1;   % 1x1:1
    2 1 1;   % 2x1:1
    2 2 1;   % 2x2:1
    2 2 2;   % 2x2:2
    3 1 1;   % 3x1:1
    3 2 1;   % 3x2:1
    3 2 2;   % 3x2:2
    4 1 1;   % 4x1:1
    4 2 1;   % 4x2:1
    4 2 2;   % 4x2:2
    ];

% snr{ichan} is a cell array of size [numConfigs x 10(MCS)]
% Each row corresponds to the same row index in anteannaSNRConfigs.
snr = {
    % =========================
    % Model-B
    % =========================
    [ ...
    {... % 1x1:1
    [-10:4:2,5:3:20], ...  % MCS 0
    [-6:4:6,9:3:24], ...   % MCS 1
    [-4:4:8,11:3:26], ...  % MCS 2
    [-2:4:10,13:3:28], ... % MCS 3
    [2:4:14,17:3:32], ...  % MCS 4
    [6:4:18,21:3:36], ...  % MCS 5
    [8:4:20,23:3:38], ...  % MCS 6
    [10:4:22,25:3:40], ... % MCS 7
    [12:4:24,27:3:42], ... % MCS 8
    [14:4:26,29:3:44], ... % MCS 9
    }; ...
    {... % 2x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ...  
    {... % 2x2:1
    [-8:4:4, 6:2:16], ...  % MCS 0
    [-4:4:8, 10:2:20], ... % MCS 1
    [0:4:12, 14:2:24], ... % MCS 2
    [4:4:16, 18:2:28], ... % MCS 3
    [8:4:20, 22:2:32], ... % MCS 4
    [10:4:22, 24:2:34], ...% MCS 5
    [12:4:24, 26:2:36], ...% MCS 6
    [14:4:26, 28:2:38], ...% MCS 7
    [16:4:28, 30:2:40], ...% MCS 8
    [18:4:30, 32:2:42], ...% MCS 9
    }; ...
    {... % 2x2:2
    -4:3:23,  ... % MCS 0
    0:3:27,   ... % MCS 1
    4:3:31,   ... % MCS 2
    8:3:35,   ... % MCS 3
    12:3:39,  ... % MCS 4
    16:3:43,  ... % MCS 5
    20:3:47,  ... % MCS 6
    24:3:51,  ... % MCS 7
    28:3:55,  ... % MCS 8
    32:3:59,  ... % MCS 9
    }; ...
    {... % 3x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ... 
    {... % 3x2:1
    [-6:4:6, 8:2:18], ...  % MCS 0
    [-2:4:10, 12:2:22], ...% MCS 1
    [2:4:14, 16:2:26], ... % MCS 2
    [6:4:18, 20:2:30], ... % MCS 3
    [10:4:22, 24:2:32], ...% MCS 4
    [12:4:24, 26:2:36], ...% MCS 5
    [14:4:26, 28:2:38], ...% MCS 6
    [16:4:28, 30:2:40], ...% MCS 7
    [18:4:30, 32:2:42], ...% MCS 8
    [20:4:32, 34:2:44], ...% MCS 9
    }; ...
    {... % 3x2:2
    -2:3:25,  ... % MCS 0
    2:3:29,   ... % MCS 1
    6:3:33,   ... % MCS 2
    10:3:37,  ... % MCS 3
    14:3:41,  ... % MCS 4
    18:3:45,  ... % MCS 5
    22:3:49,  ... % MCS 6
    26:3:53,  ... % MCS 7
    30:3:57,  ... % MCS 8
    34:3:61,  ... % MCS 9
    }; ...
    {... % 4x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ... 
    {... % 4x2:1
    [-4:4:8,10:2:20], ...  % MCS 0
    [0:4:12,14:2:24], ...  % MCS 1
    [4:4:16,18:2:28], ...  % MCS 2
    [8:4:20,22:2:32], ...  % MCS 3
    [12:4:24,26:2:36], ... % MCS 4
    [14:4:26,28:2:38], ... % MCS 5
    [16:4:28,30:2:40], ... % MCS 6
    [18:4:30,32:2:42], ... % MCS 7
    [20:4:32,34:2:44], ... % MCS 8
    [22:4:34,36:2:46], ... % MCS 9
    }; ...
    {... % 4x2:2
    0:3:27,  ... % MCS 0
    4:3:31,  ... % MCS 1
    8:3:35,  ... % MCS 2
    12:3:39, ... % MCS 3
    16:3:43, ... % MCS 4
    20:3:47, ... % MCS 5
    24:3:51, ... % MCS 6
    28:3:55, ... % MCS 7
    32:3:59, ... % MCS 8
    36:3:63, ... % MCS 9
    }; ...
    ] ;

    % =========================
    % Model-D
    % =========================
    [ ...
    {... % 1x1:1
    [-10:4:2,5:3:20], ...  % MCS 0
    [-6:4:6,9:3:24], ...   % MCS 1
    [-4:4:8,11:3:26], ...  % MCS 2
    [-2:4:10,13:3:28], ... % MCS 3
    [2:4:14,17:3:32], ...  % MCS 4
    [6:4:18,21:3:36], ...  % MCS 5
    [8:4:20,23:3:38], ...  % MCS 6
    [10:4:22,25:3:40], ... % MCS 7
    [12:4:24,27:3:42], ... % MCS 8
    [14:4:26,29:3:44], ... % MCS 9
    }; ...
    {... % 2x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ...
    {... % 2x2:1
    -10:2:8, ...  % MCS 0
    -6:2:12, ...  % MCS 1
    -2:2:16, ...  % MCS 2
    2:2:20,  ...  % MCS 3
    6:2:24,  ...  % MCS 4
    10:2:28, ...  % MCS 5
    12:2:30, ...  % MCS 6
    14:2:32, ...  % MCS 7
    16:2:34, ...  % MCS 8
    18:2:36, ...  % MCS 9
    }; ...
    {... % 2x2:2
    -2:2:16,  ... % MCS 0
    2:2:20,   ... % MCS 1
    6:2:24,   ... % MCS 2
    10:2:28,  ... % MCS 3
    14:2:32,  ... % MCS 4
    18:2:36,  ... % MCS 5
    20:2:38,  ... % MCS 6
    22:2:40,  ... % MCS 7
    24:2:42,  ... % MCS 8
    26:2:44,  ... % MCS 9
    }; ...
    {... % 3x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ... 
    {... % 3x2:1
    -9:2:9, ...  % MCS 0
    -5:2:13, ... % MCS 1
    -1:2:17, ... % MCS 2
    3:2:21, ...  % MCS 3
    7:2:25, ...  % MCS 4
    11:2:29, ... % MCS 5
    13:2:31, ... % MCS 6
    15:2:33, ... % MCS 7
    17:2:35, ... % MCS 8
    19:2:37, ... % MCS 9
    }; ...
    {... % 3x2:2
    -1:2:17,  ... % MCS 0
    3:2:21,   ... % MCS 1
    7:2:25,   ... % MCS 2
    11:2:29,  ... % MCS 3
    15:2:33,  ... % MCS 4
    19:2:37,  ... % MCS 5
    21:2:39,  ... % MCS 6
    23:2:41,  ... % MCS 7
    25:2:43,  ... % MCS 8
    27:2:45,  ... % MCS 9
    }; ...
    {... % 4x1:1
    [-7:4:5,8:3:23], ...   % MCS 0
    [-3:4:9,11:3:27], ...  % MCS 1
    [-1:4:11,14:3:29], ... % MCS 2
    [1:4:13,16:3:31], ...  % MCS 3
    [5:4:17,20:3:35], ...  % MCS 4
    [9:4:21,24:3:39], ...  % MCS 5
    [11:4:23,26:3:41], ... % MCS 6
    [13:4:25,28:3:43], ... % MCS 7
    [15:4:27,30:3:45], ... % MCS 8
    [17:4:29,32:3:47], ... % MCS 9
    }; ... 
    {... % 4x2:1
    -8:2:10, ...  % MCS 0
    -4:2:14, ...  % MCS 1
    0:2:18, ...   % MCS 2
    4:2:22, ...   % MCS 3
    8:2:26, ...   % MCS 4
    12:2:30, ...  % MCS 5
    14:2:32, ...  % MCS 6
    16:2:34, ...  % MCS 7
    18:2:36, ...  % MCS 8
    20:2:38, ...  % MCS 9
    }; ...
    {... % 4x2:2
    0:2:18,  ... % MCS 0
    4:2:22,  ... % MCS 1
    8:2:26,  ... % MCS 2
    12:2:30, ... % MCS 3
    16:2:34, ... % MCS 4
    20:2:38, ... % MCS 5
    22:2:40, ... % MCS 6
    24:2:42, ... % MCS 7
    26:2:44, ... % MCS 8
    28:2:46, ... % MCS 9
    }; ...
    ] ...
    };

% -----------------------------
% Create channel configuration
% -----------------------------
tgaxChannel = wlanTGaxChannel;
tgaxChannel.DelayProfile = 'Model-D';
tgaxChannel.NumTransmitAntennas = cfgHE.NumTransmitAntennas;
tgaxChannel.NumReceiveAntennas = 1;
tgaxChannel.TransmitReceiveDistance = 15; % meters (NLOS)
tgaxChannel.ChannelBandwidth = cfgHE.ChannelBandwidth;
tgaxChannel.LargeScaleFadingEffect = 'None';
fs = wlanSampleRate(cfgHE);
tgaxChannel.SampleRate = fs;
tgaxChannel.PathGainsOutputPort = true;
tgaxChannel.NormalizeChannelOutputs = false;

% Reference sim param struct
simParamsRef = struct('MCS',0,'SNR',0,'RandomSubstream',0,'Config',cfgHE, ...
    'MaxNumPackets',maxNumPackets,'MaxNumErrors',maxNumErrors, ...
    'NumTransmitAntennas',0,'NumReceiveAntennas',0,'DelayProfile',"Model-B",...
    'Channel',tgaxChannel);

simParams = repmat(simParamsRef,0,0);

% -----------------------------
% Sanity checks
% -----------------------------
assert(numel(channelConfigs)==numel(snr), ...
    'snr must have one entry per channelConfigs element.');

% For each channel, snr{ichan} must have one row per antennaSNR config
assert(all(cellfun(@(x)size(x,1),snr) == size(anteannaSNRConfigs,1)), ...
    'Each snr{ichan} must have size [size(anteannaSNRConfigs,1) x 10].');

% -----------------------------
% Build simParams
% -----------------------------
for ichan = 1:numel(chans)
    channelIdx = find(chans(ichan)==channelConfigs, 1);
    if isempty(channelIdx)
        error('Unsupported channel "%s". Supported: %s', chans(ichan), join(channelConfigs,", "));
    end

    for itxrx = 1:size(numTxRx,1)
        cfgTriplet = [numTxRx(itxrx,1), numTxRx(itxrx,2), numSs];
        numTxRxSsIdx = find(all(anteannaSNRConfigs == cfgTriplet, 2), 1);

        if isempty(numTxRxSsIdx)
            error('Unsupported (numTx,numRx,numSs) = (%d,%d,%d). Supported rows in anteannaSNRConfigs:\n%s', ...
                cfgTriplet(1), cfgTriplet(2), cfgTriplet(3), mat2str(anteannaSNRConfigs));
        end

        for imcs = 1:numel(mcs)
            snrIdx = mcs(imcs) + 1; % 1..10

            snrVec = snr{channelIdx}{numTxRxSsIdx, snrIdx};
            for isnr = 1:numel(snrVec)
                sp = simParamsRef;

                % Simulation-specific parameters
                sp.MCS = mcs(imcs);
                sp.NumTransmitAntennas = cfgTriplet(1);
                sp.NumReceiveAntennas  = cfgTriplet(2);
                sp.DelayProfile = chans(ichan);

                % Reproducible random substream
                sp.RandomSubstream = isnr;

                % PHY config
                sp.Config.MCS = mcs(imcs);
                sp.Config.NumTransmitAntennas   = cfgTriplet(1);
                sp.Config.NumSpaceTimeStreams   = cfgTriplet(3);
                sp.Config.SpatialMapping        = 'Fourier';

                % Channel config
                sp.Channel = clone(tgaxChannel);
                sp.Channel.DelayProfile        = chans(ichan);
                sp.Channel.NumTransmitAntennas = cfgTriplet(1);
                sp.Channel.NumReceiveAntennas  = cfgTriplet(2);

                % Lookup SNR
                sp.SNR = snrVec(isnr);

                simParams = [simParams sp]; %#ok<AGROW>
            end
        end
    end
end

end
