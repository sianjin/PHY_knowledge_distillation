classdef tgaxLinkPerformanceModel
%tgaxLinkPerformanceModel Create a link performance model object
%   abstraction = tgaxLinkPerformanceModel returns a TGax link abstraction
%   model. This model is used to estimate the packet error rate for an
%   802.11ax single-user link assuming perfect synchronization.
%
%   tgaxLinkPerformanceModel methods:
%
%   estimateLinkPerformance - Returns the expected packet error rate given 
%                             the SINR per subcarrier.
%   effectiveSINR           - Calculates the effective SINR given the SINR 
%                             per subcarrier.
%   estimatePER             - Returns the estimate packet error rate given 
%                             the effective SINR.
%   selectAWGNLUT           - Returns the appropriate AWGN lookup table.
%
%   % Example: Estimate packet error rate for a link with HE SU packet.
%
%   abstraction = tgaxLinkPerformanceModel;
%   sinrVal = 30*rand(224,2); % Number of subcarriers-by-number of spatial streams
%   dataLen = 1e3; % Bytes
%   mcs = 3;
%   coding = 'BCC';
%   per = abstraction.estimateLinkPerformance(sinrVal,dataLen,'HE_SU',mcs,coding);
%
%   See also calculateSINR.

%   Copyright 2019-2026 The MathWorks, Inc.

methods (Static)
    function [per,snreff] = estimateLinkPerformance(sinr,varargin)
        % [PER,SNREFF] = estimateLinkPerformance(SINR,DATALENGTH,FORMAT,MCS,CODING)
        % returns the estimated packet error rate and effective SINR.
        %
        % SINR is an array containing the SINR for each subcarrier,
        % symbol and spatial stream.
        %
        % DATALENGTH is the payload length in bytes.
        % 
        % FORMAT is one of 'NonHT', 'HTMixed', 'VHT', 'HE_SU', 'HE_EXT_SU',
        % 'HE_MU', 'HE_TB'.
        %
        % MCS is the modulation and coding scheme index and must be
        % 0-9.
        %
        % CODING is the channel coding used and must be 'BCC' or 'LDPC'.
        %
        % [PER,SNREFF] = estimateLinkPerformance(SINR,CFGSU) returns the
        % estimated packet error rate given the single-user format
        % configuration object CFGSU. CFGSU is a format configuration
        % object of type wlanHESUConfig, wlanVHTConfig, wlanHTConfig, or
        % wlanNonHTConfig.
        %
        % [PER,SNREFF] = estimateLinkPerformance(SINR,CFGMU,USERIDX)
        % returns the estimated packet error rate given the OFDMA
        % multi-user format configuration object CFGMU and user index
        % USERIDX. CFGMU is a format configuration object of type
        % wlanHEMUConfig.

        if nargin<5
            cfg = varargin{1};
            if nargin==3
                % [PER,SNREFF] = estimateLinkPerformance(OBJ,SINR,CFGMU,USERIDX)
                assert(isa(cfg,'wlanHEMUConfig'))
                userIdx = varargin{2};
                mcs = cfg.User{userIdx}.MCS;
                dataLength = cfg.User{userIdx}.APEPLength;
                coding = cfg.User{userIdx}.ChannelCoding;
                format = 'HE_MU';
            else
                % [PER,SNREFF] = estimateLinkPerformance(OBJ,SINR,CFGSU)
                mcs = cfg.MCS;

                % Get the channel coding and data long from the
                % configuration object
                switch class(cfg)
                    case {'wlanHESUConfig','wlanVHTConfig'}
                        dataLength = cfg.APEPLength;
                        coding = cfg.ChannelCoding;
                        format = 'HE_SU'; % Use HE-SU even if VHT as same MCS indices
                    case 'wlanHTConfig'
                        dataLength = cfg.PSDULength;
                        coding = cfg.ChannelCoding;
                        format = 'HTMixed';
                    case 'wlanNonHTConfig'
                        dataLength = cfg.PSDULength;
                        coding = 'BCC';
                        format = 'NonHT';
                    otherwise
                        error('Unexpected object');
                end
            end
        else
            % [PER,SNREFF] = estimateLinkPerformance(OBJ,SINR,DATALENGTH,FORMAT,MCS,CODING)
            narginchk(5,5)
            dataLength = varargin{1};
            format = varargin{2};
            mcs = varargin{3};
            coding = varargin{4};
        end

        [per,snreff] = wlan.internal.phy.l2sm.estimateLinkPerformance(sinr,dataLength,format,mcs,coding);
    end

    function [snreff,scrbir,avrbir] = effectiveSINR(sinr,format,mcs,alpha,beta)
        % [SNREFF,THETA,RBIR] = effectiveSINR(SINR,FORMAT,MCS) returns
        % the effective SNR, the RBIR per SINR (SCRBIR) and the average
        % RBIR before reverse mapping (AVRBIR).
        %
        % SINR is the SINR per subcarrier, symbol and spatial stream. 
        %
        % FORMAT is one of 'NonHT','HTMixed','VHT','HE_SU','HE_EXT_SU'.
        %
        % MCS is the modulation and coding scheme index for the specified
        % format.
        %
        % [...] = effectiveSINR(...,ALPHA,BETA) additionally allows
        % tuning parameters to be specified. If not provided 1 is
        % assumed for both.

        arguments
            % Tuning parameters
            sinr
            format
            mcs
            alpha double = 1
            beta double = 1
        end
        
        modscheme = wlan.internal.phy.l2sm.mcs2rate(format,mcs);
        [snreff,scrbir,avrbir] = wireless.internal.L2SM.calculateEffectiveSINR(sinr,modscheme,alpha,beta);
    end

    function [per,perPL0,L0,lut] = estimatePER(snreff,format,mcs,coding,dataLength)
        % [PER,PERPL0,L0,LUT] = estimatePER(SNREFF,FORMAT,MCS,CODING,DATALENGTH)
        % returns the packet error rate PER, for the reference data length,
        % PERPL0, the reference data length L0, and the selected AWGN
        % lookup table, LUT.
        %
        % SNREFF is the effective SNR.
        %
        % FORMAT is one of 'NonHT','HTMixed','VHT','HE_SU','HE_EXT_SU'.
        %
        % MCS is the modulation and coding scheme index for the specified
        % format.
        %
        % CODING is either 'BCC' or 'LDPC'.
        %
        % DATALENGTH is the PSDU length in bytes.

        [per,perPL0,L0,lut] = wlan.internal.phy.l2sm.estimatePER(snreff,format,mcs,coding,dataLength);
    end
    
    function [lut,L0] = selectAWGNLUT(format,mcs,coding,dataLength)
        % [LUT,L0] = selectAWGNLUT(FORMAT,MCS,CODING,DATALENGTH) returns
        % the appropriate AWGN lookup table, LUT, and reference data length
        % L0.
        %
        % LUT is a matrix containing the lookup table. Each row is of the
        % form [SNR PER].
        %
        % FORMAT is one of 'NonHT','HTMixed','VHT','HE_SU','HE_EXT_SU'.
        %
        % MCS is the HE modulation and coding scheme index and must be
        % 0-9.
        %
        % CODING is either 'BCC' or 'LDPC'.
        %
        % DATALENGTH is the PSDU length in bytes.

        [lut,L0] = wlan.internal.phy.l2sm.selectAWGNLUT(format,mcs,coding,dataLength);
    end
end
end