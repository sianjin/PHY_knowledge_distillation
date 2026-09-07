classdef tgaxLinkPerformanceModel
%tgaxLinkPerformanceModel Create a link performance model object
%   abstraction = tgaxLinkPerformanceModel returns a TGax link abstraction
%   model. This model is used to estimate the packet error rate for an
%   802.11ax single-user link assuming perfect synchronization.
%
%   tgaxLinkPerformanceModel methods:
%
%   estimatePER             - Returns the estimate packet error rate given 
%                             the effective SINR.
%   selectAWGNLUT           - Returns the appropriate AWGN lookup table.
%
%   See also calculateSINR.

%   Copyright 2019-2026 The MathWorks, Inc.

methods (Static)
    function per = estimatePER(snreff,varargin)
        % PER = estimatePER(SINREff,DATALENGTH,FORMAT,MCS,CODING)
        % returns the estimated packet error rate and effective SINR.
        %
        % SINREff is effective SINR in dB.
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
        % PER = estimatePER(SINR,CFGSU) returns the
        % estimated packet error rate given the single-user format
        % configuration object CFGSU. CFGSU is a format configuration
        % object of type wlanHESUConfig, wlanVHTConfig, wlanHTConfig, or
        % wlanNonHTConfig.
        %
        % PER = estimatePER(SINR,CFGMU,USERIDX)
        % returns the estimated packet error rate given the OFDMA
        % multi-user format configuration object CFGMU and user index
        % USERIDX. CFGMU is a format configuration object of type
        % wlanHEMUConfig.

        if nargin<5
            cfg = varargin{1};
            if nargin==3
                % PER = estimatePER(OBJ,SINR,CFGMU,USERIDX)
                assert(isa(cfg,'wlanHEMUConfig'))
                userIdx = varargin{2};
                mcs = cfg.User{userIdx}.MCS;
                dataLength = cfg.User{userIdx}.APEPLength;
                coding = cfg.User{userIdx}.ChannelCoding;
                format = 'HE_MU';
            else
                % PER = estimatePER(OBJ,SINR,CFGSU)
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
            % PER = estimatePER(OBJ,SINR,DATALENGTH,FORMAT,MCS,CODING)
            narginchk(5,5)
            dataLength = varargin{1};
            format = varargin{2};
            mcs = varargin{3};
            coding = varargin{4};
        end

        % Estimate the packet error rate for the given SNR and configuration
        per = wlan.internal.phy.l2sm.estimatePER(snreff,format,mcs,coding,dataLength);
    end

    function snreff = effectiveSINR(sinr_dB,beta)
        %SNR_EFF = EFFECTIVESINR(SINR, BETA) returns the EESM effective
        %SINR in dB from a set of (per‑resource/per‑symbol) SINR values in
        %dB using an exponential mapping controlled by BETA.

        % Convert to linear in double (avoid float issues)
        sinr_lin = 10.^(double(sinr_dB)/10);

        % a = -sinr_lin/beta (will be <= 0)
        a = -sinr_lin ./ double(beta);

        % Stable log-mean-exp: log(mean(exp(a)))
        % log(mean(exp(a))) = logsumexp(a) - log(N)
        amax = max(a(:));                % closest to 0
        % exp(a-amax) is in [0,1], safe
        s = sum(exp(a - amax), 'all');
        N = numel(a);
        log_mean_exp = amax + log(s) - log(N);

        snreff_lin = -double(beta) * log_mean_exp;

        % Safety clamp: snreff_lin should be positive
        snreff_lin = max(snreff_lin, realmin('double'));

        snreff = 10*log10(snreff_lin);
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