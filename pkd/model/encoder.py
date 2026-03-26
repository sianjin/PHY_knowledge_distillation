"""Compositional conditioning encoder for configuration variables."""
import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    """Encode static configuration variables."""

    def __init__(self,
                 num_channel_models: int,
                 channel_emb_dim: int = 32,
                 hidden_dim: int = 128):
        super().__init__()

        # Channel model embedding
        self.channel_embedding = nn.Embedding(num_channel_models, channel_emb_dim)

        # MLP for static features
        # Input: [channel_emb, N_t, N_r, BW]
        input_dim = channel_emb_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, channel_model_id, N_t, N_r, BW):
        """
        Args:
            channel_model_id: (batch,) channel model IDs
            N_t, N_r, BW: (batch,) static config values

        Returns:
            h_static: (batch, hidden_dim)
        """
        # Embed channel model
        ech = self.channel_embedding(channel_model_id)

        # Concatenate all static features
        x = torch.cat([
            ech,
            N_t.unsqueeze(-1).float(),
            N_r.unsqueeze(-1).float(),
            BW.unsqueeze(-1).float()
        ], dim=-1)

        return self.mlp(x)


class SNREncoder(nn.Module):
    """Smooth encoding of average SNR."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, SNR_bar):
        """
        Args:
            SNR_bar: (batch,) average SNR in dB

        Returns:
            h_snr: (batch, hidden_dim)
        """
        return self.mlp(SNR_bar.unsqueeze(-1))


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation for rate adaptation.

    Implements Equation (21) from paper:
        e = concat(e_mcs, e_ss, e_R)
        ϖ_t = g^(ω)(e_mcs, e_ss, e_R)
        β_t = g^(β)(e_mcs, e_ss, e_R)
        h_t = ϖ_t ⊙ h_base,t + β_t
    """

    def __init__(self,
                 num_mcs: int,
                 num_nss: int,
                 num_R: int = 8,
                 mcs_emb_dim: int = 32,
                 nss_emb_dim: int = 16,
                 r_emb_dim: int = 8,
                 hidden_dim: int = 128):
        super().__init__()

        # Embeddings (Equation 21: e_mcs, e_ss, e_R)
        self.mcs_embedding = nn.Embedding(num_mcs, mcs_emb_dim)
        self.nss_embedding = nn.Embedding(num_nss, nss_emb_dim)
        self.r_embedding = nn.Embedding(num_R, r_emb_dim)

        # Alpha and beta generators (FiLM parameters)
        # Input: concat(e_mcs, e_ss, e_R)
        emb_dim = mcs_emb_dim + nss_emb_dim + r_emb_dim

        # g^(ω): generates ϖ_t (multiplicative modulation)
        self.alpha_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # g^(β): generates β_t (additive modulation)
        self.beta_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize alpha close to 1
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.ones_(self.alpha_net[-1].bias)

    def forward(self, mcs_id, nss_id, r_id, h_base):
        """
        Apply FiLM modulation conditioned on rate adaptation variables.

        Args:
            mcs_id: (batch,) MCS IDs (0-indexed: 0 to num_mcs-1)
            nss_id: (batch,) number of spatial streams (0-indexed: 0 to num_nss-1)
            r_id: (batch,) resource allocation IDs (0-indexed: 0 to num_R-1)
            h_base: (batch, hidden_dim) base representation

        Returns:
            h: (batch, hidden_dim) modulated representation
        """
        # Embed rate adaptation variables (Equation 21)
        emcs = self.mcs_embedding(mcs_id)
        ess = self.nss_embedding(nss_id)
        er = self.r_embedding(r_id)

        # Concatenate embeddings: e = concat(e_mcs, e_ss, e_R)
        e = torch.cat([emcs, ess, er], dim=-1)

        # Generate modulation parameters (Equation 21)
        # ϖ_t = g^(ω)(e)
        alpha = self.alpha_net(e)
        # β_t = g^(β)(e)
        beta = self.beta_net(e)

        # Apply FiLM: h_t = ϖ_t ⊙ h_base,t + β_t
        h = alpha * h_base + beta

        return h


class CompositionalEncoder(nn.Module):
    """Complete compositional conditioning encoder."""

    def __init__(self,
                 num_channel_models: int,
                 num_mcs: int,
                 num_nss: int,
                 num_R: int = 8,
                 hidden_dim: int = 128,
                 channel_emb_dim: int = 32,
                 mcs_emb_dim: int = 32,
                 nss_emb_dim: int = 16,
                 r_emb_dim: int = 8):
        super().__init__()

        self.static_encoder = StaticEncoder(num_channel_models, channel_emb_dim, hidden_dim)
        self.snr_encoder = SNREncoder(hidden_dim)
        self.film = FiLMModulation(num_mcs, num_nss, num_R,
                                   mcs_emb_dim, nss_emb_dim, r_emb_dim,
                                   hidden_dim)

    def forward(self, config_dict):
        """
        Args:
            config_dict: dictionary with keys:
                - channel_model_id: (batch,)
                - N_t, N_r, BW: (batch,)
                - SNR_bar: (batch,)
                - MCS: (batch,) MCS values (0-indexed: 0 to num_mcs-1)
                - N_ss: (batch,) number of spatial streams (1-indexed: 1 to num_nss)
                - R_t: (batch,) resource allocation (0-indexed: 0 to num_R-1)

        Returns:
            h_t: (batch, hidden_dim) conditioning representation
        """
        # Static encoding
        h_static = self.static_encoder(
            config_dict['channel_model_id'],
            config_dict['N_t'],
            config_dict['N_r'],
            config_dict['BW']
        )

        # SNR encoding
        h_snr = self.snr_encoder(config_dict['SNR_bar'])

        # Base representation
        h_base = h_static + h_snr

        # MCS is already 0-indexed (0-9), N_ss is 1-indexed (1-4) so convert N_ss to 0-indexed
        # R_t is already 0-indexed (0 to num_R-1)
        # Create new tensors to avoid in-place modification issues
        mcs_idx = config_dict['MCS']
        nss_idx = config_dict['N_ss'] + (-1)
        r_idx = config_dict['R_t']

        # FiLM modulation with rate adaptation (Equation 21)
        h_t = self.film(mcs_idx, nss_idx, r_idx, h_base)

        return h_t
