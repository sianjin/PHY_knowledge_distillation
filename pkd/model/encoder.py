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
    """Feature-wise Linear Modulation for rate adaptation."""

    def __init__(self,
                 num_mcs: int,
                 num_nss: int,
                 mcs_emb_dim: int = 32,
                 nss_emb_dim: int = 16,
                 hidden_dim: int = 128):
        super().__init__()

        # Embeddings
        self.mcs_embedding = nn.Embedding(num_mcs, mcs_emb_dim)
        self.nss_embedding = nn.Embedding(num_nss, nss_emb_dim)

        # Alpha and beta generators (FiLM parameters)
        emb_dim = mcs_emb_dim + nss_emb_dim

        self.alpha_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize alpha close to 1
        nn.init.zeros_(self.alpha_net[-1].weight)
        nn.init.ones_(self.alpha_net[-1].bias)

    def forward(self, mcs_id, nss_id, h_base):
        """
        Args:
            mcs_id: (batch,) MCS IDs
            nss_id: (batch,) number of spatial streams
            h_base: (batch, hidden_dim) base representation

        Returns:
            h: (batch, hidden_dim) modulated representation
        """
        # Embed rate adaptation variables
        emcs = self.mcs_embedding(mcs_id)
        ess = self.nss_embedding(nss_id)

        # Concatenate embeddings
        e = torch.cat([emcs, ess], dim=-1)

        # Generate modulation parameters (Equation 19 in paper)
        alpha = self.alpha_net(e)
        beta = self.beta_net(e)

        # Apply FiLM: h_t = α_t ⊙ h_base,t + β_t
        h = alpha * h_base + beta

        return h


class CompositionalEncoder(nn.Module):
    """Complete compositional conditioning encoder."""

    def __init__(self,
                 num_channel_models: int,
                 num_mcs: int,
                 num_nss: int,
                 hidden_dim: int = 128,
                 channel_emb_dim: int = 32,
                 mcs_emb_dim: int = 32,
                 nss_emb_dim: int = 16):
        super().__init__()

        self.static_encoder = StaticEncoder(num_channel_models, channel_emb_dim, hidden_dim)
        self.snr_encoder = SNREncoder(hidden_dim)
        self.film = FiLMModulation(num_mcs, num_nss, mcs_emb_dim, nss_emb_dim, hidden_dim)

    def forward(self, config_dict):
        """
        Args:
            config_dict: dictionary with keys:
                - channel_model_id: (batch,)
                - N_t, N_r, BW: (batch,)
                - SNR_bar: (batch,)
                - MCS: (batch,) MCS values (1-indexed: 1 to num_mcs)
                - N_ss: (batch,) number of spatial streams (1-indexed: 1 to num_nss)

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

        # Convert 1-indexed MCS and N_ss to 0-indexed for embeddings
        # Create new tensors to avoid in-place modification issues
        mcs_idx = config_dict['MCS'] + (-1)
        nss_idx = config_dict['N_ss'] + (-1)

        # FiLM modulation with rate adaptation
        h_t = self.film(mcs_idx, nss_idx, h_base)

        return h_t
