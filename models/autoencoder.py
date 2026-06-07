"""Feed-forward autoencoder for regime detection via reconstruction error."""

import torch
from torch import nn


class Autoencoder(nn.Module):
    """Symmetric encoder-decoder trained on stable-VIX days.

    Architecture follows Section 3.3 of the manuscript:
        input (d_in) -> 64 -> 32 (latent) -> 32 -> 64 -> output (d_in)

    Reconstruction error e_t = || x_t - f_dec(f_enc(x_t)) ||_2^2 serves as the
    anomaly score that gates routing between the normal and event pathways.
    """

    def __init__(self, input_dim: int, hidden_layers=(64, 32), latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_layers[:-1]:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_layers[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample squared L2 reconstruction error."""
        x_hat, _ = self.forward(x)
        return torch.sum((x - x_hat) ** 2, dim=-1)
