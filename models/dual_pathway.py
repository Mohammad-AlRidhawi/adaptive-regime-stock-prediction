"""Dual-pathway framework: autoencoder -> router -> normal/event NF -> adaptive blending."""

import torch
from torch import nn

from .autoencoder import Autoencoder
from .node_transformer import NodeTransformer


class DualPathwayFramework(nn.Module):
    """End-to-end inference graph for the regime-aware prediction system.

    Routing follows Section 3.3.2:
        e_t = || x_t - dec(enc(x_t)) ||_2^2
        if e_t < tau: y_t = y_normal
        else:        y_t = y_event(x_t, c_t)
        y_hat = alpha * y_normal + (1 - alpha) * y_event   (eq. 5)
    """

    def __init__(
        self,
        autoencoder: Autoencoder,
        normal_nf: NodeTransformer,
        event_nf: NodeTransformer,
        initial_tau: float = 0.0,
        initial_alpha: float = 0.5,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.normal_nf = normal_nf
        self.event_nf = event_nf
        self.register_buffer("tau", torch.tensor(initial_tau))
        self.register_buffer("alpha", torch.tensor(initial_alpha))

    def set_routing_parameters(self, tau: float, alpha: float) -> None:
        self.tau.fill_(tau)
        self.alpha.clamp_(0.0, 1.0)
        self.alpha.fill_(alpha)

    def reconstruction_error(self, x_router: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.reconstruction_error(x_router)

    def forward(
        self,
        features: torch.Tensor,
        stock_ids: torch.Tensor,
        x_router: torch.Tensor,
        context: torch.Tensor,
        regime_label: torch.Tensor | None = None,
    ) -> dict:
        e_t = self.reconstruction_error(x_router)
        y_normal = self.normal_nf(features, stock_ids)
        y_event = self.event_nf(features, stock_ids, context=context, regime_label=regime_label)

        alpha = self.alpha.clamp(0.0, 1.0)
        y_blend = alpha * y_normal + (1.0 - alpha) * y_event

        return {
            "reconstruction_error": e_t,
            "normal_prediction": y_normal,
            "event_prediction": y_event,
            "blended_prediction": y_blend,
            "routing_mask": (e_t < self.tau).float(),
        }
