"""Smoke tests for the model components."""

import torch

from models import Autoencoder, NodeTransformer
from models.dual_pathway import DualPathwayFramework


def test_autoencoder_forward_and_error():
    ae = Autoencoder(input_dim=17, hidden_layers=(64, 32), latent_dim=32)
    x = torch.randn(8, 17)
    x_hat, z = ae(x)
    assert x_hat.shape == x.shape
    assert z.shape == (8, 32)
    err = ae.reconstruction_error(x)
    assert err.shape == (8,)
    assert torch.all(err >= 0)


def test_node_transformer_shapes():
    nf = NodeTransformer(
        feature_dim=17, num_nodes=20, num_layers=2, num_heads=4,
        model_dim=64, ffn_dim=128, dropout=0.1, context_dim=0,
    )
    bsz, n, seq_len = 2, 20, 30
    features = torch.randn(bsz, n, seq_len, 17)
    stock_ids = torch.arange(n).unsqueeze(0).expand(bsz, n)
    out = nf(features, stock_ids)
    assert out.shape == (bsz, n)


def test_dual_pathway_forward():
    ae = Autoencoder(input_dim=17, hidden_layers=(64, 32), latent_dim=32)
    normal_nf = NodeTransformer(17, num_nodes=20, num_layers=2, num_heads=4, model_dim=64, ffn_dim=128, context_dim=0)
    event_nf = NodeTransformer(17, num_nodes=20, num_layers=2, num_heads=4, model_dim=64, ffn_dim=128, context_dim=12)
    framework = DualPathwayFramework(ae, normal_nf, event_nf, initial_tau=0.5, initial_alpha=0.5)

    bsz, n, seq_len = 2, 20, 30
    features = torch.randn(bsz, n, seq_len, 17)
    stock_ids = torch.arange(n).unsqueeze(0).expand(bsz, n)
    x_router = torch.randn(bsz * n, 17)
    context = torch.randn(bsz, 12)

    out = framework(features, stock_ids, x_router=features.reshape(bsz * n, seq_len, 17)[:, -1, :], context=context)
    assert "reconstruction_error" in out
    assert "blended_prediction" in out
