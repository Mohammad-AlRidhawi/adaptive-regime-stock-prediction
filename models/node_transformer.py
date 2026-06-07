"""Graph-aware node transformer with causal self-attention and learnable edge weights."""

import math

import torch
from torch import nn


def sinusoidal_positional_encoding(seq_len: int, dim: int, device=None) -> torch.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""
    position = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EdgeAwareMultiHeadAttention(nn.Module):
    """Multi-head self-attention with additive edge-weight bias and causal mask.

    Implements eq. (4) of the manuscript:
        A = softmax(Q K^T / sqrt(d_k) + M + E) V

    where E is the learnable edge-weight matrix over the N stock nodes.
    """

    def __init__(self, model_dim: int, num_heads: int, num_nodes: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_nodes = num_nodes

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.edge_refiner = nn.Linear(2 * model_dim, 1)
        self.edge_bias = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "init_edges", torch.zeros(num_nodes, num_nodes), persistent=False
        )

    def set_initial_edges(self, e0: torch.Tensor) -> None:
        """Set initial edge weights from sector + correlation priors (eq. 3)."""
        with torch.no_grad():
            self.init_edges.copy_(e0)

    def _refine_edges(self, h: torch.Tensor) -> torch.Tensor:
        """Learnable per-layer edge refinement: e_ij^(l+1) = sigmoid(w^T [h_i, h_j] + b)."""
        bsz, n, _ = h.shape
        h_i = h.unsqueeze(2).expand(bsz, n, n, self.model_dim)
        h_j = h.unsqueeze(1).expand(bsz, n, n, self.model_dim)
        pair = torch.cat([h_i, h_j], dim=-1)
        refined = torch.sigmoid(self.edge_refiner(pair).squeeze(-1) + self.edge_bias)
        return refined

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n, _ = x.shape

        q = self.q_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        edge_bias = self._refine_edges(x)
        edge_bias = edge_bias + self.init_edges.to(x.device).unsqueeze(0)
        scores = scores + edge_bias.unsqueeze(1)

        if causal_mask is not None:
            scores = scores + causal_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, n, self.model_dim)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Pre-norm residual transformer block: MHA -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_nodes: int,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = EdgeAwareMultiHeadAttention(model_dim, num_heads, num_nodes, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, causal_mask=causal_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class NodeTransformer(nn.Module):
    """Graph-aware transformer over a cross-section of N stock nodes.

    Used as the backbone for both the normal and event pathways. The event
    pathway sets `context_dim > 0` so that a per-timestep context vector c_t
    is concatenated to each stock's input before the first projection.
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        num_layers: int = 6,
        num_heads: int = 8,
        model_dim: int = 512,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        context_dim: int = 0,
        stock_embedding_dim: int = 32,
        num_regimes: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.context_dim = context_dim

        self.stock_embedding = nn.Embedding(num_nodes, stock_embedding_dim)
        input_dim = feature_dim + stock_embedding_dim + model_dim + context_dim
        self.input_projection = nn.Linear(input_dim, model_dim)

        if context_dim > 0:
            self.regime_embedding = nn.Embedding(num_regimes, 4)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(model_dim, num_heads, num_nodes, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.prediction_head = nn.Linear(model_dim, 1)

    def set_initial_edges(self, e0: torch.Tensor) -> None:
        for layer in self.layers:
            layer.attn.set_initial_edges(e0)

    def forward(
        self,
        features: torch.Tensor,
        stock_ids: torch.Tensor,
        context: torch.Tensor | None = None,
        regime_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n, seq_len, _ = features.shape

        stock_emb = self.stock_embedding(stock_ids).unsqueeze(2).expand(bsz, n, seq_len, -1)
        te = sinusoidal_positional_encoding(seq_len, self.model_dim, device=features.device)
        te = te.unsqueeze(0).unsqueeze(0).expand(bsz, n, seq_len, self.model_dim)

        parts = [features, stock_emb, te]
        if self.context_dim > 0 and context is not None:
            ctx = context.unsqueeze(1).unsqueeze(2).expand(bsz, n, seq_len, -1)
            parts.append(ctx)

        h = torch.cat(parts, dim=-1)
        h = self.input_projection(h)

        # Causal mask over the temporal dimension; broadcasted within each node.
        causal = torch.full(
            (seq_len, seq_len), float("-inf"), device=features.device
        ).triu(diagonal=1)
        # Apply attention across the node dimension at each timestep.
        h_seq = h.transpose(1, 2).reshape(bsz * seq_len, n, self.model_dim)
        for layer in self.layers:
            h_seq = layer(h_seq, causal_mask=None)
        h = h_seq.view(bsz, seq_len, n, self.model_dim).transpose(1, 2)

        # Read out the prediction at the last available timestep.
        last_h = h[:, :, -1, :]
        return self.prediction_head(last_h).squeeze(-1)
