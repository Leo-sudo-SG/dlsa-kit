"""
Graph-Arb: Spatiotemporal Graph Transformer for portfolio signal extraction.

This module provides a PyTorch Module compatible with the repository's
model conventions. It implements:
- Temporal encoder (LSTM) per asset over the lookback window.
- Graph Transformer layers (PyTorch Geometric TransformerConv) that use
  edge attributes (e.g. pairwise correlation + edge-type encodings).
- MLP head producing scalar weight per asset, normalized to sum |w| = 1.

Notes:
- This file requires `torch` and `torch_geometric` to be installed. If
  `torch_geometric` is not available the import will raise with a helpful
  message.
- Forward signature: `forward(self, x, edge_index=None, edge_attr=None)`
  where `x` can be a tensor of shape `(N, L)` or `(N, L, C)`. If `edge_index`
  is omitted the model can build a statistical graph from `x` using
  correlation-based top-k neighbor selection (costly for very large N).

This implementation is intentionally minimal and focused on clarity and
compatibility with the rest of the repo. It can be extended with richer
edge features (sector, supply-chain, dynamic regime indicators) and
scalable graph construction (approximate k-NN) as needed.
"""

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    # Import PyG layers used here
    from torch_geometric.nn import TransformerConv
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
    _PYG_AVAILABLE = True
except Exception:
    # Provide a lightweight fallback implementation of a graph convolution
    # that uses dense aggregation. This allows running the example without
    # installing PyTorch Geometric (useful for small-scale testing).
    _PYG_AVAILABLE = False

    def to_undirected(edge_index):
        if edge_index.numel() == 0:
            return edge_index
        rows, cols = edge_index
        rev = torch.stack([cols, rows], dim=0)
        return torch.cat([edge_index, rev], dim=1)

    class TransformerConv(nn.Module):
        """Fallback TransformerConv-like layer using dense aggregation.

        It computes a normalized adjacency matrix from `edge_index` and
        `edge_attr` and performs a single linear transform followed by
        neighborhood aggregation: out = A_norm @ (x W).
        """

        def __init__(self, in_channels, out_channels, heads=1, concat=False, edge_dim=None, dropout=0.0):
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)
            self.dropout = dropout

        def forward(self, x, edge_index, edge_attr=None):
            # x: (N, F)
            N = x.size(0)
            device = x.device
            if edge_index.numel() == 0:
                return self.lin(x)
            rows = edge_index[0].to(device)
            cols = edge_index[1].to(device)
            vals = None
            if edge_attr is not None and edge_attr.numel() > 0:
                vals = edge_attr.squeeze(-1).to(device)
            else:
                vals = torch.ones(rows.size(0), device=device)

            # build dense adjacency (N x N) - OK for small N used in examples
            A = torch.zeros((N, N), device=device)
            A[rows, cols] = vals
            # row-normalize
            s = A.sum(dim=1, keepdim=True)
            s[s == 0] = 1.0
            A = A / s
            h = self.lin(x)
            out = A @ h
            return out


class GraphArb(nn.Module):
    """Spatio-temporal Graph Transformer.

    Conforms to model constructor conventions used in this repo:
    def __init__(self, logdir, random_seed=0, lookback=30, device='cpu', **hyperparams)

    Example hyperparams supported:
    - d_model (int): latent embedding dimension
    - lstm_hidden (int): hidden dim for temporal LSTM
    - gnn_layers (int): number of TransformerConv layers
    - edge_dim (int): width of edge attribute vectors
    - topk (int): when building statistical graph, keep top-k neighbors per node
    - dropout (float)
    """

    def __init__(
        self,
        logdir: str,
        random_seed: int = 0,
        lookback: int = 30,
        device: str = "cpu",
        d_model: int = 64,
        lstm_hidden: int = 64,
        gnn_layers: int = 2,
        edge_dim: int = 2,
        topk: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        self.logdir = logdir
        self.device = device
        self.random_seed = random_seed
        # model training flags expected by train_test.py
        self.is_trainable = True
        self.lookback = lookback

        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.gnn_layers = gnn_layers
        self.edge_dim = edge_dim
        self.topk = topk
        self.dropout = dropout

        # Input projection (handles scalar or multi-channel signals per time-step)
        # We'll lazily initialize LSTM input size on first forward if needed.
        self.input_proj = None
        self.lstm = None

        # Graph transformer layers
        self.gnn_layers_list = nn.ModuleList()
        for i in range(self.gnn_layers):
            self.gnn_layers_list.append(
                TransformerConv(
                    in_channels=self.d_model if i > 0 else self.lstm_hidden,
                    out_channels=self.d_model,
                    heads=1,
                    concat=False,
                    edge_dim=self.edge_dim,
                    dropout=self.dropout,
                )
            )

        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.gnn_layers)])

        # Output MLP -> scalar
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
        )

    def _lazy_init_lstm(self, in_features: int):
        """Initialize input projection and LSTM based on observed input features."""
        if self.input_proj is None:
            self.input_proj = nn.Linear(in_features, self.lstm_hidden)
            # LSTM: batch_first expects (batch=N, seq=lookback, feat)
            self.lstm = nn.LSTM(
                input_size=self.lstm_hidden, hidden_size=self.lstm_hidden, batch_first=True
            )

    @staticmethod
    def build_statistical_graph_from_x(x: torch.Tensor, topk: int = 16):
        """Build a simple correlation-based k-NN graph from input `x`.

        x: Tensor (N, L) or (N, L, C) -> aggregated to last residual channel for correlation
        Returns: edge_index (2, E), edge_attr (E, 1) where edge_attr is correlation value.

        Note: This is O(N^2) and not suitable for very large N (>~2000) without approximations.
        """
        if x.dim() == 3:
            # use first channel or mean across channels as the signal for correlation
            sig = x[:, :, 0]
        else:
            sig = x
        # Center and normalize
        sig = sig - sig.mean(dim=1, keepdim=True)
        denom = sig.norm(dim=1, keepdim=True)
        denom[denom == 0] = 1.0
        sign = sig / denom

        # Compute pairwise correlations via matrix multiply: (N, L) @ (L, N) -> (N, N)
        corr = torch.matmul(sign, sign.t())
        # Clamp numerical noise
        corr = torch.clamp(corr, -1.0, 1.0)

        N = corr.size(0)
        rows = []
        cols = []
        vals = []
        # For each node select top-k neighbors (excluding self)
        k = min(topk, N - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        topk_vals, topk_idx = torch.topk(corr.abs(), k=k + 1, dim=1)
        # topk includes self at index 0 (value 1), so ignore it
        for i in range(N):
            neigh_idx = topk_idx[i].tolist()
            neigh_vals = corr[i, neigh_idx]
            # remove self if present
            filtered = [(j, float(v)) for j, v in zip(neigh_idx, neigh_vals) if j != i]
            # keep up to k
            for j, v in filtered[:k]:
                rows.append(i)
                cols.append(j)
                vals.append(v)

        if len(rows) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(vals, dtype=torch.float).unsqueeze(1)
        # When PyG is available we prefer to keep directed edges (no to_undirected),
        # because PyG convolution layers accept directed edge lists. When PyG is not
        # available we convert to undirected and duplicate edge_attr so the dense
        # fallback can use a symmetric adjacency.
        if not _PYG_AVAILABLE:
            edge_index = to_undirected(edge_index)
            if edge_attr.size(0) * 2 == edge_index.size(1):
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.LongTensor] = None, edge_attr: Optional[torch.Tensor] = None):
        """Forward pass.

        x: (N, L) or (N, L, C)
        If `edge_index` is None the model will build a correlation k-NN graph from `x`.
        Returns: weights (N,) tensor (on same device as input)
        """
        device = x.device if isinstance(x, torch.Tensor) else torch.device(self.device)
        x = x.to(device)

        # Accept (N, L) or (N, L, C)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (N, L, 1)

        N, L, C = x.shape
        self._lazy_init_lstm(C)

        # Temporal encoder: project per-time features then LSTM
        x_proj = self.input_proj(x.view(-1, C)).view(N, L, -1)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        # Use last hidden state per node as its embedding
        h_last = h_n[-1]  # (N, lstm_hidden)

        # Map LSTM hidden to GNN input dimension if needed
        if h_last.shape[1] != self.d_model:
            h = h_last if h_last.shape[1] == self.d_model else F.relu(nn.Linear(h_last.shape[1], self.d_model).to(device)(h_last))
        else:
            h = h_last

        # Build graph if not provided
        if edge_index is None:
            edge_index, edge_attr = self.build_statistical_graph_from_x(x.detach().cpu(), topk=self.topk)
            # move to device
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)

        # Pass through GNN layers
        h_in = h
        for i, gnn in enumerate(self.gnn_layers_list):
            # TransformerConv expects (x, edge_index, edge_attr)
            h_new = gnn(h_in, edge_index, edge_attr)
            # Residual + norm
            if h_new.shape[1] == self.norms[i].normalized_shape[0]:
                h_in = self.norms[i](h_in + F.dropout(h_new, p=self.dropout, training=self.training))
            else:
                # Project then add
                proj = nn.Linear(h_new.shape[1], self.d_model).to(device)(h_new)
                h_in = self.norms[i](h_in + F.dropout(proj, p=self.dropout, training=self.training))

        H_final = h_in

        # Output scalar per node
        w_raw = self.mlp(H_final).squeeze(-1)

        # Normalize to gross leverage 1: sum |w| = 1; if all zeros, fallback to uniform
        denom = w_raw.abs().sum()
        if denom.item() == 0:
            w = torch.ones_like(w_raw) / float(w_raw.numel())
        else:
            w = w_raw / denom

        return w


def example_build_edges_from_residuals_numpy(residuals: "np.ndarray", topk: int = 16):
    """Helper: build edges from a numpy residuals array (T, N) -> returns edge_index, edge_attr tensors.

    WARNING: this is O(N^2) memory/time; use approximate k-NN for large universes.
    """
    import numpy as np
    import torch

    # Use last `lookback` window slice if residuals are longer; assume residuals shape (T, N)
    if residuals.ndim != 2:
        raise ValueError("expected residuals shape (T, N)")
    # transpose to (N, T)
    sig = residuals.T.astype(float)
    # center and normalize each series
    sig = sig - sig.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(sig, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    sign = sig / denom
    corr = sign @ sign.T
    corr = np.clip(corr, -1.0, 1.0)

    N = corr.shape[0]
    rows = []
    cols = []
    vals = []
    k = min(topk, N - 1)
    for i in range(N):
        idxs = np.argsort(-np.abs(corr[i]))[1 : k + 1]
        for j in idxs:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(corr[i, j]))

    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    edge_attr = torch.tensor(vals, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr
