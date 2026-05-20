import torch
import torch.nn as nn
import torch.nn.functional as F

from .CumSoftGate import CumSoftGate


class TopGateLayer(nn.Module):
    """A single TopGate layer implementing aggregation + heterogeneity gating.

    This implementation is intentionally small and dependency-free (no torch_scatter).
    It computes neighbor means and simple heterogeneity measures using index_add.
    """

    def __init__(self, dim: int, gate_hidden: int = None):
        super().__init__()
        self.dim = dim
        self.gate_hidden = gate_hidden or max(8, dim)
        self.cumsoft = CumSoftGate(in_dim=dim * 2, out_dim=dim)
        self.fuse = nn.Linear(2, 1)  # fuse s_a and s_s into s_f
        self.wf = nn.Parameter(torch.tensor(1.0))

    def aggregate_mean(self, h: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        # edge_index: [2, E] where src -> dst
        src, dst = edge_index
        N = h.size(0)
        device = h.device
        sums = torch.zeros_like(h)
        counts = torch.zeros(N, device=device, dtype=h.dtype)
        sums = sums.index_add(0, dst, h[src])
        counts = counts.index_add(0, dst, torch.ones(src.size(0), device=device, dtype=h.dtype))
        counts = counts.unsqueeze(-1).clamp_min(1.0)
        mean = sums / counts
        return mean, counts.squeeze(-1)

    def neighbor_variance(self, h: torch.Tensor, edge_index: torch.LongTensor, mean: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        N = h.size(0)
        device = h.device
        diffsq = (h[src] - mean[dst]) ** 2
        sumsq = torch.zeros_like(h)
        sumsq = sumsq.index_add(0, dst, diffsq)
        # variance per-node (scalar): mean of squared diffs across feature dims
        var = (sumsq.sum(dim=-1) / ( ( (edge_index.size(1) * 1.0) ) + 1e-12)).clamp_min(0.0)
        # if node has no neighbors, fallback to zero
        return var

    def neighbor_degree_std(self, edge_index: torch.LongTensor, deg: torch.Tensor) -> torch.Tensor:
        # approximate structural heterogeneity as std of neighbor degrees
        src, dst = edge_index
        N = deg.numel()
        device = deg.device
        sum_deg = torch.zeros(N, device=device)
        sumsq = torch.zeros(N, device=device)
        counts = torch.zeros(N, device=device)
        sum_deg = sum_deg.index_add(0, dst, deg[src].float())
        sumsq = sumsq.index_add(0, dst, (deg[src].float() ** 2))
        counts = counts.index_add(0, dst, torch.ones_like(sum_deg[dst]))
        counts = counts.clamp_min(1.0)
        mean = sum_deg / counts
        var = (sumsq / counts) - (mean ** 2)
        std = torch.sqrt(F.relu(var))
        return std

    def forward(self, h: torch.Tensor, edge_index: torch.LongTensor,
                deg: torch.Tensor = None, clust: torch.Tensor = None, prev_g: torch.Tensor = None):
        N, D = h.size()
        device = h.device
        if deg is None:
            deg = torch.zeros(N, device=device)
        if clust is None:
            clust = torch.zeros(N, device=device)
        if prev_g is None:
            prev_g = torch.zeros_like(h)

        m, counts = self.aggregate_mean(h, edge_index)

        # attribute heterogeneity: variance across neighbor features
        s_a = self.neighbor_variance(h, edge_index, m)
        # structural heterogeneity: std of neighbor degrees
        s_s = self.neighbor_degree_std(edge_index, deg)

        s_pair = torch.stack([s_a, s_s], dim=-1)  # [N, 2]
        s_f = torch.sigmoid(self.fuse(s_pair).view(-1))  # [N]

        # gating
        hat_g = self.cumsoft(torch.cat([h, m], dim=-1))  # [N, D]
        g_mono = prev_g + (1 - prev_g) * hat_g
        g = torch.sigmoid(g_mono + self.wf * s_f.unsqueeze(-1))

        h_new = g * h + (1 - g) * m
        return h_new, g


class TopGateStack(nn.Module):
    """Stack multiple TopGateLayer and provide a Graph-like forward signature.

    forward(x, edge_index, clustering_coefficient=None) -> h
    """

    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([TopGateLayer(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, clustering_coefficient: torch.Tensor = None):
        # compute degree from edge_index
        from utils.topology import compute_degree
        N = x.size(0)
        device = x.device
        try:
            deg = compute_degree(edge_index, N)
        except Exception:
            deg = torch.zeros(N, device=device)
        clust = clustering_coefficient if clustering_coefficient is not None else torch.zeros(N, device=device)
        prev_g = None
        h = x
        for layer in self.layers:
            h, g = layer(h, edge_index, deg=deg.float(), clust=clust, prev_g=prev_g)
            prev_g = g
        return h
