import sys
import pathlib
import torch

# ensure repo root is on sys.path when running under conda run
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.topology import TopologySampler, compute_degree, compute_clustering_coefficient
from models.CumSoftGate import CumSoftGate
from models.TopGateLayer import TopGateLayer


def test_topology_sampler_and_layer():
    # small undirected ring graph
    N = 6
    D = 8
    x = torch.randn(N, D)
    # make ring edges (undirected)
    edges = []
    for i in range(N):
        j = (i + 1) % N
        edges.append((i, j))
        edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # labels with imbalance: one minority (label 1)
    y = torch.zeros(N, dtype=torch.long)
    y[0] = 1

    sampler = TopologySampler(knn_k=2, theta=10.0)
    x_new, edge_index_new, y_new = sampler.synthesize(x, edge_index, y, n_synth=3)

    assert x_new.size(0) >= N
    assert y_new.size(0) == x_new.size(0)
    assert edge_index_new.size(0) == 2

    # compute degree/clustering helpers
    deg = compute_degree(edge_index_new, x_new.size(0)).float()
    clust = compute_clustering_coefficient(edge_index_new, x_new.size(0))

    # CumSoftGate
    cum = CumSoftGate(in_dim=D * 2, out_dim=D)
    h = x_new
    # neighbor mean requires original edge_index (use new one)

    layer = TopGateLayer(dim=D)
    h_out, g = layer(h, edge_index_new, deg=deg, clust=clust)

    assert h_out.shape == h.shape
    assert torch.isfinite(h_out).all()
    assert g.shape == h.shape


if __name__ == '__main__':
    test_topology_sampler_and_layer()
    print('ok')
