import math
import torch


def compute_degree(edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg = deg.index_add(0, src, torch.ones(src.size(0), dtype=torch.long, device=edge_index.device))
    deg = deg.index_add(0, dst, torch.ones(dst.size(0), dtype=torch.long, device=edge_index.device))
    return deg


def compute_clustering_coefficient(edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    # Simple exact local clustering: for small graphs this is fine. Returns float tensor
    src, dst = edge_index
    device = edge_index.device
    nbrs = [set() for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].add(v)
        nbrs[v].add(u)

    clust = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    for i in range(num_nodes):
        neighbors = nbrs[i]
        k = len(neighbors)
        if k < 2:
            clust[i] = 0.0
            continue
        # count existing links between neighbors
        links = 0
        for u in neighbors:
            for v in neighbors:
                if u != v and v in nbrs[u]:
                    links += 1
        # each edge counted twice
        links = links / 2
        possible = k * (k - 1) / 2
        clust[i] = links / possible if possible > 0 else 0.0

    return clust


class TopologySampler:
    """Topology-aware sampler that synthesizes node features and edges.

    This implementation is intentionally simple and vectorized for small graphs
    used in unit tests. It follows the TopGateGNN paper's TAS description.
    """

    def __init__(self, delta: float = 1e-6, sigma: float = 1e-3, knn_k: int = 3,
                 theta: float = 1.0, deg_thr: int = 0, clust_thr: float = 1.0):
        self.delta = delta
        self.sigma = sigma
        self.knn_k = knn_k
        self.theta = theta
        self.deg_thr = deg_thr
        self.clust_thr = clust_thr

    def synthesize(self, x: torch.Tensor, edge_index: torch.LongTensor,
                   y: torch.LongTensor, n_synth: int) -> tuple:
        """Return (x_new, edge_index_new, y_new).

        x: [N, D], edge_index: [2, E], y: [N]
        """
        device = x.device
        N, D = x.size()
        deg = compute_degree(edge_index, N).float()
        clust = compute_clustering_coefficient(edge_index, N)

        # pick minority label (assume binary 0/1)
        unique, counts = torch.unique(y, return_counts=True)
        if unique.numel() == 1:
            minority_label = unique[0].item()
        else:
            minority_label = unique[counts.argmin()].item()

        mask_min = (y == minority_label)
        idx_min = mask_min.nonzero(as_tuple=False).view(-1)
        if idx_min.numel() == 0:
            # nothing to synthesize from
            return x, edge_index, y

        # compute weights w_i = deg/(clust+delta) normalized over minority nodes
        w_raw = deg[idx_min] / (clust[idx_min] + self.delta)
        w = w_raw / (w_raw.sum() + 1e-12)

        # allocate counts
        alloc = torch.floor(w * float(n_synth)).long()
        # ensure sum equals n_synth by distributing remainder
        rem = n_synth - int(alloc.sum().item())
        if rem > 0:
            # distribute one by one to largest weights
            _, order = torch.sort(w, descending=True)
            for i in range(rem):
                alloc[order[i % order.numel()]] += 1

        new_nodes = []
        new_edges = []
        for src_idx, count in zip(idx_min.tolist(), alloc.tolist()):
            if count <= 0:
                continue
            xi = x[src_idx:src_idx+1]  # [1, D]
            # find nearest neighbors in feature space (exclude itself)
            dists = torch.cdist(xi, x).view(-1)
            dists[src_idx] = float('inf')
            _, nn_idx = torch.topk(-dists, k=min(self.knn_k, N-1))
            nn_idx = nn_idx.tolist()
            for _ in range(count):
                j = x[torch.tensor(nn_idx, device=device)[torch.randint(0, len(nn_idx), (1,)).item()]]
                lam = torch.rand(1, device=device)
                eps = torch.randn(1, D, device=device) * self.sigma
                x_new = xi + lam * (j - xi) + eps
                new_nodes.append(x_new.view(D))

                # connect new node to top-K nearest existing nodes
                d_all = torch.cdist(x_new.view(1, D), x).view(-1)
                topk = torch.topk(-d_all, k=min(self.knn_k, N)).indices
                for cand in topk.tolist():
                    if d_all[cand].item() < self.theta and deg[cand].item() > self.deg_thr and clust[cand].item() < self.clust_thr:
                        new_edges.append((cand, N + len(new_nodes) - 1))

        if len(new_nodes) == 0:
            return x, edge_index, y

        x_new_tensor = torch.vstack([x] + [n.unsqueeze(0) for n in new_nodes])
        y_new = torch.cat([y, torch.full((len(new_nodes),), minority_label, dtype=y.dtype, device=device)])

        if edge_index.numel() == 0:
            edge_list = []
        else:
            edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        # add new edges (undirected added both ways)
        for u, v in new_edges:
            edge_list.append((u, v))
            edge_list.append((v, u))

        edge_index_new = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        return x_new_tensor, edge_index_new, y_new
