import torch
import torch.nn as nn
import torch.nn.functional as F


class CumSoftGate(nn.Module):
    """Compute cumulative-softmax gating vectors.

    Given input features of shape [N, F], returns a gating tensor [N, Fout]
    via an MLP followed by softmax + cumsum to ensure monotonic increasing components.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        p = F.softmax(logits, dim=-1)
        cum = torch.cumsum(p, dim=-1)
        return cum
