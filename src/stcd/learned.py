"""B1 oracle — a from-scratch DTSNN-style dynamic-threshold spiking denoiser.

Pure PyTorch (no SpikingJelly / mamba-ssm), surrogate-gradient trained. Two branches
like DTSNN: an Event-Denoising Branch (EDB, a conv-LIF stack) whose final-layer
threshold is set per-pixel by a Dynamic-Threshold Branch (DTB). Two heads (toggle):

* ``head="theta"``   — DTB predicts a per-pixel THRESHOLD map (DTSNN-native; the
  learned upper bound for B3-gather / Strategy 5).
* ``head="scatter"`` — DTB predicts a per-pixel ADDITIVE contribution added to the
  input before the spatial conv (the learned upper bound for B3-scatter).

Input is a binned frame stack ``[B, 2, H, W, T]`` (ON/OFF counts per time bin);
output is a per-(pixel,bin) logit ``[B, 1, H, W, T]`` — an event's score is the logit
at its (y, x, bin) cell. This matches the rest of the harness (per-event AUC/DA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import spike


class DynThreshSNN(nn.Module):
    def __init__(self, C: int = 16, k: int = 3, tau_steps: float = 2.0,
                 head: str = "theta", beta: float = 5.0):
        super().__init__()
        if head not in {"theta", "scatter"}:
            raise ValueError(head)
        p = k // 2
        # Event-Denoising Branch (conv-LIF stack over time)
        self.edb1 = nn.Conv2d(2, C, k, padding=p)
        self.edb2 = nn.Conv2d(C, C, k, padding=p)
        self.edb_out = nn.Conv2d(C, 1, k, padding=p)
        # Dynamic-Threshold Branch (per-pixel map from the aggregated context)
        self.dtb1 = nn.Conv2d(2, C, k, padding=p)
        self.dtb_out = nn.Conv2d(C, 1, k, padding=p)
        self.alpha = float(torch.exp(torch.tensor(-1.0 / tau_steps)))  # per-step leak
        self.head = head
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,2,H,W,T]
        B, P, H, W, T = x.shape
        ctx = x.sum(-1)                                   # [B,2,H,W] temporal context
        d = F.relu(self.dtb1(ctx))
        dmap = self.dtb_out(d)                            # [B,1,H,W]
        a = self.alpha
        v1 = v2 = vo = 0.0
        out = []
        for t in range(T):
            xt = x[..., t]                               # [B,2,H,W]
            if self.head == "scatter":
                xt = xt + dmap                           # learned additive contribution
            v1 = a * v1 + self.edb1(xt); s1 = spike(v1 - 1.0, self.beta); v1 = v1 - s1
            v2 = a * v2 + self.edb2(s1); s2 = spike(v2 - 1.0, self.beta); v2 = v2 - s2
            vo = a * vo + self.edb_out(s2)               # final membrane (no reset)
            th = F.softplus(dmap) if self.head == "theta" else torch.zeros_like(dmap)
            out.append(vo - th)                          # per-cell logit
        return torch.stack(out, dim=-1)                  # [B,1,H,W,T]

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0]                     # [B,H,W,T]


class SpatialDenoiser(nn.Module):
    """B2 oracle — a learned long-range SPATIAL recombiner (dilated conv stack),
    a pure-PyTorch stand-in for EDmamba's S-SSM. Replaces STCD's fixed k×k box-sum
    with a *learned* spatial mixing of much larger receptive field; per-bin 2D convs
    (shared over time) with a global temporal-context input. Same I/O contract as
    DynThreshSNN: ``[B,2,H,W,T]`` → per-(pixel,bin) logit ``[B,1,H,W,T]``."""

    def __init__(self, C: int = 24, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv2d(4, C, 3, padding=1)         # per-bin frame + temporal ctx
        self.layers = nn.ModuleList(
            [nn.Conv2d(C, C, 3, padding=d, dilation=d) for d in dilations])  # long range
        self.out = nn.Conv2d(C, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,2,H,W,T]
        ctx = x.sum(-1)                                   # [B,2,H,W] temporal context
        out = []
        for t in range(x.shape[-1]):
            h = F.relu(self.inp(torch.cat([x[..., t], ctx], dim=1)))
            for layer in self.layers:
                h = F.relu(layer(h) + h)                  # residual dilated conv
            out.append(self.out(h))                       # [B,1,H,W]
        return torch.stack(out, dim=-1)                   # [B,1,H,W,T]

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0]
