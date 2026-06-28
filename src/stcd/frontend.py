"""The tunable spiking front-end — the contribution (proposal stages 2-4).

All stages operate on a dense tensor ``[P, H, W, T]`` and are differentiable, so
the parameters (pooling kernel, leak time-constant τ, firing threshold θ) can be
jointly optimised by surrogate-gradient descent rather than hand-tuned.

  Stage 2  ``SpatialPool``   aggregate local neighbourhoods (ON/OFF kept separate)
  Stage 3  ``TemporalLeak``  capacitive leak  v[t] = α·v[t-1] + x[t],  α=exp(-Δt/τ)
  Stage 4  ``LIFCoincidence`` leak + threshold + reset spiking neuron (surrogate grad)
  Stage 5  ``AEROut``         spikes → event stream

``SpikingFrontEnd`` wires them together and, crucially, exposes a per-event
*support score* (the membrane potential at each event's pooled cell) so we can
draw a parameter-free ROC and make per-event keep/drop decisions comparable to
the BAF baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import Events, TimeGrid, events_to_tensor, tensor_to_events

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Surrogate-gradient spike nonlinearity
# --------------------------------------------------------------------------- #
class _SpikeFn(torch.autograd.Function):
    """Heaviside in the forward pass; SuperSpike (fast-sigmoid) surrogate in the
    backward pass so the threshold is trainable. Input ``u`` is ``(v - θ)``."""

    @staticmethod
    def forward(ctx, u: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(u)
        ctx.beta = beta
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (u,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + ctx.beta * u.abs()) ** 2
        return grad_out * surrogate, None


def spike(u: Tensor, beta: float = 10.0) -> Tensor:
    return _SpikeFn.apply(u, beta)


def _pad_to_multiple(x: Tensor, pool: int) -> Tensor:
    """Zero-pad H and W (last two of [N,C,H,W]) up to a multiple of ``pool`` so
    that non-overlapping pooling makes ``out = x // pool`` an exact cell map."""
    if pool <= 1:
        return x
    H, W = x.shape[-2:]
    ph = (pool - H % pool) % pool
    pw = (pool - W % pool) % pool
    return F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x


# --------------------------------------------------------------------------- #
# Stage 2 — spatial combination
# --------------------------------------------------------------------------- #
class SpatialPool(nn.Module):
    """Spatial combination (Stage 2), in two optional sub-steps, ON/OFF separate:

    1. **Neighbourhood aggregation** (``neighbor_k`` > 1): a stride-1 depthwise box
       sum that gives every cell the support of its ``k×k`` neighbourhood while
       *preserving resolution*. This is the spatial-coincidence substrate — the
       direct analogue of BAF's neighbour check — and is what lets the front-end
       compute per-event keep/drop at full resolution.
    2. **Downsampling pool** (``pool`` > 1): non-overlapping ``pool×pool`` reduction
       (the proposal's 2×2 = 4× downsample). ``out_x = x // pool`` exactly, so each
       input pixel/event still maps to one output cell.

    ``mode`` selects the downsample combination function (``or`` / ``sum`` /
    ``learned``). With ``neighbor_k=1`` and ``pool=1`` this is a no-op.
    """

    def __init__(self, pool: int = 1, mode: str = "sum", neighbor_k: int = 1):
        super().__init__()
        if mode not in {"or", "sum", "learned"}:
            raise ValueError(f"unknown pool mode {mode!r}")
        if neighbor_k > 1 and neighbor_k % 2 == 0:
            raise ValueError("neighbor_k must be odd to preserve resolution")
        self.pool = pool
        self.mode = mode
        self.neighbor_k = neighbor_k
        if mode == "learned":
            # depthwise (per-polarity) learnable kernel, non-negative via softplus
            self.raw_w = nn.Parameter(torch.zeros(2, 1, max(pool, 1), max(pool, 1)))

    def forward(self, x: Tensor) -> Tensor:  # x: [P,H,W,T]
        P, H, W, T = x.shape
        if self.neighbor_k <= 1 and self.pool <= 1 and self.mode != "learned":
            return x
        z = x.permute(3, 0, 1, 2).contiguous()       # [T, P, H, W]

        if self.neighbor_k > 1:                        # stride-1 box sum (keeps res)
            k = self.neighbor_k
            w = torch.ones(P, 1, k, k, dtype=z.dtype, device=z.device)
            z = F.conv2d(z, w, padding=k // 2, groups=P)

        if self.pool > 1 or self.mode == "learned":    # optional downsample
            z = _pad_to_multiple(z, self.pool)
            if self.mode == "or":
                z = F.max_pool2d(z, self.pool)
            elif self.mode == "sum":
                z = F.avg_pool2d(z, self.pool) * (self.pool * self.pool)
            else:  # learned depthwise conv, stride = kernel = pool
                w = F.softplus(self.raw_w)
                z = F.conv2d(z, w, stride=self.pool, groups=P)
        return z.permute(1, 2, 3, 0).contiguous()      # [P,H',W',T]


# --------------------------------------------------------------------------- #
# Stage 3 — capacitive temporal extension (leak, no reset)
# --------------------------------------------------------------------------- #
class TemporalLeak(nn.Module):
    """Leaky integration along time: ``v[t] = α·v[t-1] + x[t]``, ``α=exp(-Δt/τ)``.

    Widens each spike into a short window so spikes close in time sum — the
    coincidence substrate. ``τ`` is learnable (stored as ``log τ``). Stateless
    w.r.t. firing (no reset); used as an ablation and for visualising membranes.
    """

    def __init__(self, tau: float = 5e-3):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def alpha(self, dt: float) -> Tensor:
        return torch.exp(-dt / self.tau)

    def forward(self, x: Tensor, dt: float) -> Tensor:  # [P,H,W,T]
        a = self.alpha(dt)
        out = torch.empty_like(x)
        v = torch.zeros_like(x[..., 0])
        for t in range(x.shape[-1]):
            v = a * v + x[..., t]
            out[..., t] = v
        return out


# --------------------------------------------------------------------------- #
# Stage 4 — coincidence / threshold (LIF with reset)
# --------------------------------------------------------------------------- #
class LIFCoincidence(nn.Module):
    """Leaky integrate-and-fire neuron: combines the Stage-3 leak (τ) with the
    Stage-4 threshold (θ) and reset-by-subtraction. Fires only when accumulated,
    *coincident* support crosses θ within the leak window.

    Optionally prepends a small learnable causal temporal kernel (the
    sequence-detecting variant) to favour temporally coherent motion over
    isolated flicker.

    Returns spikes ``s`` and the pre-threshold membrane ``v`` (the support score).
    """

    def __init__(
        self,
        tau: float = 5e-3,
        theta: float = 1.0,
        beta: float = 10.0,
        seq_kernel: int = 0,
        adapt_gain: float = 0.0,
        tau_a: float = 20e-3,
    ):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.raw_theta = nn.Parameter(torch.tensor(_inv_softplus(theta)))
        self.beta = beta
        self.seq_kernel = seq_kernel
        # Spike-frequency adaptation: each spike raises this cell's effective
        # threshold by ``adapt_gain``×(adaptation state); the state leaks with τ_a.
        # Suppresses repetitively-firing cells (hot pixels, flicker, static bursts)
        # while passing transient edges. adapt_gain=0 disables it (plain LIF).
        self.adapt_gain = adapt_gain
        self.tau_a = tau_a
        if seq_kernel > 0:
            w = torch.zeros(2, 1, 1, 1, seq_kernel)
            w[..., -1] = 1.0  # start as identity on current bin
            self.seq_w = nn.Parameter(w)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    @property
    def theta(self) -> Tensor:
        return F.softplus(self.raw_theta)

    def forward(self, x: Tensor, dt: float,
                bias: Tensor | None = None) -> tuple[Tensor, Tensor]:  # [P,H,W,T]
        """``bias`` (optional, broadcastable to ``x``) is an external per-cell offset
        added to the membrane *at the decision* (not to the integrated state): the
        neuron fires iff ``v + bias ≥ θ``. Used by B3's gather/threshold-side toggle
        to inject the per-pixel anti-Hebbian weight ``a``."""
        if self.seq_kernel > 0:
            x = self._apply_seq_kernel(x)
        a = torch.exp(-dt / self.tau)
        theta0 = self.theta
        g = self.adapt_gain
        beta_a = math.exp(-dt / self.tau_a)
        spikes = torch.empty_like(x)
        score = torch.empty_like(x)
        v = torch.zeros_like(x[..., 0])
        adapt = torch.zeros_like(x[..., 0])
        for t in range(x.shape[-1]):
            v = a * v + x[..., t]
            b = 0.0 if bias is None else bias[..., t]
            theta_eff = theta0 + g * adapt
            # adaptation-discounted support: repetitively-firing cells score low
            score[..., t] = (v + b) - g * adapt
            s = spike((v + b) - theta_eff, self.beta)
            spikes[..., t] = s
            v = v - s * theta_eff            # reset by (adaptive) threshold (state only)
            adapt = beta_a * adapt + s        # spike-frequency adaptation
        return spikes, score

    def _apply_seq_kernel(self, x: Tensor) -> Tensor:
        # causal depthwise conv along T, per polarity
        P, H, W, T = x.shape
        z = x.reshape(P, 1, H * W, T)
        k = self.seq_kernel
        z = F.pad(z, (k - 1, 0))
        w = self.seq_w.reshape(P, 1, 1, k)
        z = F.conv2d(z, w, groups=P)
        return z.reshape(P, H, W, T)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y > 0 else -10.0


# --------------------------------------------------------------------------- #
# Stage 5 — AER output
# --------------------------------------------------------------------------- #
class AEROut:
    """Convert a spike tensor back into an event stream (Address-Event Rep.)."""

    @staticmethod
    def to_events(spikes: Tensor, grid: TimeGrid, H: int, W: int) -> Events:
        return tensor_to_events(spikes, grid, H, W, threshold=0.5)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@dataclass
class FrontEndConfig:
    neighbor_k: int = 3       # stride-1 spatial support window (odd; 1 disables)
    pool: int = 1             # downsample factor (1 = keep full resolution)
    pool_mode: str = "sum"
    tau: float = 8e-3         # leak time-constant (s)
    theta: float = 1.5        # firing threshold (>1 ⇒ requires genuine coincidence)
    beta: float = 10.0        # surrogate-gradient steepness
    seq_kernel: int = 0       # >0 enables the sequence-detecting variant
    adapt_gain: float = 0.0   # spike-frequency adaptation strength (0 = plain LIF)
    tau_a: float = 20e-3      # adaptation leak time-constant (s)
    dt: float = 5e-3          # time-bin width used to build tensors
    # --- B3: per-pixel anti-Hebbian contribution weight `a` (Idea-1 1a/1b) ----- #
    # A per-pixel scalar that rises when neighbours out-fire a pixel and falls when
    # the pixel out-fires its neighbourhood (da/dt = δ(r_neigh − r_self)); a chronic
    # self-firer (hot pixel) sinks `a`. Three first-class toggles:
    b3: bool = False              # enable B3
    b3_side: str = "scatter"      # "scatter": firing pixel deposits `event + a` into
                                  #   neighbours' support; "gather": add `a` at the
                                  #   receiving decision (fire iff V + a ≥ θ)
    b3_update: str = "raw"        # drive the `a` update from "raw" camera firings or
                                  #   from "output" (forwarded/kept) spikes
    b3_clamp_nonneg: bool = False # True: clamp a ≥ 0 (pure relative discount);
                                  #   False: allow a < 0 (active inhibition of nbrs)
    b3_delta: float = 0.5         # anti-Hebbian step size δ
    b3_tau: float = 50e-3         # leak of `a` back toward 0 (s) — reuses the τ machinery
    b3_a_min: float = -4.0        # clamp floor when a < 0 allowed
    b3_a_max: float = 4.0         # clamp ceiling


class SpikingFrontEnd(nn.Module):
    """Stages 2-4 wired together. Operates on tensors and on raw event streams.

    Key methods:
      ``forward(tensor, dt)``           -> (spikes, membrane) tensors
      ``score_events(events, grid)``    -> per-event support score (for ROC)
      ``filter(events)``                -> (kept_mask, filtered_events)
    """

    def __init__(self, cfg: FrontEndConfig | None = None):
        super().__init__()
        self.cfg = cfg or FrontEndConfig()
        self.spatial = SpatialPool(self.cfg.pool, self.cfg.pool_mode, self.cfg.neighbor_k)
        self.lif = LIFCoincidence(
            tau=self.cfg.tau,
            theta=self.cfg.theta,
            beta=self.cfg.beta,
            seq_kernel=self.cfg.seq_kernel,
            adapt_gain=self.cfg.adapt_gain,
            tau_a=self.cfg.tau_a,
        )

    def forward(self, tensor: Tensor, dt: float | None = None) -> tuple[Tensor, Tensor]:
        dt = self.cfg.dt if dt is None else dt
        if not self.cfg.b3:
            pooled = self.spatial(tensor)
            return self.lif(pooled, dt)
        return self._forward_b3(tensor, dt)

    # -- B3: per-pixel anti-Hebbian contribution weight ---------------------- #
    def _b3_a_trajectory(self, firing: Tensor, dt: float) -> Tensor:
        """Causal per-pixel weight ``a`` over time from a per-pixel/bin ``firing``
        count tensor ``[H,W,T]``. Returns ``A`` ``[H,W,T]`` where ``A[...,t]`` is the
        value of ``a`` *before* bin ``t`` is integrated (so it's causal)."""
        H, W, T = firing.shape
        k = self.cfg.neighbor_k
        z = firing.permute(2, 0, 1).unsqueeze(1)            # [T,1,H,W]
        w = torch.ones(1, 1, k, k, dtype=z.dtype, device=z.device)
        box = F.conv2d(z, w, padding=k // 2).squeeze(1)     # [T,H,W] k×k box (incl centre)
        self_f = firing.permute(2, 0, 1)                     # [T,H,W]
        neigh_f = box - self_f                               # centre-excluded neighbour count
        denom = max(k * k - 1, 1)
        delta = self.cfg.b3_delta
        beta = math.exp(-dt / self.cfg.b3_tau)
        amin = 0.0 if self.cfg.b3_clamp_nonneg else self.cfg.b3_a_min
        amax = self.cfg.b3_a_max
        a = torch.zeros(H, W, dtype=firing.dtype, device=firing.device)
        A = torch.empty(T, H, W, dtype=firing.dtype, device=firing.device)
        for t in range(T):
            A[t] = a                                         # causal (pre-update) value
            a = beta * a + delta * (neigh_f[t] / denom - self_f[t])
            a = a.clamp(amin, amax)
        return A.permute(1, 2, 0).contiguous()              # [H,W,T]

    def _forward_b3(self, tensor: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        if self.cfg.pool != 1:
            raise ValueError("B3 requires pool=1 (per-pixel registers)")
        # Firing signal driving the `a` update.
        if self.cfg.b3_update == "output":
            base_spikes, _ = self.lif(self.spatial(tensor), dt)
            firing = base_spikes.sum(0)                      # [H,W,T] forwarded spikes
        elif self.cfg.b3_update == "raw":
            firing = tensor.sum(0)                           # [H,W,T] raw event counts
        else:
            raise ValueError(f"unknown b3_update {self.cfg.b3_update!r}")
        A = self._b3_a_trajectory(firing, dt)               # [H,W,T]
        if self.cfg.b3_side == "scatter":
            fired = (tensor.sum(0) > 0).to(tensor.dtype)     # [H,W,T] any-polarity firing
            x_mod = tensor + (A * fired).unsqueeze(0)        # deposit `event + a`
            return self.lif(self.spatial(x_mod), dt)
        elif self.cfg.b3_side == "gather":
            return self.lif(self.spatial(tensor), dt, bias=A.unsqueeze(0))
        else:
            raise ValueError(f"unknown b3_side {self.cfg.b3_side!r}")

    # -- per-event interface ------------------------------------------------- #
    def _event_cells(self, ev: Events, grid: TimeGrid) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pool = max(self.cfg.pool, 1)
        p = torch.from_numpy(ev.ps).long()
        oy = torch.from_numpy(ev.ys // pool).long()
        ox = torch.from_numpy(ev.xs // pool).long()
        tb = torch.from_numpy(grid.bin_index(ev.ts)).long()
        return p, oy, ox, tb

    def event_membrane(self, tensor: Tensor, grid: TimeGrid, ev: Events) -> Tensor:
        """Differentiable per-event membrane potential (for surrogate-gradient
        training). Returns a 1-D tensor aligned to ``ev``, with grad to τ/θ/weights."""
        _, membrane = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._event_cells(ev, grid)
        return membrane[p, oy, ox, tb]

    @torch.no_grad()
    def score_events(self, ev: Events, grid: TimeGrid | None = None) -> np.ndarray:
        """Per-event support score = membrane potential at the event's pooled
        cell/bin. Higher ⇒ more neighbour/temporal support ⇒ more likely signal."""
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        _, membrane = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._event_cells(ev, grid)
        return membrane[p, oy, ox, tb].cpu().numpy()

    @torch.no_grad()
    def filter(self, ev: Events, grid: TimeGrid | None = None) -> tuple[np.ndarray, Events]:
        """Run the full front-end and decide keep/drop for each *original* event:
        an event is kept iff its pooled cell emits a spike in its time-bin."""
        tensor, grid = events_to_tensor(ev, grid=grid, dt=self.cfg.dt)
        spikes, _ = self.forward(tensor, grid.dt)
        p, oy, ox, tb = self._event_cells(ev, grid)
        kept = (spikes[p, oy, ox, tb] > 0.5).cpu().numpy()
        return kept, ev.select(kept)
