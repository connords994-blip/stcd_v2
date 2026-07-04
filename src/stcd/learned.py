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

Also here: ``FeedbackDenoiser`` (Idea 2) — a FROZEN single-neuron STCD core (fixed
k×k box-sum + LIF θ) with a learned, STREAMING-CAUSAL feedback module that taps the
denoiser's own forwarded spikes at bin ``t`` and injects a facilitation bias into the
threshold neurons at bin ``t+1`` (``backend="snn"`` conv-recurrent, or ``backend="ssn"``
a pure-PyTorch spatial selective-scan SSM over a Z-order curve, the EDmamba flavour).
"""
from __future__ import annotations

import numpy as np
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


class CombinedDenoiser(nn.Module):
    """B1+B2 combined ceiling (cost ignored) — the max reach of the STCD structure with
    BOTH learned mechanisms: a B2 dilated-conv long-range **spatial-recombination** trunk,
    then B1's **leaky temporal integration** gated by a learned per-pixel **dynamic
    threshold** (DTB). Same I/O contract: ``[B,2,H,W,T]`` → logit ``[B,1,H,W,T]``."""

    def __init__(self, C: int = 24, dilations=(1, 2, 4, 8), tau_steps: float = 2.0):
        super().__init__()
        self.inp = nn.Conv2d(4, C, 3, padding=1)
        self.spatial = nn.ModuleList(
            [nn.Conv2d(C, C, 3, padding=d, dilation=d) for d in dilations])  # B2 trunk
        self.dtb = nn.Conv2d(C, 1, 3, padding=1)        # B1 dynamic-threshold head
        self.edb_out = nn.Conv2d(C, 1, 3, padding=1)    # B1 readout
        self.alpha = float(torch.exp(torch.tensor(-1.0 / tau_steps)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,2,H,W,T]
        ctx = x.sum(-1)
        feats = []
        for t in range(x.shape[-1]):
            h = F.relu(self.inp(torch.cat([x[..., t], ctx], dim=1)))
            for layer in self.spatial:
                h = F.relu(layer(h) + h)
            feats.append(h)                              # [B,C,H,W] spatial features
        dmap = F.softplus(self.dtb(torch.stack(feats, 0).mean(0)))   # [B,1,H,W] dyn threshold
        vo = 0.0
        out = []
        for h in feats:
            vo = self.alpha * vo + self.edb_out(h)       # leaky temporal integration
            out.append(vo - dmap)                        # gated by dynamic threshold
        return torch.stack(out, dim=-1)

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0]


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


# --------------------------------------------------------------------------- #
# Idea 2 — streaming-causal output→threshold feedback denoiser
# --------------------------------------------------------------------------- #
def assoc_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive first-order linear-recurrence scan over dim=1:
    ``h_l = a_l * h_{l-1} + b_l`` with ``h_{-1}=0``. Hillis–Steele, O(log L) GPU
    ops (no Python per-token loop). ``a, b`` broadcast on trailing channel dims."""
    L = a.shape[1]
    npad = (a.dim() - 2) * 2                              # pad spec for trailing dims
    h = b
    a_acc = a
    shift = 1
    while shift < L:
        pad = (0,) * npad + (shift, 0)
        a_sh = F.pad(a_acc, pad, value=1.0)[:, :L]        # identity in the front gap
        h_sh = F.pad(h, pad, value=0.0)[:, :L]            # zero in the front gap
        h = h + a_acc * h_sh
        a_acc = a_acc * a_sh
        shift *= 2
    return h


_MORTON_CACHE: dict = {}


def morton_order(H: int, W: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-order (Morton) scan permutation for an H×W grid so 2D neighbours stay close
    in the 1D SSM scan. Returns (order, inv): ``seq = flat[:, order]`` then restore
    with ``flat = seq[:, inv]``. Cached per (H, W)."""
    key = (H, W)
    cached = _MORTON_CACHE.get(key)
    if cached is None:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ys = ys.ravel().astype(np.uint64); xs = xs.ravel().astype(np.uint64)

        def part1by1(n):
            n = n & np.uint64(0xFFFFFFFF)
            n = (n | (n << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
            n = (n | (n << np.uint64(8)))  & np.uint64(0x00FF00FF00FF00FF)
            n = (n | (n << np.uint64(4)))  & np.uint64(0x0F0F0F0F0F0F0F0F)
            n = (n | (n << np.uint64(2)))  & np.uint64(0x3333333333333333)
            n = (n | (n << np.uint64(1)))  & np.uint64(0x5555555555555555)
            return n

        code = part1by1(xs) | (part1by1(ys) << np.uint64(1))
        order = np.argsort(code, kind="stable")
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        cached = (torch.from_numpy(order).long(), torch.from_numpy(inv).long())
        _MORTON_CACHE[key] = cached
    order, inv = cached
    return order.to(device), inv.to(device)


class SpatialSSM(nn.Module):
    """Pure-PyTorch selective state-space scan (Mamba-S6 style) over a 1D token
    sequence: input-dependent step ``Δ``, gate ``B``/readout ``C``, learned diagonal
    ``A``. Bidirectional. The parallel ``assoc_scan`` keeps it GPU-friendly (no
    custom CUDA / ``mamba-ssm`` wheel needed). This is the EDmamba 'Spatial State
    Network' recombiner used to predict facilitation from the forwarded-spike map."""

    def __init__(self, d_model: int, d_state: int = 4, bidir: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir = bidir
        self.in_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, 2 * d_state)     # input-dependent B, C
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def _scan(self, u: torch.Tensor) -> torch.Tensor:     # u: [B,L,D]
        delta = F.softplus(self.dt_proj(u))               # [B,L,D]
        BC = self.x_proj(u)                               # [B,L,2N]
        Bm, Cm = BC[..., :self.d_state], BC[..., self.d_state:]   # [B,L,N]
        A = -torch.exp(self.A_log)                        # [D,N]
        dA = torch.exp(delta.unsqueeze(-1) * A)           # [B,L,D,N]
        dBu = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * u.unsqueeze(-1)   # [B,L,D,N]
        h = assoc_scan(dA, dBu)                           # [B,L,D,N]
        y = (h * Cm.unsqueeze(2)).sum(-1)                 # [B,L,D]
        return y + u * self.D

    def forward(self, u: torch.Tensor) -> torch.Tensor:   # [B,L,D] -> [B,L,D]
        u = self.in_proj(u)
        y = self._scan(u)
        if self.bidir:
            y = y + torch.flip(self._scan(torch.flip(u, [1])), [1])
        return self.out_proj(F.silu(y))


class _FeedbackSNN(nn.Module):
    """Conv-recurrent spiking facilitation: a leaky spiking hidden map driven by the
    denoiser's forwarded spikes; lateral spread via a k×k conv; reads out a per-pixel
    bias added to the threshold neurons next bin. Output projection zero-init so the
    untrained model ≈ the frozen STCD core."""

    def __init__(self, Cf: int = 16, lam: float = 0.7, k: int = 5,
                 beta: float = 5.0, vth: float = 1.0):
        super().__init__()
        self.w_in = nn.Conv2d(1, Cf, k, padding=k // 2)
        self.read = nn.Conv2d(Cf, 1, 1)
        nn.init.zeros_(self.read.weight); nn.init.zeros_(self.read.bias)
        self.lam = lam; self.beta = beta; self.vth = vth

    def step(self, state, s_in):                          # state [B,Cf,H,W]|None, s_in [B,1,H,W]
        h = self.w_in(s_in)
        pre = h if state is None else self.lam * state + h
        sh = spike(pre - self.vth, self.beta)             # hidden spikes
        new_state = pre - sh * self.vth                   # LIF reset by subtraction
        bias = self.read(sh)                              # [B,1,H,W] facilitation
        return new_state, bias


class _FeedbackSSN(nn.Module):
    """Spatial State Network feedback: embed the forwarded-spike map, leaky-integrate,
    then mix long-range with a Z-ordered selective scan (``SpatialSSM``) to predict the
    per-pixel facilitation bias. Read-out zero-init (untrained ≈ frozen STCD core)."""

    def __init__(self, Cf: int = 8, lam: float = 0.7, d_state: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(1, Cf, 3, padding=1)
        self.ssm = SpatialSSM(Cf, d_state=d_state)
        self.read = nn.Conv2d(Cf, 1, 1)
        nn.init.zeros_(self.read.weight); nn.init.zeros_(self.read.bias)
        self.lam = lam

    def step(self, state, s_in):                          # state [B,Cf,H,W]|None, s_in [B,1,H,W]
        h = self.embed(s_in)
        pre = h if state is None else self.lam * state + h
        B, C, H, W = pre.shape
        seq = pre.flatten(2).transpose(1, 2)              # [B, H*W, C] row-major
        order, inv = morton_order(H, W, pre.device)
        seq = seq.index_select(1, order)                  # to Z-order
        y = self.ssm(seq)                                 # [B,L,C]
        y = y.index_select(1, inv).transpose(1, 2).reshape(B, C, H, W)
        bias = self.read(y)                               # [B,1,H,W]
        return pre, bias


class FeedbackDenoiser(nn.Module):
    """Idea 2 — output→threshold feedback denoiser (streaming-causal).

    A FROZEN single-neuron STCD core (fixed k×k all-ones box-sum incl. centre + leaky
    LIF with subtractive reset, matching ``SpikingFrontEnd``) plus a learned feedback
    module. Within the time loop, the bias applied to bin ``t``'s decision is read from
    forwarded spikes of bins ``< t`` only (read before the spike, state updated after),
    so it is strictly causal / single-pass / edge-deployable. ``backend="snn"`` =
    conv-recurrent spiking; ``backend="ssn"`` = pure-PyTorch spatial selective-scan SSM.

    Same I/O contract: ``[B,2,H,W,T]`` → per-(pixel,bin) logit ``[B,1,H,W,T]``.
    ``forward(x, use_feedback=False)`` ablates the feedback to the bare STCD core, for a
    clean in-model attribution of what the feedback adds."""

    def __init__(self, backend: str = "snn", k: int = 3, tau_steps: float = 8.0,
                 theta: float = 0.5, beta: float = 10.0, Cf: int | None = None,
                 d_state: int = 4):
        super().__init__()
        if backend not in {"snn", "ssn"}:
            raise ValueError(backend)
        self.backend = backend
        self.k = k
        self.theta = theta
        self.beta = beta
        self.alpha = float(torch.exp(torch.tensor(-1.0 / tau_steps)))
        self.register_buffer("box", torch.ones(1, 1, k, k))   # frozen box-sum (incl centre)
        if backend == "snn":
            self.fb = _FeedbackSNN(Cf=Cf or 16)
        else:
            self.fb = _FeedbackSSN(Cf=Cf or 8, d_state=d_state)

    def forward(self, x: torch.Tensor, use_feedback: bool = True) -> torch.Tensor:
        B, P, H, W, T = x.shape
        xs = x.sum(1, keepdim=True)                       # [B,1,H,W,T] polarity-summed
        v = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        bias = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        state = None
        out = []
        pad = self.k // 2
        for t in range(T):
            sup = F.conv2d(xs[..., t], self.box, padding=pad)   # [B,1,H,W] box-sum support
            v = self.alpha * v + sup
            b = bias if use_feedback else torch.zeros_like(bias)
            score = (v + b) - self.theta                  # per-cell logit
            s = spike(score, self.beta)                   # forwarded spikes (surrogate-diff)
            v = v - s * self.theta                        # reset by subtraction
            out.append(score)
            if use_feedback:
                state, bias = self.fb.step(state, s)      # uses bin-t output → bias for t+1
        return torch.stack(out, dim=-1)                   # [B,1,H,W,T]

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor, use_feedback: bool = True) -> torch.Tensor:
        return self.forward(x, use_feedback=use_feedback)[:, 0]


class SpikingUNet(nn.Module):
    """Spiking U-Net oracle (cost ignored) — a multi-scale spiking-conv encoder-decoder
    that emits a per-(pixel,bin) signal/noise logit ``[B,1,H,W,T]`` (same contract as
    B1/B2). Two encoder downsamples + a bottleneck + two skip-concat decoder upsamples;
    every conv is a LIF (membrane leak α, surrogate spike, subtractive reset) whose state
    is carried across the T bins. This is the architecture the LED-paper SOTA (DTSNN /
    EDSNN) uses, rebuilt from scratch on our rich ``[2,H,W,T]`` input + our objective, so
    the comparison to B2 is apples-to-apples (removes DTSNN's binarized-1-frame handicap).

    ``forward(x, state=...)`` returns ``(out, state)`` when ``return_state=True`` so the
    membranes can be CARRIED ACROSS 10 ms slices (temporal extent) — off by default to
    match B1/B2's per-slice reset. Inputs are zero-padded to a multiple of 4 (for the two
    2× pools) and the output is cropped back, so any H×W works."""

    def __init__(self, C: int = 16, tau_steps: float = 2.0, beta: float = 5.0):
        super().__init__()
        self.beta = beta
        self.alpha = float(torch.exp(torch.tensor(-1.0 / tau_steps)))
        self.enc1 = nn.Conv2d(2, C, 3, padding=1)
        self.enc2 = nn.Conv2d(C, 2 * C, 3, padding=1)
        self.bott = nn.Conv2d(2 * C, 4 * C, 3, padding=1)
        self.dec2 = nn.Conv2d(4 * C + 2 * C, 2 * C, 3, padding=1)   # skip-concat enc2
        self.dec1 = nn.Conv2d(2 * C + C, C, 3, padding=1)           # skip-concat enc1
        self.head = nn.Conv2d(C, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        B, P, H, W, T = x.shape
        Hp, Wp = ((H + 3) // 4) * 4, ((W + 3) // 4) * 4
        if (Hp, Wp) != (H, W):
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))     # pad W then H (T untouched)
        a, beta = self.alpha, self.beta
        v1 = v2 = vb = u2 = u1 = vo = 0.0
        if state is not None:
            v1, v2, vb, u2, u1, vo = state
        out = []
        for t in range(T):
            v1 = a * v1 + self.enc1(x[..., t]); s1 = spike(v1 - 1.0, beta); v1 = v1 - s1
            v2 = a * v2 + self.enc2(F.avg_pool2d(s1, 2)); s2 = spike(v2 - 1.0, beta); v2 = v2 - s2
            vb = a * vb + self.bott(F.avg_pool2d(s2, 2)); sb = spike(vb - 1.0, beta); vb = vb - sb
            ub = F.interpolate(sb, scale_factor=2, mode="nearest")
            u2 = a * u2 + self.dec2(torch.cat([ub, s2], 1)); su2 = spike(u2 - 1.0, beta); u2 = u2 - su2
            uu = F.interpolate(su2, scale_factor=2, mode="nearest")
            u1 = a * u1 + self.dec1(torch.cat([uu, s1], 1)); su1 = spike(u1 - 1.0, beta); u1 = u1 - su1
            vo = a * vo + self.head(su1)                    # readout membrane (no reset)
            out.append(vo)
        y = torch.stack(out, dim=-1)                        # [B,1,Hp,Wp,T]
        if (Hp, Wp) != (H, W):
            y = y[:, :, :H, :W, :]
        if return_state:
            return y, (v1, v2, vb, u2, u1, vo)
        return y

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0]


class FeedbackOnBase(nn.Module):
    """Streaming-causal feedback wrapped around a (typically frozen, pretrained) base
    denoiser that emits per-(pixel,bin) logits ``[B,1,H,W,T]`` — e.g. B2 ``SpatialDenoiser``.

    Tests whether output→threshold feedback **stacks on a learned spatial core** rather
    than the bare box-sum: at each bin the base logit is thresholded to a forwarded spike,
    which (via the SNN/SSN feedback) biases the *next* bin's logit. With ``use_feedback=
    False`` the output is exactly the base model, so on−off is a clean attribution of what
    the feedback adds on top of the spatial ceiling. ``thr`` is the spike threshold on the
    base logit (0 = the base's own decision boundary). Same I/O contract."""

    def __init__(self, base: nn.Module, backend: str = "ssn", thr: float = 0.0,
                 beta: float = 10.0, Cf: int | None = None, d_state: int = 4,
                 freeze_base: bool = True):
        super().__init__()
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.thr = thr
        self.beta = beta
        if backend == "snn":
            self.fb = _FeedbackSNN(Cf=Cf or 16)
        elif backend == "ssn":
            self.fb = _FeedbackSSN(Cf=Cf or 8, d_state=d_state)
        else:
            raise ValueError(backend)

    def forward(self, x: torch.Tensor, use_feedback: bool = True) -> torch.Tensor:
        if self.base.training != self.training:
            self.base.eval()                       # base stays in eval (frozen) regardless
        base_logits = self.base(x)                 # [B,1,H,W,T]
        if not use_feedback:
            return base_logits
        B, _, H, W, T = base_logits.shape
        bias = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        state = None
        out = []
        for t in range(T):
            lt = base_logits[..., t]               # [B,1,H,W]
            score = lt + bias                      # feedback from spikes < t
            s = spike(score - self.thr, self.beta)
            out.append(score)
            state, bias = self.fb.step(state, s)
        return torch.stack(out, dim=-1)            # [B,1,H,W,T]

    @torch.no_grad()
    def score_cells(self, x: torch.Tensor, use_feedback: bool = True) -> torch.Tensor:
        return self.forward(x, use_feedback=use_feedback)[:, 0]
