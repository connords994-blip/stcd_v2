"""Event representation and conversion between sparse event lists and dense tensors.

An event is a tuple ``(x, y, t, p)``: pixel column ``x``, row ``y``, timestamp
``t`` (seconds), polarity ``p`` (1 = ON / brightness increase, 0 = OFF / decrease).

The whole front-end operates on a dense, time-binned tensor of shape
``[P, H, W, T]`` (event counts per polarity / pixel / time-bin). This makes every
stage a vectorised, MPS-accelerated, differentiable tensor op. ``Events`` carries
an optional boolean ``labels`` array (True = real signal, False = BA noise) so we
can compute ground-truth denoising metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

Tensor = torch.Tensor


@dataclass
class Events:
    """A sparse event stream stored as parallel arrays (all length N).

    Attributes
    ----------
    xs, ys : int arrays in ``[0, W)`` / ``[0, H)``
    ts     : float array of timestamps in seconds (need not be sorted)
    ps     : int array of polarities, 0 (OFF) or 1 (ON)
    H, W   : sensor resolution
    labels : optional bool array, True = real signal, False = injected noise.
             ``None`` when ground truth is unknown (e.g. real recordings).
    kinds  : optional int array giving the *kind* of each event (see KIND_* in
             :mod:`stcd.synth`): 0=signal, 1=BA, 2=hot-pixel (1a), 3=column/row
             correlated readout (1b), 4=cluster. Lets the bake-off score 1a and 1b
             separately. ``None`` when unknown. When present, ``labels`` should
             equal ``kinds == 0``.
    """

    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    ps: np.ndarray
    H: int
    W: int
    labels: Optional[np.ndarray] = None
    kinds: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.xs = np.asarray(self.xs, dtype=np.int64)
        self.ys = np.asarray(self.ys, dtype=np.int64)
        self.ts = np.asarray(self.ts, dtype=np.float64)
        # Accept -1/+1 polarity input and fold to 0/1.
        ps = np.asarray(self.ps)
        if ps.size and ps.min() < 0:
            ps = (ps > 0).astype(np.int64)
        self.ps = ps.astype(np.int64)
        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=bool)
        if self.kinds is not None:
            self.kinds = np.asarray(self.kinds, dtype=np.int64)
        n = len(self.xs)
        if not (len(self.ys) == len(self.ts) == len(self.ps) == n):
            raise ValueError("Events arrays must all have the same length")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels length must match number of events")
        if self.kinds is not None and len(self.kinds) != n:
            raise ValueError("kinds length must match number of events")

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def duration(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self.ts.max() - self.ts.min())

    def time_sorted(self) -> "Events":
        """Return a copy sorted by timestamp (stable)."""
        order = np.argsort(self.ts, kind="stable")
        return replace(
            self,
            xs=self.xs[order],
            ys=self.ys[order],
            ts=self.ts[order],
            ps=self.ps[order],
            labels=None if self.labels is None else self.labels[order],
            kinds=None if self.kinds is None else self.kinds[order],
        )

    def select(self, mask: np.ndarray) -> "Events":
        """Return the subset of events where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        return replace(
            self,
            xs=self.xs[mask],
            ys=self.ys[mask],
            ts=self.ts[mask],
            ps=self.ps[mask],
            labels=None if self.labels is None else self.labels[mask],
            kinds=None if self.kinds is None else self.kinds[mask],
        )

    @staticmethod
    def concat(*streams: "Events") -> "Events":
        streams = [s for s in streams if len(s) > 0]
        if not streams:
            raise ValueError("concat requires at least one non-empty stream")
        H, W = streams[0].H, streams[0].W
        have_labels = all(s.labels is not None for s in streams)
        have_kinds = all(s.kinds is not None for s in streams)
        return Events(
            xs=np.concatenate([s.xs for s in streams]),
            ys=np.concatenate([s.ys for s in streams]),
            ts=np.concatenate([s.ts for s in streams]),
            ps=np.concatenate([s.ps for s in streams]),
            H=H,
            W=W,
            labels=(
                np.concatenate([s.labels for s in streams]) if have_labels else None
            ),
            kinds=(
                np.concatenate([s.kinds for s in streams]) if have_kinds else None
            ),
        )


@dataclass
class TimeGrid:
    """Maps continuous time to discrete bins. ``t0`` is the start, ``dt`` the
    bin width (s), ``T`` the number of bins; bin ``i`` covers ``[t0+i*dt, t0+(i+1)*dt)``."""

    t0: float
    dt: float
    T: int

    @classmethod
    def from_events(cls, ev: Events, dt: float) -> "TimeGrid":
        if len(ev) == 0:
            return cls(0.0, dt, 1)
        t0 = float(ev.ts.min())
        span = float(ev.ts.max()) - t0
        T = max(1, int(np.ceil((span + 1e-12) / dt)))
        return cls(t0, dt, T)

    def bin_index(self, ts: np.ndarray) -> np.ndarray:
        idx = np.floor((ts - self.t0) / self.dt).astype(np.int64)
        return np.clip(idx, 0, self.T - 1)

    @property
    def centers(self) -> np.ndarray:
        return self.t0 + (np.arange(self.T) + 0.5) * self.dt


def events_to_tensor(
    ev: Events,
    grid: Optional[TimeGrid] = None,
    dt: Optional[float] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, TimeGrid]:
    """Bin an event stream into a dense count tensor ``[P=2, H, W, T]``.

    Index 0 along P is OFF polarity, index 1 is ON. Either pass an explicit
    ``grid`` or a bin width ``dt`` (a grid is then derived from the stream).
    """
    if grid is None:
        if dt is None:
            raise ValueError("provide either grid or dt")
        grid = TimeGrid.from_events(ev, dt)

    tensor = torch.zeros((2, ev.H, ev.W, grid.T), dtype=dtype)
    if len(ev) == 0:
        return (tensor.to(device) if device else tensor), grid

    tb = grid.bin_index(ev.ts)
    # Flatten (p, y, x, t) -> linear index and scatter-add counts.
    flat = torch.from_numpy(
        ((ev.ps * ev.H + ev.ys) * ev.W + ev.xs) * grid.T + tb
    ).long()
    tensor.view(-1).scatter_add_(0, flat, torch.ones(len(ev), dtype=dtype))
    return (tensor.to(device) if device else tensor), grid


def tensor_to_events(
    tensor: Tensor,
    grid: TimeGrid,
    H: int,
    W: int,
    threshold: float = 0.5,
) -> Events:
    """Convert a dense ``[P,H,W,T]`` activation/count tensor back to an event list.

    One output event is emitted per (p, y, x, t) cell whose value exceeds
    ``threshold``; the timestamp is the bin centre. Used for AER output and for
    feeding filtered streams to downstream tasks/metrics.
    """
    t = tensor.detach().cpu()
    p_idx, y_idx, x_idx, t_idx = torch.where(t > threshold)
    centers = torch.from_numpy(grid.centers)
    return Events(
        xs=x_idx.numpy(),
        ys=y_idx.numpy(),
        ts=centers[t_idx].numpy(),
        ps=p_idx.numpy(),
        H=H,
        W=W,
    )


def labels_to_tensor(ev: Events, grid: TimeGrid) -> Tensor:
    """Per-cell ground-truth signal fraction, shape ``[P,H,W,T]`` in [0,1].

    Each cell = (signal events in cell) / (total events in cell). Used to derive
    per-cell signal/noise ground truth aligned with a binned activation tensor.
    """
    if ev.labels is None:
        raise ValueError("events have no ground-truth labels")
    sig, _ = events_to_tensor(ev.select(ev.labels), grid=grid)
    tot, _ = events_to_tensor(ev, grid=grid)
    return torch.where(tot > 0, sig / tot.clamp(min=1.0), torch.zeros_like(tot))
