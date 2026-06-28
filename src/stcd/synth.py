"""Synthetic event-stream generator with ground-truth signal/noise labels.

Two parts, mirroring the proposal's DND21/DVSCLEAN setup:

1. **Signal** events come from a proper threshold-crossing camera model applied
   to a rendered intensity video of moving shapes. As an edge sweeps across
   pixels, neighbouring pixels fire at nearby times — i.e. the events are
   spatially and temporally *correlated*, which is exactly the structure a
   coincidence filter should preserve. All such events get ``label=True``.

2. **Noise** is injected background activity: Poisson events uniform in space,
   time and polarity at a known rate (Hz per pixel). These are *uncorrelated*
   and get ``label=False``.

The known labels let us measure Signal-Retain / Noise-Removal / ROC exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .events import Events

# Noise-kind codes carried on ``Events.kinds`` so the bake-off can score the two
# Idea-1 noise processes separately (1a = hot pixels, 1b = correlated readout).
KIND_SIGNAL = 0
KIND_BA = 1       # uncorrelated background activity (Poisson)
KIND_HOT = 2      # 1a — hot/leaky pixels (fixed location, high self-rate)
KIND_COLUMN = 3   # 1b — column/row correlated readout noise
KIND_CLUSTER = 4  # static flickering blobs (spatially correlated, no motion)


@dataclass
class SynthConfig:
    H: int = 120
    W: int = 160
    duration: float = 0.5           # seconds
    fps: int = 2000                 # render frame rate (sets temporal resolution)
    contrast_threshold: float = 0.15  # log-intensity change needed to fire (C)
    noise_rate_hz: float = 1.0      # background activity, events / pixel / second
    scene: Literal["bars", "hbars", "disks", "checker"] = "bars"
    checker_cell: int = 4           # checkerboard square size (px) — fine texture/corners
    num_objects: int = 3
    bg_intensity: float = 0.5       # background grey level in (0, 1]
    fg_intensity: float = 1.0       # moving-object level
    seed: int = 0
    jitter_frac: float = 0.5        # timestamp jitter within a frame interval
    # Optional motion/size control (fractions of W/H / min(H,W)); None = defaults.
    vx_range: tuple[float, float] | None = None   # horizontal speed range, frac of W/s
    vy_range: tuple[float, float] | None = None   # vertical speed range, frac of H/s
    size_range: tuple[float, float] = (0.06, 0.14)  # object size, frac of min(H,W)
    # Hot pixels: stuck/leaky sensor pixels that fire at very high rate. Their
    # events are temporally self-correlated, so they survive naive coincidence
    # filtering — the realistic failure mode the per-pixel rate-cap addresses.
    n_hot_pixels: int = 0
    hot_pixel_rate_hz: float = 300.0
    # Clustered ("burst") noise: static k×k blobs that flicker over a short window
    # with NO coherent motion — spatially correlated, so it defeats neighbour-count
    # filters (BAF/KNoise) but should be rejected by motion-opponent detectors.
    n_clusters: int = 0
    cluster_size: int = 3
    cluster_events: int = 40
    cluster_burst: float = 0.015   # each blob fires its events within this window (s)
    # Column/row correlated-readout noise (1b): a shared amplifier/arbiter fires a
    # group of pixels along one readout line *near-simultaneously*. Because the
    # firing pixels are spatial neighbours along the line, this noise genuinely HAS
    # neighbour support and passes coincidence filters (BAF/time-surface/STCD) — the
    # 1b failure mode that per-pixel 1a methods miss. Targeted by the column detector
    # and by B3's scatter-side contribution discount.
    n_columns: int = 0
    column_axis: Literal["col", "row"] = "col"
    column_rate_hz: float = 50.0   # bursts per line per second
    column_jitter: float = 2e-5    # coincidence window of a burst (s) — << τ
    column_frac: float = 0.5       # fraction of the line's pixels that fire per burst
    column_span_frac: float = 1.0  # fraction of the line length the noise spans
    column_mode: Literal["fixed", "roaming"] = "fixed"  # persistent hot line vs roaming arbiter


def _render_video(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """Render an intensity video of shape ``[F, H, W]`` in (0, 1]."""
    F = max(2, int(round(cfg.duration * cfg.fps)))
    H, W = cfg.H, cfg.W
    video = np.full((F, H, W), cfg.bg_intensity, dtype=np.float32)
    ts = np.arange(F) / cfg.fps
    yy, xx = np.mgrid[0:H, 0:W]

    vx_lo, vx_hi = cfg.vx_range if cfg.vx_range is not None else (-0.6, 0.6)
    vy_lo, vy_hi = cfg.vy_range if cfg.vy_range is not None else (-0.6, 0.6)
    objects = []
    for _ in range(cfg.num_objects):
        objects.append(
            dict(
                x0=rng.uniform(0.1 * W, 0.9 * W),
                y0=rng.uniform(0.1 * H, 0.9 * H),
                vx=rng.uniform(vx_lo, vx_hi) * W,   # px / s
                vy=rng.uniform(vy_lo, vy_hi) * H,
                size=rng.uniform(*cfg.size_range) * min(H, W),
                level=cfg.fg_intensity if rng.random() > 0.4 else cfg.bg_intensity * 0.4,
            )
        )

    # Recall stress: a globally-translating checkerboard is dense in edges AND
    # corners, where a pixel genuinely out-fires its neighbours — the exact real
    # signal B3's anti-Hebbian discount risks suppressing. Measure SR here.
    if cfg.scene == "checker":
        o = objects[0]
        cell = max(2, int(cfg.checker_cell))
        for f, t in enumerate(ts):
            sx = o["vx"] * t; sy = o["vy"] * t
            cb = (((xx + sx) // cell).astype(np.int64)
                  + ((yy + sy) // cell).astype(np.int64)) % 2
            video[f] = np.where(cb == 0, cfg.bg_intensity, cfg.fg_intensity).astype(np.float32)
        return video

    for f, t in enumerate(ts):
        for o in objects:
            # Reflect off the borders to keep objects in frame.
            cx = _bounce(o["x0"] + o["vx"] * t, 0, W - 1)
            cy = _bounce(o["y0"] + o["vy"] * t, 0, H - 1)
            if cfg.scene == "disks":
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= o["size"] ** 2
            elif cfg.scene == "hbars":  # horizontal bars sweeping vertically
                mask = np.abs(yy - cy) <= o["size"]
            else:  # vertical bars of fixed width sweeping horizontally
                mask = np.abs(xx - cx) <= o["size"]
            video[f][mask] = o["level"]
    return video


def _bounce(pos: float, lo: float, hi: float) -> float:
    """Triangle-wave reflection of ``pos`` into ``[lo, hi]``."""
    span = hi - lo
    if span <= 0:
        return lo
    m = (pos - lo) % (2 * span)
    return lo + (m if m <= span else 2 * span - m)


def _events_from_video(
    video: np.ndarray, cfg: SynthConfig, rng: np.random.Generator
) -> Events:
    """Threshold-crossing event model (ESIM/v2e style) on a log-intensity video."""
    F, H, W = video.shape
    log = np.log(np.clip(video, 1e-3, None)).astype(np.float32)
    ref = log[0].copy()                       # per-pixel reference level
    C = cfg.contrast_threshold
    frame_dt = 1.0 / cfg.fps

    xs, ys, ts, ps = [], [], [], []
    for f in range(1, F):
        diff = log[f] - ref
        n = np.floor(np.abs(diff) / C).astype(np.int64)   # crossings per pixel
        n = np.minimum(n, 8)                              # guard pathological counts
        fired = n > 0
        if not fired.any():
            continue
        pol = (diff > 0).astype(np.int64)
        fy, fx = np.nonzero(fired)
        counts = n[fy, fx]
        polf = pol[fy, fx]
        # Expand each fired pixel into ``count`` events.
        rep = np.repeat(np.arange(len(fy)), counts)
        ex, ey, epol = fx[rep], fy[rep], polf[rep]
        # Timestamps within this frame interval, with sub-frame jitter.
        base = f * frame_dt
        jit = rng.uniform(0.0, cfg.jitter_frac * frame_dt, size=len(rep))
        et = base + jit
        xs.append(ex); ys.append(ey); ts.append(et); ps.append(epol)
        # Advance the reference by the consumed crossings (signed).
        signed = np.where(pol > 0, 1.0, -1.0) * n * C
        ref += signed.astype(np.float32)

    if not xs:
        return Events(np.array([]), np.array([]), np.array([]), np.array([]),
                      H=H, W=W, labels=np.array([], dtype=bool),
                      kinds=np.array([], dtype=np.int64))
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    ts = np.concatenate(ts); ps = np.concatenate(ps)
    return Events(xs, ys, ts, ps, H=H, W=W,
                  labels=np.ones(len(xs), dtype=bool),
                  kinds=np.full(len(xs), KIND_SIGNAL, dtype=np.int64))


def inject_noise(
    signal: Events, rate_hz: float, duration: float, rng: np.random.Generator
) -> Events:
    """Add uncorrelated Poisson background-activity events (``label=False``)."""
    H, W = signal.H, signal.W
    expected = rate_hz * H * W * duration
    n = int(rng.poisson(expected)) if expected > 0 else 0
    if n == 0:
        return signal
    t0 = float(signal.ts.min()) if len(signal) else 0.0
    noise = Events(
        xs=rng.integers(0, W, size=n),
        ys=rng.integers(0, H, size=n),
        ts=rng.uniform(t0, t0 + duration, size=n),
        ps=rng.integers(0, 2, size=n),
        H=H, W=W,
        labels=np.zeros(n, dtype=bool),
        kinds=np.full(n, KIND_BA, dtype=np.int64),
    )
    return Events.concat(signal, noise).time_sorted()


def inject_hot_pixels(
    stream: Events, n_hot: int, rate_hz: float, duration: float,
    rng: np.random.Generator
) -> Events:
    """Add a few stuck/leaky pixels firing at a high rate (``label=False``).

    Hot-pixel events repeat at the *same* pixel, so they are temporally
    self-correlated and survive naive coincidence filtering — defeated by the
    per-pixel rate cap (see :func:`stcd.baselines.rate_cap`)."""
    H, W = stream.H, stream.W
    if n_hot <= 0:
        return stream
    t0 = float(stream.ts.min()) if len(stream) else 0.0
    hx = rng.integers(0, W, size=n_hot)
    hy = rng.integers(0, H, size=n_hot)
    xs, ys, ts, ps = [], [], [], []
    for px, py in zip(hx, hy):
        k = int(rng.poisson(rate_hz * duration))
        if k == 0:
            continue
        xs.append(np.full(k, px)); ys.append(np.full(k, py))
        ts.append(rng.uniform(t0, t0 + duration, size=k))
        ps.append(rng.integers(0, 2, size=k))
    if not xs:
        return stream
    n_hot_ev = sum(len(t) for t in ts)
    hot = Events(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts),
                 np.concatenate(ps), H=H, W=W,
                 labels=np.zeros(n_hot_ev, dtype=bool),
                 kinds=np.full(n_hot_ev, KIND_HOT, dtype=np.int64))
    return Events.concat(stream, hot).time_sorted()


def inject_cluster_noise(
    stream: Events, n_clusters: int, size: int, events_per_cluster: int,
    duration: float, rng: np.random.Generator, burst: float = 0.015
) -> Events:
    """Add static ``size×size`` blobs that each fire ``events_per_cluster`` events
    within a short ``burst`` window (``label=False``). The burst makes them locally
    *dense* (high instantaneous event rate, like a real edge) yet *static* (no
    coherent motion). Neighbour-counting filters keep them; an opponent motion
    detector should reject them."""
    H, W = stream.H, stream.W
    if n_clusters <= 0:
        return stream
    t0 = float(stream.ts.min()) if len(stream) else 0.0
    cx = rng.integers(size, W - size, size=n_clusters)
    cy = rng.integers(size, H - size, size=n_clusters)
    cstart = rng.uniform(t0, t0 + max(duration - burst, 1e-3), size=n_clusters)
    r = size // 2
    xs, ys, ts, ps = [], [], [], []
    for bx, by, t_start in zip(cx, cy, cstart):
        k = events_per_cluster
        ox = rng.integers(-r, r + 1, size=k)
        oy = rng.integers(-r, r + 1, size=k)
        xs.append(np.clip(bx + ox, 0, W - 1)); ys.append(np.clip(by + oy, 0, H - 1))
        ts.append(rng.uniform(t_start, t_start + burst, size=k))   # dense burst
        ps.append(rng.integers(0, 2, size=k))
    n_blob_ev = sum(len(t) for t in ts)
    blob = Events(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts),
                  np.concatenate(ps), H=H, W=W,
                  labels=np.zeros(n_blob_ev, dtype=bool),
                  kinds=np.full(n_blob_ev, KIND_CLUSTER, dtype=np.int64))
    return Events.concat(stream, blob).time_sorted()


def inject_column_noise(
    stream: Events, n_lines: int, duration: float, rng: np.random.Generator,
    axis: str = "col", rate_hz: float = 50.0, jitter: float = 2e-5,
    frac: float = 0.5, span_frac: float = 1.0, mode: str = "fixed",
) -> Events:
    """Add column/row correlated-readout noise (1b, ``label=False``, ``kind=COLUMN``).

    A shared readout amplifier/arbiter fires a *contiguous run* of pixels along one
    line (``axis='col'`` → a column at fixed x; ``'row'`` → a row at fixed y)
    *near-simultaneously* (within ``jitter`` ≪ τ). The run is contiguous, so the
    firing pixels are spatial neighbours and this noise genuinely HAS neighbour
    support → it passes coincidence filters, the 1b failure mode per-pixel 1a
    methods miss. ``rate_hz`` bursts/line/s; each burst lights a run of
    ``frac·span_frac·line_len`` pixels.

    ``mode``:
    * ``"fixed"`` — a persistent hot line (column of stuck pixels). High per-pixel
      rate, so ``rate_cap`` and B3 (persistent self-firer) CAN catch it.
    * ``"roaming"`` — each burst hits a fresh random line + position (arbiter
      contention). Per-pixel rate stays LOW so ``rate_cap`` and B3 miss it; only a
      column-aggregate/anisotropy detector (Strategy 8/9) should catch it. This is
      the genuinely hard 1b that distinguishes the strategies."""
    H, W = stream.H, stream.W
    if n_lines <= 0:
        return stream
    t0 = float(stream.ts.min()) if len(stream) else 0.0
    n_idx = W if axis == "col" else H
    line_len = H if axis == "col" else W
    seg = max(2, int(round(span_frac * line_len)))
    run = min(seg, max(2, int(round(frac * seg))))   # contiguous run length
    xs, ys, ts, ps = [], [], [], []
    for _ in range(n_lines):
        idx0 = rng.integers(0, n_idx)
        start0 = rng.integers(0, line_len - seg + 1) if seg < line_len else 0
        n_bursts = int(rng.poisson(rate_hz * duration))
        for _ in range(n_bursts):
            if mode == "roaming":
                idx = int(rng.integers(0, n_idx))
                rstart = int(rng.integers(0, line_len - run + 1)) if run < line_len else 0
            else:  # fixed line; contiguous run anchored within its segment
                idx = int(idx0)
                rstart = start0 + (int(rng.integers(0, seg - run + 1)) if seg > run else 0)
            pix = np.arange(rstart, rstart + run)
            tb = rng.uniform(t0, t0 + duration)
            et = tb + rng.uniform(0.0, jitter, size=run)
            if axis == "col":
                xs.append(np.full(run, idx)); ys.append(pix)
            else:
                xs.append(pix); ys.append(np.full(run, idx))
            ts.append(et)
            ps.append(rng.integers(0, 2, size=run))
    if not xs:
        return stream
    n_col_ev = sum(len(t) for t in ts)
    line = Events(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts),
                  np.concatenate(ps), H=H, W=W,
                  labels=np.zeros(n_col_ev, dtype=bool),
                  kinds=np.full(n_col_ev, KIND_COLUMN, dtype=np.int64))
    return Events.concat(stream, line).time_sorted()


def generate(cfg: SynthConfig | None = None) -> Events:
    """Generate a labelled noisy event stream from a ``SynthConfig``."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    video = _render_video(cfg, rng)
    signal = _events_from_video(video, cfg, rng)
    stream = inject_noise(signal, cfg.noise_rate_hz, cfg.duration, rng)
    stream = inject_hot_pixels(stream, cfg.n_hot_pixels, cfg.hot_pixel_rate_hz,
                               cfg.duration, rng)
    stream = inject_cluster_noise(stream, cfg.n_clusters, cfg.cluster_size,
                                  cfg.cluster_events, cfg.duration, rng,
                                  burst=cfg.cluster_burst)
    return inject_column_noise(stream, cfg.n_columns, cfg.duration, rng,
                               axis=cfg.column_axis, rate_hz=cfg.column_rate_hz,
                               jitter=cfg.column_jitter, frac=cfg.column_frac,
                               span_frac=cfg.column_span_frac, mode=cfg.column_mode)


def generate_with_video(cfg: SynthConfig | None = None) -> tuple[Events, np.ndarray]:
    """Like :func:`generate` but also returns the clean intensity video
    (``[F,H,W]``) — used as a reconstruction reference for the FireNet head."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    video = _render_video(cfg, rng)
    signal = _events_from_video(video, cfg, rng)
    noisy = inject_noise(signal, cfg.noise_rate_hz, cfg.duration, rng)
    return noisy, video
