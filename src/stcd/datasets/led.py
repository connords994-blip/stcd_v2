"""LED dataset loader (Duan et al., "LED", CVPR 2024; arXiv 2405.19718).

Large-scale real-world *paired* event-denoising dataset. Ground truth is generated
label-free by the DED dual-camera consistency framework (no APS/IMU), so unlike
DVSNOISE20 we get exact per-event signal/noise labels — and the noise contains the
hot-pixel / correlated-readout artifacts STCD structurally can't remove.

Layout (test split as downloaded):
    <root>/LED_Test_upload/LED_Test_upload/
        raw_event/scene_XXXX/NNNNN.npy        # raw   = signal + noise
        denoised_event/scene_XXXX/NNNNN.npy   # signal (GT)
        noise_event/scene_XXXX/NNNNN.npy      # noise (GT)
Each .npy is ``(4, N)`` int64 = rows [x, y, p, t], t in microseconds, one file per
10 ms slice. raw == denoised ∪ noise, so we build a labelled stream directly from
denoised (label=True) + noise (label=False) — no event matching needed.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..events import Events

# EVK4 sensor; observed x<1280, y<720. Fixed so the dense tensor grid is consistent.
H, W = 720, 1280


def _base(root: str) -> str:
    """Resolve the directory that holds {raw,denoised,noise}_event."""
    nested = os.path.join(root, "LED_Test_upload", "LED_Test_upload")
    if os.path.isdir(os.path.join(nested, "denoised_event")):
        return nested
    single = os.path.join(root, "LED_Test_upload")
    if os.path.isdir(os.path.join(single, "denoised_event")):
        return single
    return root


def available(root: str) -> bool:
    return os.path.isdir(os.path.join(_base(root), "denoised_event"))


def test_scenes(root: str) -> list[str]:
    den = os.path.join(_base(root), "denoised_event")
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(den, "scene_*")))


def scene_slices(root: str, scene: str) -> list[str]:
    den = os.path.join(_base(root), "denoised_event", scene)
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(den, "*.npy")))


def _load_npy(path: str):
    a = np.load(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected 2-D (4,N), got {a.shape}")
    if a.shape[0] != 4 and a.shape[1] == 4:
        a = a.T
    x, y, p, t = a[0], a[1], a[2], a[3]
    return (np.asarray(x), np.asarray(y), np.asarray(p),
            np.asarray(t, dtype=np.float64) * 1e-6)  # µs -> s


def load_slice(root: str, scene: str, fname: str,
               H: int = H, W: int = W) -> Events:
    """One 10 ms slice as a labelled stream (signal=denoised, noise=noise_event)."""
    base = _base(root)
    sx, sy, sp, st = _load_npy(os.path.join(base, "denoised_event", scene, fname))
    npath = os.path.join(base, "noise_event", scene, fname)
    if os.path.exists(npath):
        nx, ny, npp, nt = _load_npy(npath)
    else:
        nx = ny = npp = nt = np.array([], dtype=np.float64)
    xs = np.concatenate([sx, nx]).astype(np.int64)
    ys = np.concatenate([sy, ny]).astype(np.int64)
    ps = np.concatenate([sp, npp]).astype(np.int64)
    ts = np.concatenate([st, nt]).astype(np.float64)
    labels = np.concatenate([np.ones(len(sx), bool), np.zeros(len(nx), bool)])
    # Defensive: keep only in-bounds pixels (sensor edge / stray indices).
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    ev = Events(xs[ok], ys[ok], ts[ok], ps[ok], H=H, W=W, labels=labels[ok])
    return ev.time_sorted()


def raw_count(root: str, scene: str, fname: str) -> int:
    """#events in the raw stream for the slice (to sanity-check the partition)."""
    p = os.path.join(_base(root), "raw_event", scene, fname)
    if not os.path.exists(p):
        return -1
    a = np.load(p)
    return a.shape[1] if a.shape[0] == 4 else a.shape[0]
