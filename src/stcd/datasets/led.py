"""LED dataset loader (Duan et al., "LED", CVPR 2024; arXiv 2405.19718).

Large-scale real-world *paired* event-denoising dataset with label-free DED ground
truth (no APS/IMU) → exact per-event signal/noise labels, including the hot-pixel /
correlated-readout noise STCD structurally can't remove.

Layouts differ by split (handled by prefix-matching the stream dirs):
  test:  <root>/LED_Test_upload/LED_Test_upload/{raw_event,denoised_event,noise_event}/scene_*/NNNNN.npy
  train: <root>/LED_Train_upload/{raw_events,denoised_events,noise_events}/scene_*/NNNNN.npy
Each .npy is (4, N) int64 = [x, y, p, t], t in microseconds, one file per 10 ms slice.
raw == denoised ∪ noise, so we build a labelled stream from denoised(True)+noise(False).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..events import Events

H, W = 720, 1280  # EVK4 sensor


def _base(root: str, split: str) -> str:
    if split == "train":
        cands = [os.path.join(root, "LED_Train_upload"), root]
    else:
        cands = [os.path.join(root, "LED_Test_upload", "LED_Test_upload"),
                 os.path.join(root, "LED_Test_upload"), root]
    for c in cands:
        if glob.glob(os.path.join(c, "*denoised*")):
            return c
    return root


def _streams(base: str) -> dict:
    subs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]

    def by_prefix(pref):
        for d in subs:
            if os.path.basename(d).startswith(pref):
                return d
        return None
    return {"sig": by_prefix("denoised"), "noise": by_prefix("noise"), "raw": by_prefix("raw")}


def available(root: str, split: str = "test") -> bool:
    return _streams(_base(root, split))["sig"] is not None


def scenes(root: str, split: str = "test") -> list[str]:
    sig = _streams(_base(root, split))["sig"]
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(sig, "scene_*")))


def test_scenes(root: str) -> list[str]:
    return scenes(root, "test")


def train_scenes(root: str) -> list[str]:
    return scenes(root, "train")


def scene_slices(root: str, scene: str, split: str = "test") -> list[str]:
    sig = _streams(_base(root, split))["sig"]
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(sig, scene, "*.npy")))


def _load_npy(path: str):
    a = np.load(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected 2-D (4,N), got {a.shape}")
    if a.shape[0] != 4 and a.shape[1] == 4:
        a = a.T
    return (a[0], a[1], a[2], np.asarray(a[3], dtype=np.float64) * 1e-6)  # µs -> s


def load_slice(root: str, scene: str, fname: str, split: str = "test",
               H: int = H, W: int = W) -> Events:
    st = _streams(_base(root, split))
    sx, sy, sp, st_ = _load_npy(os.path.join(st["sig"], scene, fname))
    npath = os.path.join(st["noise"], scene, fname)
    if os.path.exists(npath):
        nx, ny, npp, nt = _load_npy(npath)
    else:
        nx = ny = npp = nt = np.array([], dtype=np.float64)
    xs = np.concatenate([sx, nx]).astype(np.int64)
    ys = np.concatenate([sy, ny]).astype(np.int64)
    ps = np.concatenate([sp, npp]).astype(np.int64)
    ts = np.concatenate([st_, nt]).astype(np.float64)
    labels = np.concatenate([np.ones(len(sx), bool), np.zeros(len(nx), bool)])
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return Events(xs[ok], ys[ok], ts[ok], ps[ok], H=H, W=W, labels=labels[ok]).time_sorted()


def raw_count(root: str, scene: str, fname: str, split: str = "test") -> int:
    raw = _streams(_base(root, split))["raw"]
    p = os.path.join(raw, scene, fname) if raw else ""
    if not p or not os.path.exists(p):
        return -1
    a = np.load(p)
    return a.shape[1] if a.shape[0] == 4 else a.shape[0]
