"""LED test-split bake-off: zero-shot STCD + B3 vs DTSNN's 86.2% DA.

LED is the first dataset with real hot/correlated-readout noise AND exact per-event
GT labels, and the first metric shared with DTSNN (DA = 1/2(SR+NR)). STCD is
zero-shot (no training) vs DTSNN supervised, so the bar is "how close to 86.2 / does
B3 help on real non-isolated noise", not superiority.

DA is operating-point-dependent, so we report BOTH the threshold-free AUC and the
best achievable DA (global threshold sweep on the per-event score) per method.

Env: LED_PER_SCENE (slices/scene, default 2), LED_MAX_SLICES (cap, default 120).
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.frontend import SpikingFrontEnd, FrontEndConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
PER_SCENE = int(os.environ.get("LED_PER_SCENE", "2"))
MAX_SLICES = int(os.environ.get("LED_MAX_SLICES", "120"))

BASE = dict(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=1e-3)
VARIANTS = {
    "baseline":           dict(),
    "B3 scatter/raw/neg": dict(b3=True, b3_side="scatter", b3_update="raw", b3_clamp_nonneg=False),
    "B3 gather/raw/neg":  dict(b3=True, b3_side="gather",  b3_update="raw", b3_clamp_nonneg=False),
}


def best_da(scores, labels):
    """Max DA = max_t 1/2(SR(t)+NR(t)) over thresholds, with the SR/NR there."""
    s = np.asarray(scores); y = np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan"), float("nan"), float("nan")
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    best = (-1.0, 0.0, 0.0)
    P, N = y.sum(), (~y).sum()
    for t in thr:
        keep = s >= t
        sr = (keep & y).sum() / P
        nr = (~keep & ~y).sum() / N
        da = 0.5 * (sr + nr)
        if da > best[0]:
            best = (da, sr, nr)
    return best


def main():
    if not led.available(ROOT):
        print(f"LED not found under {ROOT}"); return
    fes = {k: SpikingFrontEnd(FrontEndConfig(**BASE, **e)) for k, e in VARIANTS.items()}
    scenes = led.test_scenes(ROOT)
    print(f"LED test scenes: {len(scenes)}  (PER_SCENE={PER_SCENE}, MAX_SLICES={MAX_SLICES})")

    acc = {k: {"s": [], "y": []} for k in VARIANTS}
    n_slices = n_ev = 0
    sanity_done = False
    for sc in scenes:
        for fn in led.scene_slices(ROOT, sc)[:PER_SCENE]:
            if n_slices >= MAX_SLICES:
                break
            ev = led.load_slice(ROOT, sc, fn)
            if len(ev) < 100 or ev.labels.all() or not ev.labels.any():
                continue
            if not sanity_done:
                rc = led.raw_count(ROOT, sc, fn)
                print(f"sanity [{sc}/{fn}]: signal+noise={len(ev)} vs raw={rc} "
                      f"(signal_frac={ev.labels.mean():.3f})")
                sanity_done = True
            for k, fe in fes.items():
                acc[k]["s"].append(fe.score_events(ev))
                acc[k]["y"].append(ev.labels)
            n_slices += 1; n_ev += len(ev)
        if n_slices >= MAX_SLICES:
            break

    print(f"\nscored {n_slices} slices, {n_ev} events\n")
    print(f"{'method':22s} {'AUC':>7s} {'maxDA':>7s} {'SR@DA':>7s} {'NR@DA':>7s}")
    print("-" * 54)
    for k in VARIANTS:
        s = np.concatenate(acc[k]["s"]); y = np.concatenate(acc[k]["y"])
        auc = float(metrics.roc(s, y)["auc"])
        da, sr, nr = best_da(s, y)
        print(f"{k:22s} {auc:7.3f} {da*100:7.1f} {sr*100:7.1f} {nr*100:7.1f}")
    print("\nReference: DTSNN 86.2 DA (supervised), EDnCNN 81.0, AEDNet 82.4 (LED paper).")
    print("STCD here is ZERO-SHOT (no training); bar is closeness / does B3 help on real noise.")


if __name__ == "__main__":
    main()
