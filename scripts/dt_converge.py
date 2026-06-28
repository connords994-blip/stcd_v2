"""dt-convergence check: baseline STCD AUC vs time-bin width on a fixed LED subset.

dt is the dense-tensor discretization (leak α=exp(-dt/τ)); the deployed STCD is
continuous. We want the largest dt whose AUC has stopped changing (cheapest faithful
value) to FIX before sweeping the real params (τ, θ, δ, W)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.frontend import SpikingFrontEnd, FrontEndConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
N = int(os.environ.get("N", "24"))
DTS = [5e-3, 2e-3, 1e-3, 5e-4]

# Fixed slice set (1 per scene for diversity).
slices = []
for sc in led.test_scenes(ROOT):
    fns = led.scene_slices(ROOT, sc)
    if fns:
        slices.append((sc, fns[0]))
    if len(slices) >= N:
        break
print(f"convergence on {len(slices)} LED slices (tau=8ms, theta=1.5, k=3)\n")
print(f"{'dt(ms)':>7s} {'T~':>4s} {'AUC':>8s}")
prev = None
for dt in DTS:
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=dt))
    S, Y = [], []
    for sc, fn in slices:
        ev = led.load_slice(ROOT, sc, fn)
        if len(ev) < 100 or ev.labels.all() or not ev.labels.any():
            continue
        S.append(fe.score_events(ev)); Y.append(ev.labels)
    s = np.concatenate(S); y = np.concatenate(Y)
    auc = float(metrics.roc(s, y)["auc"])
    d = "" if prev is None else f"  Δ={auc-prev:+.4f}"
    print(f"{dt*1e3:7.1f} {int(round(0.01/dt)):4d} {auc:8.4f}{d}")
    prev = auc
