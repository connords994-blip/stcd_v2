"""Scalar sweep on LED at fixed dt=1ms: baseline τ×θ grid, then B3 δ at the best τ/θ.

dt fixed (see dt_converge.py: AUC is dt-sensitive, so we hold it constant and read
RELATIVE rankings, which are dt-robust). Slices pre-loaded once and reused.
Env: N (slices, default 40)."""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.frontend import SpikingFrontEnd, FrontEndConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
N = int(os.environ.get("N", "40"))
DT = 1e-3
TAUS = [2e-3, 4e-3, 8e-3, 16e-3, 32e-3]
THETAS = [0.5, 1.0, 1.5, 2.0, 3.0]
DELTAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def best_da(s, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 200)))
    P, N_ = y.sum(), (~y).sum()
    best = -1.0
    for t in thr:
        keep = s >= t
        da = 0.5 * ((keep & y).sum() / P + (~keep & ~y).sum() / N_)
        best = max(best, da)
    return best


# Pre-load a fixed slice set once.
evs = []
for sc in led.test_scenes(ROOT):
    fns = led.scene_slices(ROOT, sc)
    if not fns:
        continue
    ev = led.load_slice(ROOT, sc, fns[0])
    if len(ev) >= 100 and ev.labels.any() and not ev.labels.all():
        evs.append(ev)
    if len(evs) >= N:
        break
print(f"preloaded {len(evs)} LED slices; dt={DT*1e3}ms\n")


def evaluate(cfg_extra):
    fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, dt=DT, **cfg_extra))
    S, Y = [], []
    for ev in evs:
        S.append(fe.score_events(ev)); Y.append(ev.labels)
    s = np.concatenate(S); y = np.concatenate(Y)
    return float(metrics.roc(s, y)["auc"]), best_da(s, y)


print("=== baseline τ×θ grid (AUC | maxDA%) ===")
print("tau\\theta " + "".join(f"{th:>13.1f}" for th in THETAS))
best = (-1.0, None)
for tau in TAUS:
    row = []
    for th in THETAS:
        auc, da = evaluate(dict(tau=tau, theta=th))
        row.append(f"{auc:.3f}/{da*100:.1f}")
        if auc > best[0]:
            best = (auc, dict(tau=tau, theta=th, auc=auc, da=da))
    print(f"{tau*1e3:6.0f}ms " + "".join(f"{c:>13s}" for c in row))

b = best[1]
print(f"\nbest baseline: tau={b['tau']*1e3:.0f}ms theta={b['theta']} -> AUC={b['auc']:.3f} maxDA={b['da']*100:.1f}")

print("\n=== B3 δ sweep at best τ/θ (AUC | maxDA%) ===")
print(f"{'delta':>7s} {'scatter':>14s} {'gather':>14s}")
for d in DELTAS:
    sca = evaluate(dict(tau=b['tau'], theta=b['theta'], b3=True, b3_side='scatter',
                        b3_update='raw', b3_clamp_nonneg=False, b3_delta=d))
    gat = evaluate(dict(tau=b['tau'], theta=b['theta'], b3=True, b3_side='gather',
                        b3_update='raw', b3_clamp_nonneg=False, b3_delta=d))
    print(f"{d:7.2f} {sca[0]:.3f}/{sca[1]*100:4.1f}   {gat[0]:.3f}/{gat[1]*100:4.1f}")
print(f"\n(baseline at same τ/θ: AUC={b['auc']:.3f} maxDA={b['da']*100:.1f}) — B3 must beat this to earn its keep")
