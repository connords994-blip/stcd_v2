"""Learn STCD's spatial kernel W on LED by unsupervised STDP, eval vs baseline.

Trains the k=5 unit-sum kernel from a centre-only ("delta") blind start on a raw
(unlabelled) LED stream, then evaluates the learned denoiser on a held-out LED
subset. Mirrors the paper's STDP experiment (0.927 blind -> 0.989 learned), now on LED.

Env: TAU (ms, default 8), DT (ms, default 1), N_TRAIN (consecutive slices, default 6),
N_EVAL (default 40), EPOCHS (default 40).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.events import Events
from stcd.stdp import STDPDenoiser, STDPConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
TAU = float(os.environ.get("TAU", "8")) * 1e-3
DT = float(os.environ.get("DT", "1")) * 1e-3            # eval dt (matches sweep)
TRAIN_DT = float(os.environ.get("TRAIN_DT", "4")) * 1e-3  # coarser → fewer bins → fits 6GB
N_TRAIN = int(os.environ.get("N_TRAIN", "6"))
N_EVAL = int(os.environ.get("N_EVAL", "40"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))


def best_da(s, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 200)))
    P, N_ = y.sum(), (~y).sum()
    return max(0.5 * ((s >= t) & y).sum() / P + 0.5 * ((s < t) & ~y).sum() / N_ for t in thr)


scenes = led.test_scenes(ROOT)
# Training stream: consecutive slices of ONE scene (timestamps already continuous).
tsc = scenes[0]
tfns = led.scene_slices(ROOT, tsc)[:N_TRAIN]
train = Events.concat(*[led.load_slice(ROOT, tsc, f) for f in tfns]).time_sorted()
print(f"train on {tsc} x{len(tfns)} slices: {len(train)} events; train_dt={TRAIN_DT*1e3}ms "
      f"eval_dt={DT*1e3}ms tau={TAU*1e3}ms")

# Held-out labelled eval subset (later scenes).
eval_evs = []
for sc in scenes[1:]:
    fns = led.scene_slices(ROOT, sc)
    if fns:
        ev = led.load_slice(ROOT, sc, fns[0])
        if len(ev) >= 100 and ev.labels.any() and not ev.labels.all():
            eval_evs.append(ev)
    if len(eval_evs) >= N_EVAL:
        break
mon = eval_evs[0]  # single slice for per-epoch monitoring

den = STDPDenoiser(STDPConfig(k=5, tau=TAU, dt=TRAIN_DT, theta=0.5, eta=0.02, epochs=EPOCHS),
                   init="delta")
hist = den.train_unsupervised(train, eval_ev=mon)
aucs = hist["auc"]
print(f"\nblind(delta) AUC={aucs[0]:.4f} -> learned AUC={aucs[-1]:.4f} (monitor slice, train_dt)")
print("epoch auc:", " ".join(f"{a:.3f}" for a in aucs[::max(1, EPOCHS // 10)]))

den.cfg.dt = DT  # switch to eval dt (kernel W is what we learned; leak just uses dt)
# Full eval on the held-out subset.
S, Y = [], []
for ev in eval_evs:
    S.append(den.score_events(ev)); Y.append(ev.labels)
s = np.concatenate(S); y = np.concatenate(Y)
print(f"\nSTDP-learned kernel on {len(eval_evs)} LED slices: AUC={float(metrics.roc(s,y)['auc']):.4f}  maxDA={best_da(s,y)*100:.1f}")
K = den.kernel()  # [P,k,k]
np.set_printoptions(precision=3, suppress=True)
print("learned kernel (ON polarity), unit-sum:\n", K[1])
print("centre weight ON:", K[1, 2, 2], " surround sum:", K[1].sum() - K[1, 2, 2])
