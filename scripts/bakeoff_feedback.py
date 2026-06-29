"""Same-harness bake-off for the Idea-2 output->threshold feedback denoisers.

Evaluates every available model on the SAME N_EVAL LED-test slices at ONE fixed dt
(so AUC/DA are directly comparable across rows; absolute AUC is dt-dependent, see
RESULTS), and reports:
  * AUC, best-DA, and the SR/NR at that operating point
  * feedback ablation (fb models run with use_feedback=False = bare frozen STCD core)
  * HARD signal recall: SR on the under-supported real events (bottom-quartile baseline
    STCD support among signal events) -- the contour events feedback should rescue
  * NR is the hallucination guard (lifting hard-SR while tanking NR = hallucinated edges)

Env: DT(ms, default 1), N_EVAL(default 40).
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.events import events_to_tensor, TimeGrid
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.learned import DynThreshSNN, SpatialDenoiser, CombinedDenoiser, FeedbackDenoiser

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
DT = float(os.environ.get("DT", "1")) * 1e-3
N_EVAL = int(os.environ.get("N_EVAL", "40"))
T = max(1, round(0.01 / DT))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} dt={DT*1e3}ms T={T} N_EVAL={N_EVAL}")


def load_pool(split, n):
    out = []
    for sc in led.scenes(ROOT, split):
        for fn in led.scene_slices(ROOT, sc, split)[:2]:
            ev = led.load_slice(ROOT, sc, fn, split)
            if len(ev) >= 200 and ev.labels.any() and not ev.labels.all():
                out.append(ev)
            if len(out) >= n:
                return out
    return out


eval_set = load_pool("test", N_EVAL)
print(f"eval slices = {len(eval_set)}")


@torch.no_grad()
def score_learned(model, ev, use_feedback=None):
    g = TimeGrid(float(ev.ts.min()), DT, T)
    tot, _ = events_to_tensor(ev, grid=g)
    x = tot.unsqueeze(0).to(dev)
    out = model(x, use_feedback=use_feedback) if use_feedback is not None else model(x)
    out = out[0, 0]
    yb = torch.from_numpy(g.bin_index(ev.ts)).long().to(dev)
    return out[torch.from_numpy(ev.ys).long().to(dev),
               torch.from_numpy(ev.xs).long().to(dev), yb].cpu().numpy()


def best_da(s, y):
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    P, N = y.sum(), (~y).sum()
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in thr:
        keep = s >= t
        sr = (keep & y).sum() / P
        nr = (~keep & ~y).sum() / N
        da = 0.5 * (sr + nr)
        if da > best[0]:
            best = (da, sr, nr, t)
    return best  # da, sr, nr, thr


def load_ckpt(name):
    path = os.path.join(ROOT, f"{name}.pth")
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=dev, weights_only=False)
    m, head = ck.get("model", "b1"), ck.get("head", "theta")
    if m == "b2":
        net = SpatialDenoiser()
    elif m == "combined":
        net = CombinedDenoiser()
    elif m in ("fb_snn", "fb_ssn"):
        net = FeedbackDenoiser(backend=m.split("_")[1])
    else:
        net = DynThreshSNN(C=16, head=head)
    net.load_state_dict(ck["state"])
    return net.to(dev).eval()


# --- baseline STCD per-event support (defines the hard/contour subset) ---
fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=0.5, dt=DT))
base_S, Y = [], []
for ev in eval_set:
    base_S.append(fe.score_events(ev)); Y.append(ev.labels)
base = np.concatenate(base_S); y = np.concatenate(Y)
# hard signal subset = under-supported real events (bottom quartile baseline support)
sig_mask = y
q = np.quantile(base[sig_mask], 0.25)
hard = sig_mask & (base <= q)
print(f"events={len(y)}  signal={int(y.sum())}  noise={int((~y).sum())}  "
      f"hard(under-supported) signal={int(hard.sum())}  (baseline support <= {q:.3f})")


def report(name, s):
    auc = float(metrics.roc(s, y)["auc"])
    da, sr, nr, thr = best_da(s, y)
    keep = s >= thr
    hard_sr = (keep & hard).sum() / max(hard.sum(), 1)
    print(f"{name:24s} AUC {auc:.4f}  DA {da*100:5.1f}  SR {sr*100:5.1f}  NR {nr*100:5.1f}"
          f"  hardSR {hard_sr*100:5.1f}")
    return auc, da


rows = []
rows.append(("baseline STCD (frontend)", report("baseline STCD", base)))

for name in ["b1_theta", "b1_scatter", "b2_spatial", "combined", "fb_snn", "fb_ssn"]:
    net = load_ckpt(name)
    if net is None:
        print(f"{name:24s} (no checkpoint)")
        continue
    if isinstance(net, FeedbackDenoiser):
        s_off = np.concatenate([score_learned(net, ev, use_feedback=False) for ev in eval_set])
        report(name + " (fb OFF)", s_off)
        s_on = np.concatenate([score_learned(net, ev, use_feedback=True) for ev in eval_set])
        rows.append((name, report(name + " (fb ON)", s_on)))
    else:
        rows.append((name, report(name, np.concatenate([score_learned(net, ev) for ev in eval_set]))))

print("\nNote: DTSNN (same-harness, RESULTS 2026-06-28) = 0.926 AUC / 89.2 maxDA.")
print("All rows above at fixed dt; B1/B2 headline (0.946/87.3, 0.970/91.0) were measured at dt=2ms.")
