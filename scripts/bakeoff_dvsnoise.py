"""DVSNOISE20 no-regression check for B3.

Mirrors repro_stcd_auc.py (same loader, window pick, APS-motion labels, seed-0
balanced sampling) but scores baseline STCD against B3 variants. DVSNOISE20's noise
is mostly isolated BA (not hot pixels), so B3 has little to suppress here — the goal
is to confirm it does NOT regress the ~0.797 AUC we already have.

EDN_SCENES=bike,classroom,conference,soccer,stairs,toys for the paper subset.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import dvsnoise20 as dv
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.downstream.edncnn_real import NEIGH

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "dvsnoise20", "2_mat")
WIN = 1.0
PER_CLASS = 400
K = 3

BASE = dict(neighbor_k=K, pool=1, tau=8e-3, theta=1.5, dt=5e-3)
VARIANTS = {
    "baseline":           dict(),
    "B3 scatter/raw/neg": dict(b3=True, b3_side="scatter", b3_update="raw", b3_clamp_nonneg=False),
    "B3 gather/raw/neg":  dict(b3=True, b3_side="gather",  b3_update="raw", b3_clamp_nonneg=False),
}


def scenes():
    want = [s for s in os.environ.get("EDN_SCENES", "").split(",") if s]
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*.mat"))):
        name = os.path.basename(p).split("-")[0]
        if want and name not in want:
            continue
        out.append((os.path.basename(p)[:-4], name, p))
    return out


def score_one(label: str, path: str, fes: dict) -> dict | None:
    """Score one recording; returns {method: auc} or None if too few samples.
    Isolated so a driver can run one recording per subprocess (bounds peak RAM
    against the 2.5 GB .mat loads on the 6 GB WSL cap)."""
    methods = list(fes)
    ev, fts, aps, _ = dv.load_full(path)
    t0 = ev.ts.min(); best, bn = t0, -1
    for a in np.arange(t0, ev.ts.max() - WIN, WIN):
        n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
        if n > bn:
            best, bn = a, n
    w = ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()
    sig, val = dv.aps_motion_labels(w, fts, aps)
    notedge = (w.xs >= NEIGH) & (w.xs < w.W - NEIGH) & (w.ys >= NEIGH) & (w.ys < w.H - NEIGH)
    late = w.ts >= (best + 0.2)
    cand = np.where(val & notedge & late)[0]
    rng = np.random.default_rng(0)
    pos, neg = cand[sig[cand]], cand[~sig[cand]]
    m = min(len(pos), len(neg), PER_CLASS)
    if m < 50:
        return None
    q = np.sort(np.concatenate([rng.choice(pos, m, replace=False),
                                rng.choice(neg, m, replace=False)]))
    lab = sig[q]
    return {k: float(metrics.roc(fe.score_events(w)[q], lab)["auc"]) for k, fe in fes.items()}


def main():
    fes = {k: SpikingFrontEnd(FrontEndConfig(**BASE, **extra)) for k, extra in VARIANTS.items()}
    methods = list(VARIANTS)
    # Single-recording mode: `python bakeoff_dvsnoise.py <path.mat>` -> one ROW line.
    if len(sys.argv) > 1:
        path = sys.argv[1]
        label = os.path.basename(path)[:-4]
        r = score_one(label, path, fes)
        if r is None:
            print(f"ROW\t{label}\tSKIP", flush=True)
        else:
            print("ROW\t" + label + "\t" + "\t".join(f"{k}={r[k]:.4f}" for k in methods), flush=True)
        return
    records = []
    for label, name, path in scenes():
        ev, fts, aps, _ = dv.load_full(path)
        t0 = ev.ts.min(); best, bn = t0, -1
        for a in np.arange(t0, ev.ts.max() - WIN, WIN):
            n = int(((ev.ts >= a) & (ev.ts < a + WIN)).sum())
            if n > bn:
                best, bn = a, n
        w = ev.select((ev.ts >= best) & (ev.ts < best + WIN)).time_sorted()
        sig, val = dv.aps_motion_labels(w, fts, aps)
        notedge = (w.xs >= NEIGH) & (w.xs < w.W - NEIGH) & (w.ys >= NEIGH) & (w.ys < w.H - NEIGH)
        late = w.ts >= (best + 0.2)
        cand = np.where(val & notedge & late)[0]
        rng = np.random.default_rng(0)
        pos, neg = cand[sig[cand]], cand[~sig[cand]]
        m = min(len(pos), len(neg), PER_CLASS)
        if m < 50:
            print(f"  {label}: too few samples; skip", flush=True)
            continue
        q = np.sort(np.concatenate([rng.choice(pos, m, replace=False),
                                    rng.choice(neg, m, replace=False)]))
        lab = sig[q]
        aucs = {k: float(metrics.roc(fe.score_events(w)[q], lab)["auc"]) for k, fe in fes.items()}
        records.append(aucs)
        print(f"  {label:34s} " + "  ".join(f"{k}={aucs[k]:.3f}" for k in methods), flush=True)
    if not records:
        print("No recordings scored."); return
    n = len(records)
    A = {k: np.array([r[k] for r in records]) for k in methods}
    print(f"\nn={n} recordings    AUC (mean +/- 95% CI)")
    for k in methods:
        mean = A[k].mean(); ci = 1.96 * A[k].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"  {k:20s} {mean:.3f} +/- {ci:.3f}")
    d = A["B3 scatter/raw/neg"].mean() - A["baseline"].mean()
    print(f"\nB3 scatter vs baseline delta = {d:+.3f}  (>= -0.01 ⇒ no meaningful regression)")
    print("Paper baseline STCD: 0.797")


if __name__ == "__main__":
    main()
