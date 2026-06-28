"""Two B3 stress tests on synthetic data:

A. Recall stress (#11) — a translating checkerboard is dense in corners where a
   real pixel out-fires its neighbours. B3's anti-Hebbian discount risks dropping
   exactly these. Report signal-retain (SR) + AUC; a big SR drop = false-negative tax.

B. 1b fixed vs roaming (#12) — a persistent hot line (fixed) vs a roaming arbiter
   (fresh line each burst, low per-pixel rate). Expect: rate_cap & B3 kill fixed but
   MISS roaming (no persistent self-firer) -> roaming needs a column-aggregate
   detector (Strategy 8). Honest scope-finding for B3.
"""
import numpy as np
from stcd.synth import SynthConfig, generate, KIND_COLUMN
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd import metrics
from stcd.baselines import rate_cap

BASE = dict(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3)
VARS = {
    "baseline":           dict(),
    "B3 scatter/raw/neg": dict(b3=True, b3_side="scatter", b3_update="raw", b3_clamp_nonneg=False),
    "B3 gather/raw/neg":  dict(b3=True, b3_side="gather",  b3_update="raw", b3_clamp_nonneg=False),
}

print("=== A. RECALL STRESS — translating checkerboard (corner-rich signal) ===")
cfg = SynthConfig(H=120, W=160, duration=0.3, scene="checker", checker_cell=4,
                  num_objects=1, vx_range=(0.3, 0.4), vy_range=(0.1, 0.2),
                  noise_rate_hz=1.0, n_hot_pixels=0, n_columns=0)
ev = generate(cfg)
print(f"checker events={len(ev)}  signal_frac={ev.labels.mean():.3f}")
print(f"{'variant':22s} {'AUC':>6s} {'SR':>7s} {'NR':>7s}")
for name, extra in VARS.items():
    fe = SpikingFrontEnd(FrontEndConfig(**BASE, **extra))
    scores = fe.score_events(ev); kept, _ = fe.filter(ev)
    auc = float(metrics.roc(scores, ev.labels)["auc"])
    m = metrics.evaluate_filter(ev.labels, kept)
    print(f"{name:22s} {auc:6.3f} {m.signal_retain:7.3f} {m.noise_removal:7.3f}")
print("(SR drop under B3 vs baseline = real corner/texture signal lost to the discount)\n")

print("=== B. 1b FIXED vs ROAMING — column noise (NR on the 1b subset) ===")
for mode in ("fixed", "roaming"):
    cfg = SynthConfig(H=120, W=160, duration=0.3, scene="bars", noise_rate_hz=1.0,
                      n_hot_pixels=0, n_columns=8, column_mode=mode,
                      column_rate_hz=120.0, column_frac=0.5, column_span_frac=0.7)
    ev = generate(cfg)
    col = int((ev.kinds == KIND_COLUMN).sum())
    # per-pixel max rate of the column noise (drives whether rate_cap can see it)
    colmask = ev.kinds == KIND_COLUMN
    cnt = np.zeros((ev.H, ev.W), np.int64)
    np.add.at(cnt, (ev.ys[colmask], ev.xs[colmask]), 1)
    ppmax = cnt.max() / (ev.duration or 1.0)
    print(f"\n[{mode}] column events={col}  peak per-pixel rate={ppmax:.0f} Hz "
          f"(rate_cap threshold 50 Hz)")
    print(f"  {'variant':22s} {'1b NR':>7s} {'1b SR':>7s}")
    for name, extra in VARS.items():
        fe = SpikingFrontEnd(FrontEndConfig(**BASE, **extra))
        kept, _ = fe.filter(ev)
        m = metrics.kind_subset_metrics(ev.labels, kept, ev.kinds, [KIND_COLUMN])
        print(f"  {name:22s} {m['noise_removal']:7.3f} {m['signal_retain']:7.3f}")
    rc_kept, _ = rate_cap(ev, max_rate_hz=50.0)
    mrc = metrics.kind_subset_metrics(ev.labels, rc_kept, ev.kinds, [KIND_COLUMN])
    print(f"  {'rate_cap(50Hz)':22s} {mrc['noise_removal']:7.3f} {mrc['signal_retain']:7.3f}")
print("\nSTRESS OK")
