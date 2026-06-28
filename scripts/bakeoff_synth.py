"""B3 bake-off smoke: baseline STCD vs the three anti-Hebbian toggle axes,
scored separately on 1a (hot) and 1b (column) injected noise."""
import numpy as np
from stcd.synth import SynthConfig, generate, KIND_HOT, KIND_COLUMN
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd import metrics

cfg = SynthConfig(H=120, W=160, duration=0.3, noise_rate_hz=1.0,
                  n_hot_pixels=12, hot_pixel_rate_hz=300.0,
                  n_columns=5, column_rate_hz=90.0, column_frac=0.6, column_span_frac=0.8)
ev = generate(cfg)
print(f"events={len(ev)}  signal_frac={ev.labels.mean():.3f}  "
      f"hot={(ev.kinds==KIND_HOT).sum()}  col={(ev.kinds==KIND_COLUMN).sum()}\n")

base = dict(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=5e-3)
variants = {
    "baseline":             dict(),
    "B3 scatter/raw/neg":   dict(b3=True, b3_side="scatter", b3_update="raw",   b3_clamp_nonneg=False),
    "B3 scatter/raw/>=0":   dict(b3=True, b3_side="scatter", b3_update="raw",   b3_clamp_nonneg=True),
    "B3 gather/raw/neg":    dict(b3=True, b3_side="gather",  b3_update="raw",   b3_clamp_nonneg=False),
    "B3 scatter/out/neg":   dict(b3=True, b3_side="scatter", b3_update="output", b3_clamp_nonneg=False),
}

hdr = f"{'variant':22s} {'AUC':>6s} | {'1a NR':>6s} {'1a SR':>6s} | {'1b NR':>6s} {'1b SR':>6s}"
print(hdr); print("-" * len(hdr))
for name, extra in variants.items():
    fe = SpikingFrontEnd(FrontEndConfig(**base, **extra))
    scores = fe.score_events(ev)
    kept, _ = fe.filter(ev)
    auc = float(metrics.roc(scores, ev.labels)["auc"])
    m1a = metrics.kind_subset_metrics(ev.labels, kept, ev.kinds, [KIND_HOT], scores)
    m1b = metrics.kind_subset_metrics(ev.labels, kept, ev.kinds, [KIND_COLUMN], scores)
    print(f"{name:22s} {auc:6.3f} | {m1a['noise_removal']:6.3f} {m1a['signal_retain']:6.3f} | "
          f"{m1b['noise_removal']:6.3f} {m1b['signal_retain']:6.3f}")
print("\n(want: B3 raises 1a NR vs baseline while keeping SR high; allow-neg should help 1b NR)")
print("B3 OK")
