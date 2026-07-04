"""Targeted test: does a spiking U-Net's multi-scale hierarchy catch ROAMING
correlated-readout (1b) noise that B2's flat dilated convs (and per-pixel methods) miss?

Train B2 (SpatialDenoiser) and SpikingUNet from scratch on synthetic streams =
moving-bar signal + BA + roaming column noise (each burst hits a fresh random line, so
per-pixel rate stays low -> only a column-AGGREGATE view should catch it). Identical
setup for both models. Report overall AUC/DA and, crucially, **NR on the 1b column
subset** (kind_subset_metrics over KIND_COLUMN) at the best-DA operating point. Baseline
single-neuron STCD included as the reference it's supposed to fail on.

Env: MODE(roaming|fixed), STEPS(800), DT(ms,30), NTRAIN(24), NEVAL(8), B(8), LR.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.synth import SynthConfig, generate, KIND_COLUMN
from stcd.events import events_to_tensor, TimeGrid
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.learned import SpatialDenoiser, SpikingUNet

MODE = os.environ.get("MODE", "roaming")
STEPS = int(os.environ.get("STEPS", "800"))
DT = float(os.environ.get("DT", "30")) * 1e-3
NTRAIN = int(os.environ.get("NTRAIN", "24"))
NEVAL = int(os.environ.get("NEVAL", "8"))
B = int(os.environ.get("B", "8"))
LR = float(os.environ.get("LR", "1e-3"))
DUR = float(os.environ.get("DUR", "0.3"))
T = max(1, round(DUR / DT))
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
print(f"device={dev} MODE={MODE} STEPS={STEPS} dt={DT*1e3}ms T={T} ntrain={NTRAIN} neval={NEVAL}")


def make(seed):
    cfg = SynthConfig(H=120, W=160, duration=DUR, scene="bars", num_objects=3,
                      noise_rate_hz=1.0, n_columns=8, column_mode=MODE,
                      column_rate_hz=120.0, column_frac=0.5, column_span_frac=0.7, seed=seed)
    ev = generate(cfg)
    g = TimeGrid(float(ev.ts.min()), DT, T)
    tot, _ = events_to_tensor(ev, grid=g)
    sig, _ = events_to_tensor(ev.select(ev.labels), grid=g)
    totc, sigc = tot.sum(0), sig.sum(0)
    target = (sigc / totc.clamp(min=1) >= 0.5).float()
    return dict(ev=ev, g=g, x=tot, tgt=target, mask=totc > 0)


print("generating synth pools...")
train = [make(s) for s in range(NTRAIN)]
evals = [make(1000 + s) for s in range(NEVAL)]
colfrac = np.mean([(e["ev"].kinds == KIND_COLUMN).mean() for e in evals])
print(f"eval column(1b) event fraction ~ {colfrac:.3f}")


def batch():
    idx = rng.integers(0, NTRAIN, B)
    X = torch.stack([train[i]["x"] for i in idx]).to(dev)
    TG = torch.stack([train[i]["tgt"] for i in idx]).to(dev)
    MK = torch.stack([train[i]["mask"] for i in idx]).to(dev)
    return X, TG, MK


@torch.no_grad()
def eval_model(model):
    model.eval()
    S, Y, K = [], [], []
    for e in evals:
        ev, g = e["ev"], e["g"]
        out = model(e["x"].unsqueeze(0).to(dev))[0, 0]
        yb = torch.from_numpy(g.bin_index(ev.ts)).long().to(dev)
        sc = out[torch.from_numpy(ev.ys).long().to(dev),
                 torch.from_numpy(ev.xs).long().to(dev), yb].cpu().numpy()
        S.append(sc); Y.append(ev.labels); K.append(ev.kinds)
    model.train()
    s, y, k = np.concatenate(S), np.concatenate(Y), np.concatenate(K)
    auc = float(metrics.roc(s, y)["auc"])
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    P, N = y.sum(), (~y).sum()
    best = (-1, 0.0, None)
    for t in thr:
        keep = s >= t
        da = 0.5 * ((keep & y).sum() / P + (~keep & ~y).sum() / N)
        if da > best[0]:
            best = (da, t, keep)
    da, t, keep = best
    col = metrics.kind_subset_metrics(y, keep, k, [KIND_COLUMN], scores=s)
    return auc, da, col


def train_model(model, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        X, TG, MK = batch()
        loss = F.binary_cross_entropy_with_logits(model(X)[:, 0][MK], TG[MK])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (STEPS // 3) == 0 or step == STEPS:
            auc, da, col = eval_model(model)
            print(f"  {name} step {step:4d} loss {loss.item():.3f}  AUC {auc:.4f} DA {da*100:.1f}"
                  f"  1b-NR {col['noise_removal']*100:.1f} 1b-AUC {col.get('auc', float('nan')):.3f}"
                  f"  ({(time.time()-t0)/step:.2f}s/st)", flush=True)
    return eval_model(model)


# baseline single-neuron STCD (the reference it should fail on for roaming 1b)
fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=DT))
S, Y, K = [], [], []
for e in evals:
    ev = e["ev"]
    S.append(fe.score_events(ev)); Y.append(ev.labels); K.append(ev.kinds)
s, y, k = np.concatenate(S), np.concatenate(Y), np.concatenate(K)
auc = float(metrics.roc(s, y)["auc"])
thr = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
P, N = y.sum(), (~y).sum()
bda, bt, bk = max((0.5 * ((s >= t) & y).sum() / P + 0.5 * ((s < t) & ~y).sum() / N, t, s >= t) for t in thr)
bcol = metrics.kind_subset_metrics(y, bk, k, [KIND_COLUMN], scores=s)

print(f"\n=== RESULTS ({MODE} column-1b, {NEVAL} held-out synth) ===")
print(f"{'model':16s} {'AUC':>7s} {'DA':>6s} {'1b-NR':>7s} {'1b-AUC':>7s}")
print(f"{'STCD baseline':16s} {auc:7.4f} {bda*100:6.1f} {bcol['noise_removal']*100:7.1f} {bcol.get('auc', float('nan')):7.3f}")
for cls, nm in [(SpatialDenoiser, "B2 spatial"), (SpikingUNet, "spiking U-Net")]:
    torch.manual_seed(0)
    a, d, c = train_model(cls().to(dev), nm)
    print(f"{nm:16s} {a:7.4f} {d*100:6.1f} {c['noise_removal']*100:7.1f} {c.get('auc', float('nan')):7.3f}")
print("1b-NR = noise-removal on the roaming-column subset (higher = catches the 1b noise).")
print("ROAMING_1B DONE")
