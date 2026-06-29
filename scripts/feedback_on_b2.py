"""Pairing test: does streaming-causal SSN feedback stack on the B2 spatial ceiling?

Load the trained B2 SpatialDenoiser (data/led/b2_spatial.pth), FREEZE it, wrap it in
FeedbackOnBase (SSN feedback into B2's own per-bin threshold decision), train ONLY the
feedback on LED, and compare B2-alone (feedback OFF) vs B2+feedback (ON) on the same 40
held-out slices at fixed dt=1ms. Reports AUC/DA/SR/NR and hard-contour recall (hardSR).

Env: STEPS(1200), PS(96), B(4), DT(1ms), N_POOL(40), N_EVAL(40), EVAL_EVERY(300), LR.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.events import events_to_tensor, TimeGrid, Events
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.learned import SpatialDenoiser, FeedbackOnBase

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
STEPS = int(os.environ.get("STEPS", "1200"))
PS = int(os.environ.get("PS", "96"))
B = int(os.environ.get("B", "4"))
DT = float(os.environ.get("DT", "1")) * 1e-3
N_POOL = int(os.environ.get("N_POOL", "40"))
N_EVAL = int(os.environ.get("N_EVAL", "40"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "300"))
LR = float(os.environ.get("LR", "1e-3"))
T = max(1, round(0.01 / DT))
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
print(f"device={dev} STEPS={STEPS} PS={PS} B={B} dt={DT*1e3}ms T={T}")


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


pool = load_pool("train", N_POOL)
eval_set = load_pool("test", N_EVAL)
print(f"train pool={len(pool)} eval={len(eval_set)}")


def patch_batch():
    xs, tg, mk = [], [], []
    tries = 0
    while len(xs) < B and tries < B * 20:
        tries += 1
        ev = pool[rng.integers(len(pool))]
        y0 = rng.integers(0, ev.H - PS); x0 = rng.integers(0, ev.W - PS)
        m = (ev.ys >= y0) & (ev.ys < y0 + PS) & (ev.xs >= x0) & (ev.xs < x0 + PS)
        if m.sum() < 50:
            continue
        pe = ev.select(m)
        pe = Events(pe.xs - x0, pe.ys - y0, pe.ts, pe.ps, H=PS, W=PS, labels=pe.labels)
        gg = TimeGrid(float(ev.ts.min()), DT, T)
        tot, _ = events_to_tensor(pe, grid=gg)
        sig, _ = events_to_tensor(pe.select(pe.labels), grid=gg)
        totc = tot.sum(0); sigc = sig.sum(0)
        xs.append(tot); tg.append((sigc / totc.clamp(min=1) >= 0.5).float()); mk.append(totc > 0)
    return (torch.stack(xs).to(dev), torch.stack(tg).to(dev), torch.stack(mk).to(dev))


@torch.no_grad()
def scores_of(model, use_feedback):
    S, Y = [], []
    model.eval()
    for ev in eval_set:
        g = TimeGrid(float(ev.ts.min()), DT, T)
        tot, _ = events_to_tensor(ev, grid=g)
        out = model(tot.unsqueeze(0).to(dev), use_feedback=use_feedback)[0, 0]
        yb = torch.from_numpy(g.bin_index(ev.ts)).long().to(dev)
        sc = out[torch.from_numpy(ev.ys).long().to(dev),
                 torch.from_numpy(ev.xs).long().to(dev), yb].cpu().numpy()
        S.append(sc); Y.append(ev.labels)
    model.train()
    return np.concatenate(S), np.concatenate(Y)


def best_da(s, y):
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 256)))
    P, N = y.sum(), (~y).sum()
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in thr:
        keep = s >= t
        da = 0.5 * ((keep & y).sum() / P + (~keep & ~y).sum() / N)
        if da > best[0]:
            best = (da, (keep & y).sum() / P, (~keep & ~y).sum() / N, t)
    return best


# baseline STCD support -> hard (under-supported real) subset for contour recall
fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=0.5, dt=DT))
base_sup = np.concatenate([fe.score_events(ev) for ev in eval_set])
y_all = np.concatenate([ev.labels for ev in eval_set])
hard = y_all & (base_sup <= np.quantile(base_sup[y_all], 0.25))


def report(tag, s, y):
    auc = float(metrics.roc(s, y)["auc"])
    da, sr, nr, thr = best_da(s, y)
    hard_sr = ((s >= thr) & hard).sum() / max(hard.sum(), 1)
    print(f"{tag:26s} AUC {auc:.4f}  DA {da*100:5.1f}  SR {sr*100:5.1f}  NR {nr*100:5.1f}  hardSR {hard_sr*100:5.1f}")
    return auc, da


# build frozen B2 base + SSN feedback
base = SpatialDenoiser()
ck = torch.load(os.path.join(ROOT, "b2_spatial.pth"), map_location=dev, weights_only=False)
base.load_state_dict(ck["state"])
model = FeedbackOnBase(base, backend="ssn", freeze_base=True).to(dev)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable params (feedback only) = {sum(p.numel() for p in trainable)}")

s, y = scores_of(model, use_feedback=False)
print("\n--- B2 alone (feedback OFF) ---")
report("B2 alone", s, y)
print("--- training SSN feedback on frozen B2 ---")

opt = torch.optim.Adam(trainable, lr=LR)
t0 = time.time()
for step in range(1, STEPS + 1):
    X, TG, MK = patch_batch()
    logit = model(X, use_feedback=True)[:, 0]
    loss = F.binary_cross_entropy_with_logits(logit[MK], TG[MK])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS:
        s, y = scores_of(model, use_feedback=True)
        auc = float(metrics.roc(s, y)["auc"]); da = best_da(s, y)[0]
        print(f"step {step:4d} loss {loss.item():.4f} AUC {auc:.4f} DA {da*100:.1f} "
              f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

print("\n=== FINAL (same 40 slices, dt=1ms) ===")
report("B2 alone (fb OFF)", *scores_of(model, use_feedback=False))
report("B2 + SSN feedback (fb ON)", *scores_of(model, use_feedback=True))
torch.save({"state": model.state_dict(), "model": "fb_on_b2", "head": "ssn"},
           os.path.join(ROOT, "fb_on_b2.pth"))
print("fb_on_b2 DONE")
