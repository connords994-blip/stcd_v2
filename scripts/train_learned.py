"""Train the B1 dynamic-threshold SNN on LED (from scratch) and eval on LED test.

Patch-based training (random HxW crops) to fit 8GB VRAM at 1280x720; coarse train dt
for tractable BPTT; full-frame eval. Env: MODEL(b1|b2|combined|fb_snn|fb_ssn),
HEAD(theta|scatter), STEPS, PS(patch), B(batch), DT(ms), N_POOL(train slices),
N_EVAL(test slices), EVAL_EVERY, LR.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.datasets import led
from stcd.events import events_to_tensor, TimeGrid
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.learned import DynThreshSNN, SpatialDenoiser, CombinedDenoiser, FeedbackDenoiser

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "led")
MODEL = os.environ.get("MODEL", "b1")   # b1=dyn-threshold SNN, b2=spatial recombiner, combined=B1+B2, fb_snn/fb_ssn=feedback
HEAD = os.environ.get("HEAD", "theta")
STEPS = int(os.environ.get("STEPS", "400"))
PS = int(os.environ.get("PS", "160"))
B = int(os.environ.get("B", "8"))
DT = float(os.environ.get("DT", "2")) * 1e-3
N_POOL = int(os.environ.get("N_POOL", "40"))
N_EVAL = int(os.environ.get("N_EVAL", "24"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "100"))
LR = float(os.environ.get("LR", "1e-3"))
T = max(1, round(0.01 / DT))           # bins per 10ms slice
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
print(f"device={dev} model={MODEL} head={HEAD} steps={STEPS} PS={PS} B={B} dt={DT*1e3}ms T={T} pool={N_POOL}")


def load_pool(split, n):
    out = []
    for sc in led.scenes(ROOT, split):
        for fn in led.scene_slices(ROOT, sc, split)[:2]:
            ev = led.load_slice(ROOT, sc, fn, split)
            if len(ev) >= 200 and ev.labels.any() and not ev.labels.all():
                g = TimeGrid(float(ev.ts.min()), DT, T)
                out.append((ev, g))
            if len(out) >= n:
                return out
    return out


print("loading pools...")
pool = load_pool("train", N_POOL)
eval_set = load_pool("test", N_EVAL)
print(f"train pool={len(pool)} slices, eval={len(eval_set)} slices")


def patch_batch():
    xs, tg, mk = [], [], []
    tries = 0
    while len(xs) < B and tries < B * 20:
        tries += 1
        ev, g = pool[rng.integers(len(pool))]
        y0 = rng.integers(0, ev.H - PS); x0 = rng.integers(0, ev.W - PS)
        m = (ev.ys >= y0) & (ev.ys < y0 + PS) & (ev.xs >= x0) & (ev.xs < x0 + PS)
        if m.sum() < 50:
            continue
        pe = ev.select(m)
        from stcd.events import Events
        pe = Events(pe.xs - x0, pe.ys - y0, pe.ts, pe.ps, H=PS, W=PS, labels=pe.labels)
        gg = TimeGrid(g.t0, DT, T)
        tot, _ = events_to_tensor(pe, grid=gg)
        sig, _ = events_to_tensor(pe.select(pe.labels), grid=gg)
        totc = tot.sum(0); sigc = sig.sum(0)
        mask = totc > 0
        target = (sigc / totc.clamp(min=1) >= 0.5).float()
        xs.append(tot); tg.append(target); mk.append(mask)
    X = torch.stack(xs).to(dev)              # [B,2,PS,PS,T]
    TG = torch.stack(tg).to(dev)             # [B,PS,PS,T]
    MK = torch.stack(mk).to(dev)
    return X, TG, MK


@torch.no_grad()
def evaluate(model):
    model.eval()
    S, Y = [], []
    for ev, _ in eval_set:
        g = TimeGrid(float(ev.ts.min()), DT, T)
        tot, _ = events_to_tensor(ev, grid=g)
        x = tot.unsqueeze(0).to(dev)
        out = model(x)[0, 0]                  # [H,W,T]
        yb = torch.from_numpy(g.bin_index(ev.ts)).long().to(dev)
        sc = out[torch.from_numpy(ev.ys).long().to(dev),
                 torch.from_numpy(ev.xs).long().to(dev), yb].cpu().numpy()
        S.append(sc); Y.append(ev.labels)
    model.train()
    s = np.concatenate(S); y = np.concatenate(Y)
    auc = float(metrics.roc(s, y)["auc"])
    thr = np.unique(np.quantile(s, np.linspace(0, 1, 200)))
    P, N_ = y.sum(), (~y).sum()
    da = max(0.5 * ((s >= t) & y).sum() / P + 0.5 * ((s < t) & ~y).sum() / N_ for t in thr)
    return auc, da


_models = {"b2": lambda: SpatialDenoiser(), "combined": lambda: CombinedDenoiser(),
           "b1": lambda: DynThreshSNN(C=16, head=HEAD),
           "fb_snn": lambda: FeedbackDenoiser(backend="snn"),
           "fb_ssn": lambda: FeedbackDenoiser(backend="ssn")}
model = _models.get(MODEL, _models["b1"])().to(dev)
tag = {"b2": "b2_spatial", "combined": "combined",
       "fb_snn": "fb_snn", "fb_ssn": "fb_ssn"}.get(MODEL, f"b1_{HEAD}")
print(f"model={MODEL} ({tag})")
opt = torch.optim.Adam(model.parameters(), lr=LR)
print(f"init eval: AUC/DA = {evaluate(model)}")
t0 = time.time()
for step in range(1, STEPS + 1):
    X, TG, MK = patch_batch()
    logit = model(X)[:, 0]
    loss = F.binary_cross_entropy_with_logits(logit[MK], TG[MK])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS:
        auc, da = evaluate(model)
        print(f"step {step:4d}  loss {loss.item():.4f}  test AUC {auc:.4f} DA {da*100:.1f}  "
              f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

torch.save({"state": model.state_dict(), "model": MODEL, "head": HEAD},
           os.path.join(ROOT, f"{tag}.pth"))
# baseline reference on the same eval set
fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=0.5, dt=1e-3))
S, Y = [], []
for ev, _ in eval_set:
    S.append(fe.score_events(ev)); Y.append(ev.labels)
b_auc = float(metrics.roc(np.concatenate(S), np.concatenate(Y))["auc"])
auc, da = evaluate(model)
print(f"\n{tag} LED test: AUC={auc:.4f} DA={da*100:.1f} | baseline STCD AUC={b_auc:.4f} | DTSNN 86.2 DA")
print(f"{tag} LED DONE")
