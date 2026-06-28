"""B1 smoke trainer on SYNTHETIC labelled data — validates the dynamic-threshold SNN
learns (loss down, AUC up) before wiring LED patches. Env: HEAD (theta|scatter), STEPS."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stcd import metrics
from stcd.synth import SynthConfig, generate
from stcd.events import events_to_tensor
from stcd.frontend import SpikingFrontEnd, FrontEndConfig
from stcd.learned import DynThreshSNN

HEAD = os.environ.get("HEAD", "theta")
STEPS = int(os.environ.get("STEPS", "300"))
DT = 5e-3
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} head={HEAD} steps={STEPS}")


def make(seed):
    cfg = SynthConfig(H=120, W=160, duration=0.3, seed=seed, noise_rate_hz=1.0,
                      n_hot_pixels=10, hot_pixel_rate_hz=300.0,
                      n_columns=5, column_rate_hz=90.0, column_frac=0.6)
    ev = generate(cfg)
    tensor, grid = events_to_tensor(ev, dt=DT)
    x = tensor.unsqueeze(0).to(dev)                       # [1,2,H,W,T]
    yb = torch.from_numpy(grid.bin_index(ev.ts)).long().to(dev)
    yy = torch.from_numpy(ev.ys).long().to(dev)
    xx = torch.from_numpy(ev.xs).long().to(dev)
    lab = torch.from_numpy(ev.labels.astype(np.float32)).to(dev)
    return ev, x, (yy, xx, yb), lab


tr_ev, xtr, (tyy, txx, tyb), tlab = make(0)
va_ev, xva, (vyy, vxx, vyb), vlab = make(1)

model = DynThreshSNN(C=16, head=HEAD).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def val_auc():
    model.eval()
    with torch.no_grad():
        out = model(xva)[0, 0]                            # [H,W,T]
        sc = out[vyy, vxx, vyb].cpu().numpy()
    model.train()
    return float(metrics.roc(sc, va_ev.labels)["auc"])


print(f"init val AUC = {val_auc():.4f}")
for step in range(1, STEPS + 1):
    out = model(xtr)[0, 0]                                # [H,W,T]
    logit = out[tyy, txx, tyb]
    loss = F.binary_cross_entropy_with_logits(logit, tlab)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % max(1, STEPS // 6) == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}  valAUC {val_auc():.4f}")

# baseline STCD on the same val stream, for reference
fe = SpikingFrontEnd(FrontEndConfig(neighbor_k=3, pool=1, tau=8e-3, theta=1.5, dt=DT))
b_auc = float(metrics.roc(fe.score_events(va_ev), va_ev.labels)["auc"])
print(f"\nB1({HEAD}) trained valAUC={val_auc():.4f}  vs baseline STCD valAUC={b_auc:.4f}")
print("B1 SMOKE OK")
