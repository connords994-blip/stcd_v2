import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "src")
import torch
from stcd.learned import FeedbackDenoiser, assoc_scan, SpatialSSM, morton_order

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev)

# 1) assoc_scan vs slow reference loop
B, L, D, N = 2, 200, 3, 4
a = torch.rand(B, L, D, N) * 0.9 + 0.05
b = torch.randn(B, L, D, N)
ref = torch.zeros(B, L, D, N)
h = torch.zeros(B, D, N)
for l in range(L):
    h = a[:, l] * h + b[:, l]
    ref[:, l] = h
got = assoc_scan(a, b)
err = (got - ref).abs().max().item()
print(f"[scan] max|assoc_scan - loop| = {err:.2e}  (B,L,D,N={B,L,D,N})")
assert err < 1e-4, "assoc_scan mismatch"

# 2) morton order is a permutation
order, inv = morton_order(16, 24, "cpu")
assert torch.equal(order[inv], torch.arange(16 * 24)), "morton inv broken"
print("[morton] permutation + inverse OK")

# 3) SpatialSSM shape
ssm = SpatialSSM(8, d_state=4)
y = ssm(torch.randn(2, 50, 8))
print("[ssm] out", tuple(y.shape))
assert tuple(y.shape) == (2, 50, 8)

# 4) shape + causality for both backends
for backend in ["snn", "ssn"]:
    m = FeedbackDenoiser(backend=backend).to(dev).eval()
    x = torch.rand(2, 2, 32, 32, 5, device=dev)
    out = m(x)
    print(f"[{backend}] out", tuple(out.shape))
    assert tuple(out.shape) == (2, 1, 32, 32, 5)
    # causality: zeroing the LAST bin's input must not change scores at t<last
    x2 = x.clone(); x2[..., -1] = 0.0
    with torch.no_grad():
        o1 = m(x); o2 = m(x2)
    d_pre = (o1[..., :-1] - o2[..., :-1]).abs().max().item()
    d_last = (o1[..., -1] - o2[..., -1]).abs().max().item()
    print(f"[{backend}] causal: max|delta| bins<last = {d_pre:.2e} ; last bin changed = {d_last:.2e}")
    assert d_pre < 1e-6, "feedback leaked future info into earlier bins"
    # ablation path runs and differs once trained-ish weights perturbed
    ofb = m(x, use_feedback=False)
    assert tuple(ofb.shape) == (2, 1, 32, 32, 5)

# 5) gradient flows into feedback params (BCE smoke)
m = FeedbackDenoiser(backend="ssn").to(dev)
x = torch.rand(2, 2, 48, 48, 6, device=dev)
tgt = (torch.rand(2, 1, 48, 48, 6, device=dev) > 0.5).float()
loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), tgt)
loss.backward()
g = m.fb.read.weight.grad
print(f"[grad] ssn read.weight grad norm = {None if g is None else g.norm().item():.4e}")
assert g is not None and g.norm().item() > 0, "no gradient into feedback readout"
print("ALL SMOKE CHECKS PASSED")
