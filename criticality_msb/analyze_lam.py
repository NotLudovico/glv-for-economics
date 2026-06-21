"""Immigration scan: does λ>0 (firms floor at ~λ instead of crashing to 0) flatten the FAITHFUL β toward
the empirical 0.20? Uses the permanent-delist listing (a firm below the floor is dead). At λ≥floor no firm
dies, so this becomes the all-listed-firms β at the immigration floor.  uv run python -m criticality_msb.analyze_lam
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from criticality_msb import config as C
from criticality_msb.dataset import Run
from criticality_msb import observables as O

LAMS = ["0.0", "0.001", "0.01", "0.1"]
LAMV = [1e-4, 1e-3, 1e-2, 1e-1]   # x positions (λ=0 drawn at 1e-4)


def load(lam, f):
    return [Run(p) for p in glob.glob(f"{C.DATA_DIR}/run_*lam={lam}_*f={f}.npz")]


def beta_faithful(rs, floor=1e-4):
    Sb, Vl = [], []
    for r in rs:
        s, v = O.delisted_size_volatility(r, dt=1.0, floor=floor, min_growths=2)
        Sb.append(s); Vl.append(v)
    return O.beta_fit(np.concatenate(Sb), np.concatenate(Vl)) if Sb else np.nan


fig, ax = plt.subplots(figsize=(7.6, 5.2))
print("FAITHFUL β vs immigration λ (empirical 0.204±0.01):")
for f, col in zip(["0.6", "0.9"], ["#2a9d8f", "#c1121f"]):
    bs = []
    for lam in LAMS:
        rs = load(lam, f)
        bs.append(beta_faithful(rs) if rs else np.nan)
    ax.plot(LAMV, bs, "o-", color=col, label=f"f={f}")
    print(f"  f={f}:  " + "  ".join(f"λ={l}:β={b:.3f}" for l, b in zip(LAMS, bs)))
ax.axhspan(0.15, 0.20, color="gray", alpha=0.25, label="empirical MSB")
ax.set(xscale="log", xlabel=r"immigration $\lambda$ (λ=0 drawn at 1e-4)", ylabel=r"$\beta$ (faithful, permanent-delist)",
       title="Does immigration flatten β toward 0.20?")
ax.legend()
plt.tight_layout()
out = f"{C.FIG_DIR}/lam_scan.png"
plt.savefig(out, dpi=120)
print("saved", out)
