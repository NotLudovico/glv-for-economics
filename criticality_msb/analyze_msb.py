"""Analyze the criticality–MSB dataset: β(f) (all vs persistent), survival φ(f), and the persistent-firm
growth PDF. Reads the dataset; never re-simulates. Demonstrates the analysis pattern.

    uv run python -m criticality_msb.analyze_msb            # full dataset
    uv run python -m criticality_msb.analyze_msb --smoke    # smoke dataset
    uv run python -m criticality_msb.analyze_msb --dt 2 --alive 1e-3
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from criticality_msb import config as C
from criticality_msb.dataset import load_runs, f_values
from criticality_msb import observables as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dt", type=float, default=1.0, help="growth horizon (multiple of dt_store)")
    ap.add_argument("--alive", type=float, default=1e-4, help="persistence threshold")
    args = ap.parse_args()

    data_dir = C.DATA_DIR + ("_smoke" if args.smoke else "")
    fig_dir = C.FIG_DIR + ("_smoke" if args.smoke else "")
    os.makedirs(fig_dir, exist_ok=True)
    runs = load_runs(data_dir)
    if not runs:
        raise SystemExit(f"no runs in {data_dir} — run `simulate` first")
    fs = f_values(runs)
    print(f"{len(runs)} runs | f={fs} | dt={args.dt} | alive={args.alive:g}")

    ba, bp, pp, gper = {}, {}, {}, {}
    for f in fs:
        rf = [r for r in runs if abs(r.f - f) < 1e-9 and not r.diverged]
        b_all, b_per, persist, gp = [], [], [], []
        for r in rf:
            Sbar, vol = O.size_volatility(r, dt=args.dt)
            pm = O.persistent_mask(r, alive=args.alive)
            b_all.append(O.beta_fit(Sbar, vol))
            b_per.append(O.beta_fit(Sbar, vol, mask=pm))
            persist.append(O.extinction_fractions(r, alive=args.alive)["persist"])
            _, g = O.growth(r, dt=args.dt)
            gp.append(g[pm].ravel())
        ba[f] = np.nanmean(b_all) if b_all else np.nan
        bp[f] = np.nanmean(b_per) if b_per else np.nan
        pp[f] = 100 * np.mean(persist) if persist else np.nan
        gp = [x for x in gp if x.size]
        gper[f] = np.concatenate(gp) if gp else np.array([])

    fa = np.array(fs)
    fig, (axB, axE, axP) = plt.subplots(1, 3, figsize=(16, 4.6))
    axB.plot(fa, [ba[f] for f in fs], "s-", color="#888", label="all firms")
    axB.plot(fa, [bp[f] for f in fs], "o-", color="#2a9d8f", label="persistent (listed)")
    axB.axhspan(0.15, 0.20, color="gray", alpha=0.2, label="empirical MSB")
    axB.set(xlabel=r"$f=\sigma/\sigma_c$", ylabel=r"$\beta$", title="β: persistent vs all")
    axB.legend()
    axE.plot(fa, [pp[f] for f in fs], "o-", color="#2a9d8f")
    axE.set(xlabel=r"$f$", ylabel="% persistent", title="survival vs criticality")
    e = np.linspace(-8, 8, 80); cc = 0.5 * (e[:-1] + e[1:])
    pick = fs[::max(1, len(fs) // 3)][-3:]
    for f, col in zip(pick, ["#457b9d", "#e76f51", "#c1121f"]):
        g = gper[f]; g = g[np.isfinite(g)]
        if g.size:
            d, _ = np.histogram(g, bins=e, density=True)
            axP.semilogy(cc, np.where(d > 0, d, np.nan), "-", color=col, label=f"f={f}")
    axP.set(xlabel=r"$g=\Delta\ln S$", ylabel="density", title="persistent-firm growth PDF")
    axP.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(fig_dir, "msb_overview.png")
    plt.savefig(out, dpi=120)
    print("saved", out)
    for f in fs:
        print(f"  f={f}: β_all={ba[f]:.3f}  β_persist={bp[f]:.3f}  %persist={pp[f]:.0f}")


if __name__ == "__main__":
    main()
