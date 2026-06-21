"""Turn ch4_data.npz into the five Chapter-4 figures (relative GLV, locked regime sigma=1.75, mu=1.76 sweep =
chapter mu=-1.76). Writes directly into the thesis directory, replacing the old shrinking-regime figures.
Re-run freely to restyle without recomputing."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm

HERE = os.path.dirname(__file__)
THESIS = os.path.abspath(os.path.join(HERE, "..", "thesis"))
d = np.load(os.path.join(HERE, "ch4_data.npz"), allow_pickle=True)
MU_CH = -1.76   # chapter convention


# ---- 1. single-run churn: M(t) growing + relative sizes churning ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
t, lnM, S = d["sc_t"], d["sc_lnM"], d["sc_share_S"]
ax[0].plot(t, lnM / np.log(10), color="C0")
ax[0].set(xlabel="$t$", ylabel=r"$\log_{10} M(t)$",
          title=fr"Total output ($g_{{\mathrm{{eff}}}}={float(d['sc_g_eff']):.2f}$)")
for row in S:
    ax[1].plot(t, np.log10(np.maximum(row, 1e-8)), lw=0.7)
ax[1].set(xlabel="$t$", ylabel=r"$\log_{10} S_i$", title=r"Relative firm sizes $S_i = N y_i$ (sample)")
plt.tight_layout(); plt.savefig(os.path.join(THESIS, "growing_growth_churn.png"), dpi=140); plt.close()

# ---- 2. MSB: size-volatility (single economy, decline fit) + the tent ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
svS, svV = d["msb_sv_S"].astype(float), d["msb_sv_vol"].astype(float)
bx, by, pk = d["msb_bin_S"], d["msb_bin_vol"], int(d["msb_pk"])
ax[0].scatter(svS, svV, s=5, alpha=0.15, color="0.6")
ax[0].plot(bx, by, "o-", color="C0", ms=5, label="binned median")
if bx.size > pk + 1:
    c = np.polyfit(np.log10(bx[pk:]), np.log10(by[pk:]), 1)
    xx = np.array([bx[pk], bx[-1]])
    ax[0].plot(xx, 10 ** np.polyval(c, np.log10(xx)), "r--", lw=2,
               label=fr"decline fit $\beta={float(d['msb_beta']):.2f}$ ($R^2={float(d['msb_r2']):.2f}$)")
    ax[0].axvspan(bx[0], bx[pk], color="0.85", alpha=0.5)
    ax[0].text(bx[0], by.min(), " floor\n plateau", fontsize=8, va="bottom", color="0.4")
ax[0].set(xscale="log", yscale="log", xlabel=r"firm size $\bar S$", ylabel=r"growth volatility $\sigma(\bar S)$",
          title="Size--volatility relation (one economy)")
ax[0].legend(fontsize=9)
z = d["msb_z"]; xx = np.linspace(-8, 8, 400)
# MSB rescaling: sqrt(pi/2)*MAD, NOT std (std is inflated by the fat tails). Idempotent on already-rescaled
# data, so this also corrects any std-scaled msb_z from a pre-fix ch4_data.npz without a regen.
z = (z - z.mean()) / (np.sqrt(np.pi / 2) * np.abs(z - z.mean()).mean())
ax[1].hist(z, bins=160, density=True, color="C0", alpha=0.6, label="growth rates")
ax[1].plot(xx, laplace.pdf(xx, scale=np.sqrt(2 / np.pi)), "r--", lw=1.5, label="Laplace")
ax[1].plot(xx, norm.pdf(xx), "k:", lw=1.2, label="Gaussian")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, None)
ax[1].set(xlabel=r"rescaled growth rate $\Delta\ln S$ ($\sqrt{\pi/2}\,$MAD)", ylabel="PDF",
          title=fr"Growth distribution (Bowley $={float(d['msb_bowley']):.2f}$, exc.\ kurt $={float(d['msb_exk']):.0f}$)")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(THESIS, "growing_stationary_msb.png"), dpi=140); plt.close()

# ---- 3. mean-field: persistence fraction + beta, both vs N ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
N = d["ns_N"]; pf = d["ns_persistf"]; b = d["ns_beta"]; lo = d["ns_blo"]; hi = d["ns_bhi"]
ax[0].plot(N, pf, "s-", color="C2", ms=7)
ax[0].set(xscale="log", ylim=(0, 1.05), xlabel="$N$", ylabel="persistent fraction",
          title="Persistence of the chaos vs system size")
m = np.isfinite(b)
ax[1].errorbar(N[m], b[m], yerr=[b[m] - lo[m], hi[m] - b[m]], fmt="o-", color="C0", capsize=3)
ax[1].axhspan(0.15, 0.20, color="0.8", alpha=0.6, label=r"empirical $0.15$--$0.20$")
ax[1].set(xscale="log", xlabel="$N$", ylabel=r"size--variance exponent $\beta$",
          title=r"$\beta$ shallows toward the empirical band")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(THESIS, "growing_meanfield.png"), dpi=140); plt.close()

# ---- 4. beta extrapolation: beta vs 1/sqrt(N) ----
fig, ax = plt.subplots(figsize=(7.5, 5.5))
binf, slope = float(d["ns_beta_inf"]), float(d["ns_beta_slope"])
x = 1.0 / np.sqrt(N[m]); xx = np.linspace(0, x.max() * 1.05, 100)
ax.errorbar(x, b[m], yerr=[b[m] - lo[m], hi[m] - b[m]], fmt="o", color="C0", capsize=3, label="measured")
ax.plot(xx, binf + slope * xx, "C3-", lw=1.8, label=r"fit $\beta_\infty + c/\sqrt{N}$")
ax.plot(0, binf, "D", color="C3", ms=9, label=fr"$\beta_\infty={binf:.2f}$")
ax.axhspan(0.15, 0.20, color="0.8", alpha=0.6, label=r"empirical $0.15$--$0.20$")
for xi, Ni in zip(x, N[m]):
    ax.annotate(f"{int(Ni)}", (xi, binf + slope * xi), fontsize=7, color="0.5",
                xytext=(3, 4), textcoords="offset points")
ax.set(xlabel=r"$1/\sqrt{N}$", ylabel=r"size--variance exponent $\beta$",
       title="Mean-field extrapolation of $\\beta$"); ax.set_xlim(left=-0.001)
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(THESIS, "growing_beta_extrapolation.png"), dpi=140); plt.close()

# ---- 5. freeze vs persist trajectories (finite size) ----
per = d["traj_persistent"].astype(bool); gstd = d["traj_gstd"]; tt = d["traj_t"]; sh = d["traj_shares"]
i_freeze = int(np.where(~per)[0][np.argmin(gstd[~per])]) if (~per).any() else 0
i_persist = int(np.where(per)[0][np.argmax(gstd[per])]) if per.any() else 1
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for a, idx, ttl in [(ax[0], i_freeze, "freezes (finite-size relaxation)"),
                    (ax[1], i_persist, "keeps fluctuating")]:
    for row in sh[idx]:
        a.plot(tt, np.log10(np.maximum(row, 1e-8)), lw=0.7)
    a.set(xlabel="$t$", title=fr"seed {idx}: {ttl}")
ax[0].set_ylabel(r"$\log_{10} S_i$")
plt.tight_layout(); plt.savefig(os.path.join(THESIS, "growing_trajectories.png"), dpi=140); plt.close()

print("wrote 5 figures to", THESIS)
print(f"  churn g_eff={float(d['sc_g_eff']):.2f} | msb beta={float(d['msb_beta']):.2f} r2={float(d['msb_r2']):.2f}"
      f" bowley={float(d['msb_bowley']):.2f} exk={float(d['msb_exk']):.0f} | beta_inf={binf:.3f}"
      f" | traj freeze=seed{i_freeze} persist=seed{i_persist}")
