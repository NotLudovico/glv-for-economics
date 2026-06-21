"""Validate the relative-GLV DMFT solver against direct simulation.

The solver (dmft_solver.py) is derived for the regular / fully-connected graph in the
high-connectivity limit. So we test it against a *matched* simulation: fully-connected
Gaussian couplings alpha_ij = mu/N + (sigma/sqrt N) z_ij, z iid (gamma=0), zero diagonal.
(The existing phase_space_data.npz uses a power-law graph, a different ensemble, so it is
not the clean test -- see the DMFT doc, section 7.)

Three checks:
  1. relaxed phase (sigma < sqrt2): simulated growth rate g_eff ~= DMFT g*, survival ~= phi.
  2. chaos onset: the persistent-fluctuation amplitude turns on near sigma_c = sqrt2.
  3. mu-independence: at fixed sigma, varying mu leaves survival/fluctuation unchanged and
     shifts g_eff by exactly -delta_mu.

Run: uv run python growing_glv/dmft_validate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmft_solver import solve_fixed_point, sigma_c

N = 1500
TMAX = 160.0
LATE = (100.0, 150.0)
LAM = 0.0                       # match the lambda=0 DMFT
RTOL = 1e-6


def fc_alpha(seed, mu, sigma):
    """Fully-connected Gaussian couplings, zero diagonal, gamma=0."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N, N))
    a = mu / N + (sigma / np.sqrt(N)) * z
    np.fill_diagonal(a, 0.0)
    return a


def simulate(seed, mu, sigma):
    """Integrate the relative GLV (share + lnM split); return g_eff, survival, fluct."""
    alpha = fc_alpha(seed, mu, sigma)
    rng = np.random.default_rng(seed + 1)
    w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()

    def rhs(t, st):
        w = np.clip(st[:N], 0.0, None); s = w.sum()
        if s > 0: w = w / s
        f = 1.0 - N * w - N * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / N - w), [fb]])

    r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="LSODA",
                  t_eval=np.linspace(0, TMAX, 800), rtol=RTOL, atol=1e-9)
    W = np.clip(r.y[:N], 0.0, None); W = W / W.sum(0, keepdims=True); t = r.t; lnM = r.y[N]
    late = t > LATE[0]
    g_eff = float(np.polyfit(t[late], lnM[late], 1)[0])
    win = (t >= LATE[0]) & (t <= LATE[1])
    surv_mask = W[:, win].min(1) > 1e-6
    surv = float(surv_mask.mean())
    # persistent-fluctuation amplitude: std of dln(share) over survivors in the late window
    if surv_mask.sum() > 20:
        tg = np.arange(LATE[0], LATE[1] + 1e-9, 0.5)
        lnS = np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12)))
                        for i in np.where(surv_mask)[0]])
        fluct = float(np.diff(lnS, axis=1).std())
    else:
        fluct = 0.0
    return g_eff, surv, fluct


def sweep(mu, sigmas, seeds):
    out = {}
    for sig in sigmas:
        rows = np.array([simulate(int(s), mu, sig) for s in seeds])
        out[sig] = rows.mean(0)            # (g_eff, surv, fluct)
        print(f"  mu={mu:+.2f} sigma={sig:.3f}: g_eff={out[sig][0]:+.3f} "
              f"surv={out[sig][1]:.3f} fluct={out[sig][2]:.4f}", flush=True)
    return out


if __name__ == "__main__":
    sc = sigma_c(0.0)
    sigmas = np.array([0.4, 0.7, 1.0, 1.2, 1.35, 1.6, 2.0, 2.5])
    mui_mus = np.array([0.0, 1.0, 2.0])
    MU = 0.5
    NPZ = os.path.join(os.path.dirname(__file__), "dmft_validation.npz")

    # dataset-oriented: simulate once, store raw results, replot for free (delete the npz to recompute)
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        sigmas, sim_arr, mui_mus, mui_arr = d["sigmas"], d["sim"], d["mui_mus"], d["mui"]
        print(f"[loaded] {NPZ}  (delete to recompute)")
    else:
        seeds = np.random.default_rng(11).integers(0, 2**31 - 1, size=3)
        print(f"[matched sim] N={N}, fully-connected Gaussian, gamma=0, lambda={LAM}, sigma_c={sc:.3f}")
        print("sweep over sigma at mu=%.2f:" % MU)
        s = sweep(MU, sigmas, seeds)
        sim_arr = np.array([s[sig] for sig in sigmas])
        print("mu-independence at sigma=1.0 (shape fixed, g_eff shifts by -mu):")
        rows = []
        for mu in mui_mus:
            g, sv, f = simulate(int(seeds[0]), float(mu), 1.0)
            rows.append([g, sv, f])
            print(f"  mu={mu:+.2f}: g_eff={g:+.3f} surv={sv:.3f} fluct={f:.4f}", flush=True)
        mui_arr = np.array(rows)
        np.savez(NPZ, sigmas=sigmas, sim=sim_arr, mui_mus=mui_mus, mui=mui_arr)
        print(f"[saved] {NPZ}")

    sim = {float(sig): sim_arr[i] for i, sig in enumerate(sigmas)}
    mui = {float(mu): tuple(mui_arr[i]) for i, mu in enumerate(mui_mus)}

    # ---- comparison table vs DMFT ----
    print("\n  sigma | g_eff(sim)  g*(DMFT) | surv(sim)  phi(DMFT) | phase")
    relaxed_err_g, relaxed_err_s = [], []
    for sig in sigmas:
        d = solve_fixed_point(MU, float(sig))
        ge, su, fl = sim[sig]
        phase = "relaxed" if sig < sc else "fluct."
        tag = ""
        if sig < sc:
            relaxed_err_g.append(abs(ge - d["gstar"])); relaxed_err_s.append(abs(su - d["phi"]))
        else:
            tag = "  (g* is unstable-FP, not the true rate)"
        print(f"  {sig:5.3f} | {ge:+8.3f}  {d['gstar']:+8.3f} | {su:8.3f}  {d['phi']:8.3f} | {phase}{tag}")

    print(f"\nrelaxed-phase mean |g_eff - g*| = {np.mean(relaxed_err_g):.3f}, "
          f"mean |surv - phi| = {np.mean(relaxed_err_s):.3f}")

    # ---- self-check assertions (loose: finite N, finite T, 3 seeds) ----
    assert np.mean(relaxed_err_g) < 0.12, "relaxed-phase growth rate disagrees with DMFT"
    assert np.mean(relaxed_err_s) < 0.10, "relaxed-phase survival disagrees with DMFT"
    # chaos onset: fluctuation amplitude small below sigma_c, large above
    assert sim[0.7][2] < 0.01 and sim[2.0][2] > 0.02, "chaos onset not near sigma_c"
    # mu-independence: shape unchanged, g_eff shifts by -mu
    assert abs(mui[1.0][1] - mui[0.0][1]) < 0.05, "survival should be mu-independent"
    assert abs((mui[1.0][0] - mui[0.0][0]) - (-1.0)) < 0.08, "g_eff should shift by -delta_mu"
    print("ALL CHECKS PASSED")

    # ---- figure ----
    sg = np.array(sigmas)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    dm = [solve_fixed_point(MU, float(s)) for s in sg]
    gstar = np.array([d["gstar"] for d in dm]); phi = np.array([d["phi"] for d in dm])
    rel = sg < sc

    ax[0].plot(sg, [sim[s][0] for s in sg], "o-", label="g_eff (sim)", color="#2a9d8f")
    ax[0].plot(sg[rel], gstar[rel], "s--", label="g* DMFT (relaxed)", color="#c1121f")
    ax[0].plot(sg[~rel], gstar[~rel], "x:", label="g* DMFT (unstable FP)", color="#c1121f", alpha=0.5)
    ax[0].axvline(sc, color="k", ls=":", lw=1); ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set(xlabel=r"$\sigma$", ylabel=r"growth rate $g^*$", title=f"Growth rate ($\\mu={MU}$)")
    ax[0].legend(fontsize=8)

    ax[1].plot(sg, [sim[s][1] for s in sg], "o-", label="survival (sim)", color="#2a9d8f")
    ax[1].plot(sg, phi, "s--", label=r"$\varphi$ DMFT", color="#c1121f")
    ax[1].axvline(sc, color="k", ls=":", lw=1)
    ax[1].set(xlabel=r"$\sigma$", ylabel="surviving fraction", title="Coexistence")
    ax[1].legend(fontsize=8)

    ax[2].plot(sg, [sim[s][2] for s in sg], "o-", color="#2a9d8f")
    ax[2].axvline(sc, color="#c1121f", ls="--", lw=1.4)
    ax[2].text(sc + 0.03, 0.6 * max(sim[s][2] for s in sg) + 1e-6,
               r"$\sigma_c=\sqrt{2}$ (DMFT)", color="#c1121f", fontsize=9)
    ax[2].set(xlabel=r"$\sigma$", ylabel="fluctuation amplitude",
              title="Chaos onset (relaxed $\\to$ fluctuating)")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "dmft_validation.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")
