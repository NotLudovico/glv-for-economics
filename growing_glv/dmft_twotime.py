"""Two-time DMFT for the relative GLV, fluctuating phase (gamma=0).

The appendix solves the *relaxed* phase, where the share autocorrelation freezes,
C(t,s) -> q, the coloured noise becomes a single frozen Gaussian, and the theory is
4 algebraic equations (dmft_solver.py). Above sigma_c = sqrt(2) the fixed point is
unstable: shares fluctuate forever, C(t,s) does NOT freeze, and you must solve the
single-site process self-consistently for the whole *function* C(t,s).

At gamma=0 the Onsager memory term gamma sigma^2 int G y drops out, so the response
G(t,s) is not needed for the dynamics. The ONLY self-consistent object is the noise
covariance. The single-site law (appendix eq A.9, gamma=0, M_1=1):

    d y / dt = y [ 1 - y + mu - g(t) + sigma eta(t) ],   <eta(t)eta(s)> = C(t,s),
    g(t) = E[y F],  F = 1 - y + mu + sigma eta,   E[y(t)] = 1.

Self-consistent scheme (an ensemble of n single-site firms IS the disorder average):
  1. given C(t,s) on a time grid, Cholesky L (C = L L^T)
  2. draw n noise paths eta_k = L @ randn(Nt)             (each ~ N(0, C))
  3. integrate the ensemble jointly in share form (renormalise <y>=1 each step,
     which is the -g(t) drift), accumulating ln M
  4. remeasure C_new(t,s) = mean_k y_k(t) y_k(s)  (a Gram matrix, automatically PSD)
  5. damp C <- (1-a) C + a C_new, iterate to convergence

Convention: appendix (mu<0 competitive). The matched sim in dmft_validation.npz is at
mu=-0.5, so compare there. Run: uv run python growing_glv/dmft_twotime.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from dmft_solver import solve_fixed_point, sigma_c


# ----------------------------------------------------------- colored noise
def cholesky_psd(C, jitter=1e-10):
    """Cholesky of a (numerically) PSD covariance, with escalating jitter."""
    d = np.diag(C).mean()
    for k in range(8):
        try:
            return np.linalg.cholesky(C + (jitter * d * 10.0 ** k) * np.eye(len(C)))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance not PSD even with jitter")


def draw_noise(L, n, rng):
    """n stationary-or-not Gaussian paths with covariance C = L L^T."""
    return (L @ rng.standard_normal((len(L), n))).T          # (n, Nt)


# ------------------------------------------------------------- integrator
def integrate(eta, mu, sigma, dt, u0, nsub=4, lam=0.0):
    """Integrate the share ensemble for fixed noise paths eta (n, Nt).

    Returns U (n, Nt) shares (<U[:,t]>=1 each t), g_t (Nt,) common growth rate.
    Single-site law: du/dt = u(F - g) + lam(1 - u). Operator-split per substep: exponential
    growth u *= exp(h(F-g)), then the firm-entry floor u += h lam (1-u), then renormalise
    <u>=1. lam>0 (entry) keeps abundances off the extinction boundary -> stationary phase;
    lam=0 -> the closed-economy aging idealisation. nsub substeps (eta interpolated) keep the
    explicit step stable at large sigma; the exp is clipped as a float-overflow guard.
    """
    n, Nt = eta.shape
    u = u0.copy()
    U = np.empty((n, Nt)); g_t = np.empty(Nt)
    h = dt / nsub
    for t in range(Nt):
        e0 = eta[:, t]; e1 = eta[:, t + 1] if t + 1 < Nt else e0
        F = 1.0 - u + mu + sigma * e0
        g_t[t] = float(u @ F) / n
        U[:, t] = u
        for j in range(nsub):                               # substep within [t, t+1]
            e = e0 + (e1 - e0) * (j + 0.5) / nsub
            F = 1.0 - u + mu + sigma * e
            g = float(u @ F) / n
            u = u * np.exp(np.clip(h * (F - g), -30.0, 30.0))
            if lam:
                u = u + h * lam * (1.0 - u)                 # firm entry (anti-extinction floor)
            u = np.maximum(u, 0.0); u *= n / u.sum()        # pin <u> = 1 each substep
    return U, g_t


# --------------------------------------------------------------- solver
def solve_twotime(mu, sigma, Nt=320, dt=0.2, n=4000, iters=34, burn=18, damp=0.4,
                  late=0.6, seed=0, lam=0.0, verbose=False):
    """Self-consistent two-time DMFT at gamma=0. Returns dict of observables.

    The self-consistency is a Monte-Carlo map (finite ensemble), so it does not converge
    to a point but fluctuates about the attractor. We iterate `iters` times and POOL the
    late-window samples of the post-`burn` iterations to average out the MC noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(Nt) * dt
    lo = int(late * Nt)                                     # late (stationary) window
    # warm start: a DECAYING C(t,s) = 1 + (q-1) exp(-|t-s|/tau0). A constant (frozen) C
    # is the unstable relaxed FP and the iteration sticks there; seeding decorrelation
    # lets the self-consistency find the true fluctuating attractor above sigma_c.
    rel = solve_fixed_point(-mu, sigma)                    # solver uses opposite mu sign
    q0 = max(rel["q"], 1.0)
    lag = np.abs(t[:, None] - t[None, :])
    C = 1.0 + (q0 - 1.0) * np.exp(-lag / 3.0)
    u0 = rng.uniform(0.5, 1.5, n); u0 *= n / u0.sum()

    hist = []; Cbar = np.zeros_like(C); nbar = 0
    U_pool, g_pool = [], []
    for it in range(iters):
        L = cholesky_psd(C)
        eta = draw_noise(L, n, rng)
        U, g_t = integrate(eta, mu, sigma, dt, u0, lam=lam)
        Cn = (U.T @ U) / n                                  # C_new(t,s) = <y(t)y(s)>, PSD
        err = np.abs(Cn[lo:, lo:] - C[lo:, lo:]).mean()
        hist.append(err)
        C = (1 - damp) * C + damp * Cn
        if it >= burn:                                      # pool the tail to beat MC noise
            Cbar += Cn; nbar += 1
            U_pool.append(U[:, lo:]); g_pool.append(g_t[lo:])
        if verbose:
            print(f"  it{it:02d} err={err:.4f} q={Cn[lo:,lo:].diagonal().mean():.3f} "
                  f"g={g_t[lo:].mean():+.3f}", flush=True)

    Cbar /= nbar
    Up = np.concatenate(U_pool, axis=1)                    # (n, n_pooled_times)
    gp = np.concatenate(g_pool)
    q = float(Cbar[lo:, lo:].diagonal().mean())           # E[y^2]
    g_eff = float(gp.mean())                               # aggregate growth rate
    # survival is PER-RUN (each tail iteration is an independent noise realisation), then
    # averaged; threshold matches the sim's share>1e-6 (share = u/n, so u > 1e-6 n).
    thr = 1e-6 * n
    surv = float(np.mean([(W.min(1) > thr).mean() for W in U_pool]))
    taus = np.arange(0, Nt - lo)
    Ctau = np.array([np.mean([Cbar[a, a + d] for a in range(lo, Nt - d)]) for d in taus])
    decorr = float((Ctau[-1] - 1.0) / (Ctau[0] - 1.0))    # connected: ->0 decorrelated, 1 frozen
    return dict(mu=mu, sigma=sigma, t=t, C=Cbar, Ctau=Ctau, dt=dt, lo=lo,
                q=q, g_eff=g_eff, surv=surv, decorr=decorr, U=U, g_t=g_t,
                Up=Up, U_list=U_pool, err=float(np.mean(hist[burn:])), iters=it, rel=rel)


def msb_beta(r, h_units=5.0, nbin=12, floor=0.05):
    """MSB exponent from the converged single-site ensemble.

    Growth over a horizon h: G = ln u(t+h) - ln u(t); bin survivors by size u(t);
    beta = -slope of log std(G | size) vs log size. beta>0 = small firms more volatile.
    Pools (size, growth) pairs over every tail iteration's late window (within-run only).
    floor = bulk cut (in share-mean units; <y>=1): excludes near-extinction dust, whose
    extra volatility otherwise drags beta down at long horizons. beta is range-sensitive
    because the relation is convex, so floor must match between beta and size_memory_time.
    """
    dt = r["dt"]
    h = max(1, int(h_units / dt))
    sizes, grow = [], []
    for W in r["U_list"]:                                  # one late-window block per iteration
        for t in range(W.shape[1] - h):
            a, b = W[:, t], W[:, t + h]
            m = (a > floor) & (b > floor)                  # alive (bulk) at both ends
            sizes.append(a[m]); grow.append(np.log(b[m]) - np.log(a[m]))
    if not sizes or sum(len(x) for x in sizes) < 200:     # horizon exceeds the window
        return dict(beta=np.nan, r2=np.nan, lx=np.array([]), ly=np.array([]), h=h * dt)
    s = np.concatenate(sizes); G = np.concatenate(grow)
    edges = np.quantile(np.log(s), np.linspace(0, 1, nbin + 1))
    lx, ly = [], []
    for i in range(nbin):
        m = (np.log(s) >= edges[i]) & (np.log(s) < edges[i + 1])
        if m.sum() > 50:
            lx.append(np.log(s[m]).mean()); ly.append(np.log(G[m].std()))
    lx, ly = np.array(lx), np.array(ly)
    if len(lx) < 4:
        return dict(beta=np.nan, r2=np.nan, lx=lx, ly=ly, h=h * dt)
    c = np.polyfit(lx, ly, 1)
    r2 = 1.0 - np.sum((ly - np.polyval(c, lx)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return dict(beta=-float(c[0]), r2=float(r2), lx=lx, ly=ly, h=h * dt)


def size_memory_time(r, floor=0.05):
    """1/e autocorrelation time of ln(share) for persistent survivors -- the mean-reversion
    timescale that sets the MSB horizon. beta turns out to collapse on h/tau. Returns
    (tau, ac): tau in time units (= window length, a lower bound, if it never crosses 1/e,
    as happens near sigma_c where the memory diverges)."""
    U, dt, lo = r["U"], r["dt"], r["lo"]
    surv = U[:, lo:].min(1) > floor
    L = np.log(U[surv, lo:]); L = L - L.mean(0, keepdims=True)   # strip the common drift
    T = L.shape[1]
    ac = np.array([(L[:, :T - k] * L[:, k:]).mean() for k in range(T)])
    ac /= ac[0]
    below = np.where(ac < 1.0 / np.e)[0]
    tau = (below[0] if below.size else T) * dt
    return tau, ac


# ===================================================================== checks
def _selftest_noise():
    """Cholesky noise must reproduce a target covariance."""
    rng = np.random.default_rng(1)
    Nt = 40
    tt = np.arange(Nt)
    C = 1.0 + np.exp(-np.abs(tt[:, None] - tt[None, :]) / 5.0)   # ->1 tail, peak 2
    L = cholesky_psd(C)
    eta = draw_noise(L, 20000, rng)
    Cs = (eta.T @ eta) / len(eta)
    rel = np.abs(Cs - C).max() / C.max()
    assert rel < 0.05, f"noise covariance off by {rel:.3f}"
    print(f"[selftest] colored-noise covariance reproduced (max rel err {rel:.3f})  OK")


if __name__ == "__main__":
    _selftest_noise()
    sc = sigma_c(0.0)
    MU = -0.5                                              # appendix convention (matches npz)

    # ---- stage 1: relaxed phase must FREEZE to the relaxed solver's q, g ----
    print(f"\n[stage 1] relaxed phase (sigma<{sc:.3f}): two-time C must freeze to q*")
    print(f"  {'sigma':>5} {'q(2t)':>7} {'q*(rel)':>8} {'g(2t)':>7} {'g*(rel)':>8} {'decorr':>7}")
    for sig in (0.7, 1.0, 1.2):
        r = solve_twotime(MU, sig, seed=0)
        print(f"  {sig:5.2f} {r['q']:7.3f} {r['rel']['q']:8.3f} "
              f"{r['g_eff']:+7.3f} {r['rel']['gstar']:+8.3f} {r['decorr']:7.3f}")

    # ---- stage 2+3: fluctuating phase vs DIRECT SIM (not the static FP), and beta ----
    npz = os.path.join(os.path.dirname(__file__), "dmft_validation.npz")
    d = np.load(npz)
    sim = {float(s): row for s, row in zip(d["sigmas"], d["sim"])}     # (g_eff, surv, fluct)
    print(f"\n[stage 2] fluctuating phase (sigma>{sc:.3f}): two-time vs sim vs static-FP")
    print(f"  {'sigma':>5} | {'g(2t)':>7} {'g(sim)':>7} {'g*(stat)':>8} | "
          f"{'sv(2t)':>6} {'sv(sim)':>7} {'phi(stat)':>9} | {'decorr':>7} {'beta':>6} {'r2':>5}")
    for sig in (1.6, 2.0, 2.5):
        r = solve_twotime(MU, sig, seed=0)
        b = msb_beta(r)
        st = solve_fixed_point(-MU, sig)
        gs, ss, _ = sim[sig]
        print(f"  {sig:5.2f} | {r['g_eff']:+7.3f} {gs:+7.3f} {st['gstar']:+8.3f} | "
              f"{r['surv']:6.3f} {ss:7.3f} {st['phi']:9.3f} | {r['decorr']:7.3f} "
              f"{b['beta']:+6.3f} {b['r2']:5.2f}")

    # ---- seed robustness: the fluctuating attractor must not depend on the RNG seed ----
    print(f"\n[robustness] sigma=2.0 across seeds (attractor should be seed-independent)")
    for sd in (0, 1, 2):
        r = solve_twotime(MU, 2.0, seed=sd)
        b = msb_beta(r)
        print(f"  seed={sd}: q={r['q']:.3f} g={r['g_eff']:+.3f} surv={r['surv']:.3f} beta={b['beta']:+.3f}")
