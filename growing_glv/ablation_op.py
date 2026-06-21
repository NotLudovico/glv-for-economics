"""Disorder ablation at the LOCKED operating mu (chapter mu=-1.76 = sweep mu=1.76): sweep sigma across the
chaotic threshold sigma_c=sqrt(2), measure the late-window growth fluctuation + excess kurtosis. Regenerates
Table 'tab:growing-ablation' at the correct regime."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kurtosis
from joblib import Parallel, delayed
from explore import build_alpha

MU, LAM, N = 1.76, 1e-3, 3200
SIGMAS = [0.0, 0.5, 1.0, 1.41, 1.6, 1.75]
SEEDS = 5
TMAX, N_EVAL, DT = 400.0, 1600, 0.5
LATE = (320.0, 390.0)
RTOL, ATOL, MAX_NFEV = 1e-4, 1e-7, 80000
TGRID = np.linspace(0.0, TMAX, N_EVAL)


class _Capped(Exception):
    pass


def run(sigma, seed):
    alpha, _ = build_alpha(seed, N, MU, sigma)
    rng = np.random.default_rng(seed + 1); w0 = rng.uniform(0.5, 1.5, N); w0 /= w0.sum()
    nev = [0]
    def rhs(t, st):
        nev[0] += 1
        if nev[0] > MAX_NFEV:
            raise _Capped()
        w = np.clip(st[:N], 0.0, None); s = w.sum()
        if s > 0:
            w = w / s
        f = 1.0 - N * w - N * (alpha @ w); fb = float(w @ f)
        return np.concatenate([w * (f - fb) + LAM * (1.0 / N - w), [fb]])
    try:
        r = solve_ivp(rhs, (0, TMAX), np.concatenate([w0, [0.0]]), method="RK45",
                      t_eval=TGRID, rtol=RTOL, atol=ATOL)
    except _Capped:
        return np.nan, np.nan
    t = r.t; W = np.clip(r.y[:N], 0, None); W = W / W.sum(0, keepdims=True)
    surv = np.where(W[:, (t >= LATE[0]) & (t <= LATE[1])].min(1) > 1e-6)[0]
    if surv.size < 30:
        return 0.0, np.nan
    tg = np.arange(LATE[0], LATE[1] + 1e-9, DT)
    g = np.diff(np.array([np.interp(tg, t, np.log(np.maximum(N * W[i], 1e-12))) for i in surv]), axis=1)
    flat = g.ravel(); flat = flat[np.isfinite(flat)]
    return float(g.std()), float(kurtosis(flat))


if __name__ == "__main__":
    print(f"ablation at mu={MU} (chapter mu={-MU}), N={N}, lambda={LAM}; sigma_c=sqrt(2)~1.414\n")
    print(f"{'sigma':>6} {'growth fluct':>13} {'exc kurt':>9}  tent?")
    for sig in SIGMAS:
        res = Parallel(n_jobs=6)(delayed(run)(sig, s) for s in range(SEEDS))
        gs = np.nanmedian([a for a, _ in res]); ek = np.nanmedian([b for _, b in res])
        tent = "yes" if (gs > 0.02 and ek > 2) else "no"
        print(f"{sig:>6.2f} {gs:>13.3f} {ek:>9.1f}  {tent}")
