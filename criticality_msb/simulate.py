"""Build the criticality–MSB dataset: for each seed locate sigma_c, then for each f integrate at
sigma = f*sigma_c with tight tolerances, storing per-firm log-abundance on a fine grid as one npz/run.

Resumable: a run is done iff its npz exists (workers write independently). sigma_c is cached per seed.

    uv run python -m criticality_msb.simulate            # full grid (config.GRID)
    uv run python -m criticality_msb.simulate --smoke     # tiny grid, fast pipeline check
    uv run python -m criticality_msb.simulate --jobs 8
"""
import argparse
import itertools
import os
import numpy as np
from joblib import Parallel, delayed

from criticality_msb import config as C
from criticality_msb.core import build_realization, locate_sigma_c, integrate

_RUN_KEYS = ["topology", "S", "mu", "gamma", "lam", "tmax", "seed", "f"]
_SC_KEYS = ["topology", "S", "mu", "gamma", "tmax", "seed"]


def run_filename(p):
    return "run_" + "_".join(f"{k}={p[k]}" for k in _RUN_KEYS) + ".npz"


def sigmac_filename(p):
    return "sigmac_" + "_".join(f"{k}={p[k]}" for k in _SC_KEYS) + ".npz"


def simulate_seed(seed, sim, f_values, data_dir):
    base = dict(topology=sim["topology"], S=sim["S"], mu=sim["mu"], gamma=sim["gamma"],
                lam=sim["lam"], tmax=sim["tmax"], seed=int(seed))
    todo = [f for f in f_values
            if not os.path.exists(os.path.join(data_dir, run_filename({**base, "f": f})))]
    if not todo:
        return f"seed {seed}: all {len(f_values)} runs present — skipped"

    mean_a, dis_a, C_eff, x0 = build_realization(
        seed, sim["S"], topology=sim["topology"], mean_degree=sim["mean_degree"],
        alpha_exp=sim["alpha_exp"], mu=sim["mu"], gamma=sim["gamma"])

    scf = os.path.join(data_dir, sigmac_filename(base))             # sigma_c cached per seed
    if os.path.exists(scf):
        sc = float(np.load(scf)["sigma_c"])
    else:
        sc = locate_sigma_c(mean_a, dis_a, x0, sim["locate_T"], lam=sim["lam"],
                            rtol=sim["locate_rtol"], atol=sim["locate_atol"], method=sim["method"],
                            percap_thresh=sim["percap_thresh"],
                            bracket=sim["sigma_bracket"], iters=sim["bisect_iters"])
        np.savez(scf, sigma_c=sc, C_eff=C_eff, **base)

    t_eval = np.arange(0.0, sim["tmax"] + 0.5 * sim["dt_store"], sim["dt_store"])
    msgs = []
    for f in todo:
        sigma = f * sc
        r = integrate(mean_a, dis_a, x0, sigma, sim["tmax"], lam=sim["lam"], rtol=sim["rtol"],
                      atol=sim["atol"], method=sim["method"], t_eval=t_eval, percap_thresh=sim["percap_thresh"])
        sol = np.maximum(r.y, 0.0)                                  # (S, n_reached) on the reached t_eval subset
        t_snap = r.t
        diverged = (not r.success) or (t_snap[-1] < sim["tmax"] - 1e-6)
        logx = np.log(np.maximum(sol, sim["store_floor"])).astype(np.float32)
        minx = sol.min(1).astype(np.float32)                       # per-firm min over the fine grid (extinction proxy)
        p = {**base, "f": f}
        np.savez_compressed(
            os.path.join(data_dir, run_filename(p)),
            logx=logx, minx=minx, t_snap=t_snap.astype(np.float32),
            sigma=sigma, sigma_c=sc, C_eff=C_eff, reached=float(t_snap[-1]), diverged=diverged,
            dt_store=sim["dt_store"], rtol=sim["rtol"], atol=sim["atol"], **p)
        msgs.append(f"f={f}:reached={t_snap[-1]:.0f}{'!div' if diverged else ''}")
    return f"seed {seed} (σ_c={sc:.3f}, <k>={C_eff:.0f}): " + " ".join(msgs)


def _build_jobs(sim_base, axes, n_seeds, base_seed):
    """One (seed, sim) job per (axis-combo × seed). axes={} -> just the base sim over the seeds."""
    seeds = np.random.default_rng(base_seed).integers(0, 2**31 - 1, size=n_seeds)
    keys = list(axes)
    combos = list(itertools.product(*[axes[k] for k in keys])) if keys else [()]
    jobs = [(int(s), {**sim_base, **dict(zip(keys, combo))}) for combo in combos for s in seeds]
    return jobs, len(combos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a fast pipeline check")
    ap.add_argument("--sweep", action="store_true", help="run config.SWEEP (physical-axis scan) instead of the base grid")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    if args.sweep:
        sw = C.SWEEP
        sim_base = {**C.SIM, **sw.get("sim", {})}
        jobs, ncombo = _build_jobs(sim_base, sw["axes"], sw["n_seeds"], C.GRID["base_seed"])
        f_values = sw["f_values"]; label = f"SWEEP {sw['axes']}"; data_dir = C.DATA_DIR
    else:
        sim_base = dict(C.SIM); grid = dict(C.GRID)
        if args.smoke:
            sim_base.update(C.SMOKE_SIM); grid.update(C.SMOKE_GRID)
        jobs, ncombo = _build_jobs(sim_base, {}, grid["n_seeds"], grid["base_seed"])
        f_values = grid["f_values"]; label = "base grid" + (" (smoke)" if args.smoke else "")
        data_dir = C.DATA_DIR + ("_smoke" if args.smoke else "")

    os.makedirs(data_dir, exist_ok=True)
    n_seeds = len(jobs) // max(ncombo, 1)
    print(f"{label} -> {data_dir}\n{ncombo} param-combo(s) x {n_seeds} seeds x {len(f_values)} f "
          f"= {len(jobs) * len(f_values)} runs (S={sim_base['S']}, rtol={sim_base['rtol']:g})")
    results = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
        delayed(simulate_seed)(s, sim_c, f_values, data_dir) for (s, sim_c) in jobs)
    for m in results:
        print(" ", m)
    n_runs = len([f for f in os.listdir(data_dir) if f.startswith("run_")])
    print(f"\ndone: {n_runs} run files in {data_dir}")


if __name__ == "__main__":
    main()
