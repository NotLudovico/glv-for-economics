"""Load runs from a dataset directory, with optional parameter filtering. Lazy per-file npz load."""
import glob
import os
import numpy as np

_META = ["sigma", "sigma_c", "f", "seed", "S", "mu", "gamma", "lam", "topology",
         "tmax", "C_eff", "reached", "diverged", "dt_store"]


class Run:
    """One simulation run: arrays logx (S, n_snap), minx (S,), t_snap (n_snap,) + scalar metadata attrs."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.path = path
        self.logx = d["logx"]
        self.minx = d["minx"]
        self.t_snap = d["t_snap"]
        for k in _META:
            v = d[k]
            setattr(self, k, v.item() if v.ndim == 0 else v)

    def __repr__(self):
        return f"<Run topo={self.topology} S={self.S} seed={self.seed} f={self.f} σ_c={self.sigma_c:.3f}>"


def _match(val, target):
    if isinstance(target, (list, tuple, set)):
        return val in target
    if isinstance(target, float):
        return abs(val - target) < 1e-9
    return val == target


def load_runs(data_dir, **filters):
    """Runs matching filters (e.g. mu=4.0, lam=0.0, f=0.9). Peeks each file's small metadata cheaply and
    fully loads (arrays) ONLY the matches — so a filtered query over a big mixed dataset stays light."""
    runs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "run_*.npz"))):
        with np.load(path, allow_pickle=True) as d:
            ok = all(_match(d[k].item() if d[k].ndim == 0 else d[k], v) for k, v in filters.items())
        if ok:
            runs.append(Run(path))
    return runs


def f_values(runs):
    return sorted({r.f for r in runs})


def manifest(data_dir):
    """Quick summary table (no array loading) of what's in a dataset directory."""
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "run_*.npz"))):
        d = np.load(path, allow_pickle=True)
        rows.append({k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in
                     ["topology", "S", "mu", "seed", "f", "sigma_c", "diverged"]})
    return rows
