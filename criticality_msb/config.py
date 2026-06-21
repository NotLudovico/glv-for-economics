"""Campaign configuration. Edit the grid / sim settings here; scripts read these."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
FIG_DIR = os.path.join(_HERE, "figures")

# ---- simulation settings (per run) ----
SIM = dict(
    topology="powerlaw",      # "powerlaw" | "exponential" | "regular"
    mean_degree=100,          # C
    alpha_exp=2.5,            # power-law degree exponent
    mu=4.0,                   # mean competition (Roy convention: positive = competition)
    gamma=0.0,                # reciprocity
    lam=0.0,                  # immigration
    method="RK45",            # integrator; RK45 ~7-30x faster than LSODA here (LSODA available if needed)
    S=3000,                   # species (beta/tails ~ S-independent; finite-size via a separate slice)
    tmax=500.0,               # integration horizon
    dt_store=0.25,            # snapshot spacing stored (matches prior resolution; supports analysis Δt ≥ 0.25)
    rtol=1e-8, atol=1e-10,    # TIGHT tolerances for the stored measurement (fidelity; we compute once)
    locate_rtol=1e-6, locate_atol=1e-8,   # LOOSER for the sigma_c locate (divergence detection only — not stored)
    locate_T=300.0,           # horizon for the sigma_c divergence-onset locate
    percap_thresh=10.0,       # divergence = mean abundance exceeds this (S-independent)
    sigma_bracket=(1.0, 3.5), # bisection bracket for sigma_c
    bisect_iters=7,           # -> sigma_c to ~(3.5-1.0)/2^7 ≈ 0.02
    store_floor=1e-30,        # log-abundance floor for storage (re-floor at analysis time)
)

# ---- parameter grid ----
GRID = dict(
    f_values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],   # f = sigma / sigma_c
    n_seeds=50,
    base_seed=917165939,
)

# ---- parameter scan (python -m criticality_msb.simulate --sweep) ----
# Sweep physical axes (cartesian product) at reduced f/seeds to find a regime; refine later.
# Runs write to the same data/ (filenames encode every param), so overlap with the base grid is reused.
# (μ scan already run: axes={"mu":[0.5,1,2,4,8]}, f=[0.6,0.85] — data on disk.)
SWEEP = dict(
    axes={"lam": [0.0, 1e-3, 1e-2, 1e-1]},        # immigration: firms floor at λ instead of crashing to 0
    f_values=[0.9, 0.95],                          # candidate (0.9) + closer-to-critical (0.95) skew check
    n_seeds=50,                                    # solid statistics for the candidate
    sim=dict(sigma_bracket=(0.5, 5.0)),
)

# ---- smoke overrides (fast pipeline validation: python -m criticality_msb.simulate --smoke) ----
SMOKE_SIM = dict(S=400, tmax=150.0, dt_store=0.5, locate_T=120.0, rtol=1e-7, atol=1e-9)
SMOKE_GRID = dict(f_values=[0.5, 0.8, 0.95], n_seeds=2)
