# Fixed-μ high-σ volatility-vs-size sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/volatility_sigma_sweep.ipynb` that holds μ=0.2 fixed, sweeps σ∈[0.2…4.0], and plots the Moran–Secchi–Bouchaud growth-rate-volatility-vs-size relation at each σ — showing the V at the subcritical points and the **supercritical breakdown** (M diverges, physical-time clock dies) at high σ.

**Architecture:** A standalone notebook mirroring the sibling `volatility_finalsize.ipynb`/`volatility_v_cause.ipynb` conventions. All logic inline (no new library code). The measurement is the float64, shape-based `run_one` (relative size `S=N·y/Σy`, never forms `x=yM` → overflow-proof). Physical-time yearly resampling is kept (MSB-faithful). Each run is tagged subcritical/supercritical via `sol.status`. A parallel sweep pools per-firm `(avg_size, volatility, reached)`, caches to `.npz`, and renders a headline figure (solid subcritical V-curves + dashed supercritical "invalid" curves) plus a summary figure (breakdown fractions + physical-time-dies + MSB slopes). Built with `nbformat`, executed (outputs embedded) with `nbconvert`, both via `uv run --with` (no permanent dep changes).

**Tech Stack:** Python 3.14, numpy, scipy `solve_ivp`, networkx, matplotlib, the `glv` package (`apply_style`, `generate_matrix`, `rescaled_glv_sparse`); build/run via `nbformat` + `nbconvert` + the project's `ipykernel`.

**Spec:** `docs/superpowers/specs/2026-06-15-volatility-sigma-sweep-design.md` (see its "Update" section for the supercritical finding driving this plan).

---

## Mechanism notes (verified during planning)

- **Kernel:** registered into the project venv with
  `uv run python -m ipykernel install --sys-prefix --name python3 --display-name "glv (uv)"`
  (idempotent; creates `.venv/share/jupyter/kernels/python3`).
- **Execute:** `uv run --with nbconvert python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 <nb>` — verified to run a cell importing `glv`. CWD = notebook dir, so bare-name figures/`.npz` land in `notebooks/`.
- **Build:** `uv run --with nbformat python /tmp/build_volatility_sigma_sweep.py`.
- **Probe results (μ=0.2):** σ=0.2/0.5 subcritical (τ→1e6, t_phys~700 yr); σ≥0.8 supercritical (M overflows, t_phys~0.5–2 yr). Full sweep ≈ 70 integrations, est. wall-time ~15 s (high-σ runs fail fast). `sol.status==0` ⇔ subcritical.

---

## Task 1: Smoke-test the measurement logic + probe — ✅ DONE

Completed during planning. A scratch `/tmp/smoke_sigma_sweep.py` validated `run_one`/`vbin`/`two_slopes` at tiny params and probed timing. **Two findings, both folded into Task 2 below:**

1. **State-vector bug:** `run_one` must receive the full rescaled state `[y, M, t]` (length N+2), not a length-N abundance vector. Fix (matches `volatility_v_cause`): build it inside `run_one` via `np.concatenate([x0/x0.sum(), [x0.sum()], [0.0]])`.
2. **Supercritical breakdown:** at μ=0.2, σ≥0.8 diverges M; physical-time measurement degenerates. Decision (user): keep physical time, tag runs via `sol.status`, show the breakdown.

σ=0.2 gave a clean V (slope lo≈−0.9, hi≈+2.9). Timing trivial (<20 s full run). No commit (scratch).

---

## Task 2: Build the notebook (nbformat builder)

Assemble `notebooks/volatility_sigma_sweep.ipynb` from the corrected logic.

**Files:**
- Create: `notebooks/volatility_sigma_sweep.ipynb` (deliverable)
- Create (scratch, not committed): `/tmp/build_volatility_sigma_sweep.py`

- [ ] **Step 1: Write the builder script** with exactly this content:

```python
# /tmp/build_volatility_sigma_sweep.py  — assembles the deliverable notebook; throwaway
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "glv (uv)", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Growth-rate volatility vs size — fixed $\mu$, high-$\sigma$ sweep

Reproduces the **Moran–Secchi–Bouchaud** size–volatility relation, holding the mean interaction
$\mu=0.2$ **fixed** and sweeping the disorder $\sigma$ from the prior baseline ($\sigma=0.2$) up to
strong disorder ($\sigma=4.0$). For each $\sigma$ we plot per-firm growth-rate volatility
$\sigma_i=\sqrt{\pi/2}\,\overline{|g-\bar g|}$ against time-average relative size $\bar S_i$.

**What actually happens (the finding).** At $\mu=0.2$ the system is **subcritical only for
$\sigma\lesssim 0.6$** (total mass $M$ converges, physical time runs ~700 yr). For $\sigma\gtrsim 0.6$
it goes **supercritical**: $M$ diverges (overflows $\sim10^{308}$), the integrator stops early, and
physical time reaches only ~0.5–2 yr — so the physical-year MSB measurement **breaks down**. We keep
the physical-time clock and *show* this breakdown: subcritical $\sigma$ get a valid V-curve;
supercritical $\sigma$ are flagged (dashed/"invalid") and quantified in the summary.

**Caveats.**
- $\mu_c$ drifts with $\sigma$; $\mu=0.2$ is a held subcritical reference, not critical at every $\sigma$.
- Relative size $S=N\,y/\sum y$ is built from the **shape** $y$, never $x=yM$, so it cannot overflow.
- A run is **subcritical** iff the solver reaches $\tau_{\max}$ (`sol.status==0`); otherwise $M$
  diverged (supercritical). Failed integrations (`sol.t.size<3`) are dropped and counted.
- Physical-time volatility mixes transient and any persistent dynamics; this is descriptive and cannot
  attribute cause.""")

co(r"""import multiprocessing as mp
mp.set_start_method("fork", force=True)   # spawn re-imports the kernel and breaks the pool

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import solve_ivp

import glv
glv.apply_style()

N         = 1000                                  # species / firms
C         = 5                                     # mean degree
MU        = 0.2                                   # FIXED mean interaction (subcritical at low sigma)
SIGMAS    = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]   # disorder sweep: baseline -> strong
n_runs    = 10                                    # graph realizations per sigma (pooled)
tau_max   = 1e6                                   # rescaled-time horizon
n_years   = 100                                   # yearly resample grid (physical time)
FLOOR     = 1e-3                                  # live-firm threshold for binning
MIN_FIRMS = 100                                   # min firms to draw a curve for a sigma/regime
n_workers = min(8, (mp.cpu_count() or 2))
RES = "volatility_sigma_sweep_results.npz"
print(f"mu={MU} fixed | sigma grid={SIGMAS} | N={N} x n_runs={n_runs} per sigma | workers={n_workers}")""")

co(r"""# Degree law: exponential, kept inline (project convention: no library wrapper)
def exponential_degrees(C, rng):
    return np.maximum(rng.exponential(C, N).astype(int), 1)

def make_even(deg):
    deg = np.asarray(deg).copy()
    if deg.sum() % 2:
        deg[0] += 1
    return deg""")

co(r"""# Measurement: one float64 integration. Build the rescaled state [y (simplex), M, t]
# internally; relative size S = N*y/sum(y) comes from the SHAPE y, so x = y*M is never
# formed and the diverging M at high sigma cannot overflow.
#   reached == True  <=> solver reached tau_max (M finite) => SUBcritical, valid physical time
#   reached == False <=> integration stopped early (M diverged) => SUPERcritical breakdown
def run_one(W, x0):
    state0 = np.concatenate([x0 / x0.sum(), [x0.sum()], [0.0]])
    sol = solve_ivp(glv.rescaled_glv_sparse, (0.0, tau_max), state0, args=(N, W),
                    method="RK45", max_step=1e2)
    if sol.t.size < 3:
        return None
    reached = bool(sol.status == 0)
    y = sol.y[:N, :]
    t_phys = sol.y[N + 1, :]
    t_final = t_phys[-1]
    if not np.isfinite(t_final) or t_final <= t_phys[0]:
        return None
    dt = (t_final - t_phys[0]) / n_years
    t_years = np.arange(t_phys[0], t_final, dt)
    y_yearly = np.array([np.interp(t_years, t_phys, y[k]) for k in range(N)])
    totals = y_yearly.sum(axis=0, keepdims=True)
    totals = np.where(totals > 0, totals, np.nan)
    S = N * y_yearly / totals                         # = N x_i / sum_j x_j, M cancels
    log_S = np.log(np.maximum(S, 1e-15))
    g = np.diff(log_S, axis=1)
    g_bar = g.mean(axis=1, keepdims=True)
    sigma_i = np.sqrt(np.pi / 2.0) * np.mean(np.abs(g - g_bar), axis=1)
    return S.mean(axis=1), sigma_i, reached, float(t_final)

def _job(args):
    sigma, seed = args
    rng = np.random.default_rng(seed)
    W = glv.generate_matrix(make_even(exponential_degrees(C, rng)), C, mu=MU, sigma=sigma)
    return sigma, run_one(W, rng.uniform(0.1, 1.0, N))

def vbin(x, y, nb=22):
    o = np.argsort(x); xp, yp = x[o], y[o]
    parts = np.array_split(np.arange(xp.size), nb)
    return (np.array([xp[p].mean() for p in parts]),
            np.array([np.median(yp[p]) for p in parts]))

def two_slopes(x, y):
    bx, by = vbin(x, y)
    j = int(np.argmin(by))
    if j < 1 or j > len(bx) - 2:                      # no interior V-min -> single fit
        s = np.polyfit(np.log10(bx), np.log10(by), 1)[0]
        return bx[j], s, s
    lo = np.polyfit(np.log10(bx[:j + 1]), np.log10(by[:j + 1]), 1)[0]
    hi = np.polyfit(np.log10(bx[j:]), np.log10(by[j:]), 1)[0]
    return bx[j], lo, hi""")

co(r"""# Sweep over (sigma, run); pool per-firm (avg_size, volatility, reached). Cache to .npz.
# Failed integrations are dropped and COUNTED; sub/supercritical runs counted separately.
if os.path.exists(RES):
    d = np.load(RES)
    sigmas, avg_all, sig_all = d["sigmas"], d["avg_all"], d["sig_all"]
    sig_of, reached_of = d["sig_of"], d["reached_of"]
    dropped, n_sub, n_super, tfin_med = d["dropped"], d["n_sub"], d["n_super"], d["tfin_med"]
    print(f"loaded cache {RES}")
else:
    jobs = [(s, 1000 + 100 * si + r) for si, s in enumerate(SIGMAS) for r in range(n_runs)]
    s_index = {s: i for i, s in enumerate(SIGMAS)}
    avg_list, sig_list, of_list, reach_list = [], [], [], []
    dropped = np.zeros(len(SIGMAS), int)
    n_sub = np.zeros(len(SIGMAS), int)
    n_super = np.zeros(len(SIGMAS), int)
    tfin = [[] for _ in SIGMAS]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for done, fut in enumerate(as_completed(futs), 1):
            sigma, res = fut.result()
            i = s_index[sigma]
            if res is None:
                dropped[i] += 1; tag = "drop"
            else:
                a, s, reached, tf = res
                avg_list.append(a); sig_list.append(s)
                of_list.append(np.full(a.size, sigma))
                reach_list.append(np.full(a.size, reached))
                tfin[i].append(tf)
                if reached: n_sub[i] += 1
                else:       n_super[i] += 1
                tag = "ok(sub)" if reached else "ok(SUPER)"
            print(f"  [{done}/{len(jobs)}] sigma={sigma}: {tag}")
    sigmas = np.array(SIGMAS, float)
    avg_all = np.concatenate(avg_list)
    sig_all = np.concatenate(sig_list)
    sig_of = np.concatenate(of_list)
    reached_of = np.concatenate(reach_list)
    tfin_med = np.array([np.median(t) if t else np.nan for t in tfin])
    np.savez(RES, sigmas=sigmas, avg_all=avg_all, sig_all=sig_all, sig_of=sig_of,
             reached_of=reached_of, dropped=dropped, n_sub=n_sub, n_super=n_super,
             tfin_med=tfin_med)
    print(f"saved {RES}")

print("\nper-sigma report (sub=subcritical/valid, SUPER=supercritical/M diverged):")
for i, s in enumerate(sigmas):
    mm = (sig_of == s) & reached_of
    coll = float(np.mean(avg_all[mm] < FLOOR)) if mm.any() else float("nan")
    print(f"  sigma={s:4.2f}: sub={int(n_sub[i])} SUPER={int(n_super[i])} dropped={int(dropped[i])} "
          f"| median t_phys={tfin_med[i]:9.3g} yr | collapsed(<{FLOOR:g})={100*coll:5.1f}%")""")

co(r"""# Headline: size-volatility V at each sigma. Solid = subcritical (valid physical-time);
# dashed/faded = supercritical (M diverged -> physical-time data invalid, shown to expose breakdown).
fig, ax = plt.subplots(figsize=(8.0, 6))
cmap = plt.cm.viridis(np.linspace(0.08, 0.92, len(sigmas)))
onset = None
for s, col in zip(sigmas, cmap):
    base = (sig_of == s) & (avg_all > FLOOR) & (sig_all > 0) & np.isfinite(sig_all)
    sub = base & reached_of
    sup = base & ~reached_of
    if sub.sum() >= MIN_FIRMS:
        bx, by = vbin(avg_all[sub], sig_all[sub])
        ax.loglog(bx, by, "o-", ms=4, color=col, label=rf"$\sigma={s:g}$")
    else:
        if onset is None:
            onset = s
        if sup.sum() >= MIN_FIRMS:
            bx, by = vbin(avg_all[sup], sig_all[sup])
            ax.loglog(bx, by, "x--", ms=4, color=col, alpha=0.45,
                      label=rf"$\sigma={s:g}$ (supercrit.)")
title = rf"Size--volatility vs disorder $\sigma$  ($\mu={MU}$ fixed)"
if onset is not None:
    title += f"\nsupercritical from $\\sigma\\approx{onset:g}$ (dashed = invalid physical-time data)"
ax.set(xlabel=r"time-average relative size $\bar S_i$",
       ylabel=r"growth-rate volatility $\sigma_i=\sqrt{\pi/2}\,\overline{|g-\bar g|}$", title=title)
ax.grid(True, which="both", alpha=0.15)
ax.legend(title=r"disorder $\sigma$", ncol=2, framealpha=0.9, fontsize=8)
plt.tight_layout()
plt.savefig("volatility_sigma_sweep.png", dpi=120)
plt.show()""")

co(r"""# Summary vs sigma: (1) breakdown fractions, (2) the physical-time clock dying, (3) MSB slopes.
super_frac = np.array([n_super[i] / max(int(n_sub[i] + n_super[i]), 1) for i in range(len(sigmas))])
drop_frac = np.array([dropped[i] / max(int(dropped[i] + n_sub[i] + n_super[i]), 1)
                      for i in range(len(sigmas))])
slopes_lo, slopes_hi, coll_frac = [], [], []
for i, s in enumerate(sigmas):
    sub = (sig_of == s) & reached_of & (avg_all > FLOOR) & (sig_all > 0) & np.isfinite(sig_all)
    if sub.sum() >= MIN_FIRMS:
        _, lo, hi = two_slopes(avg_all[sub], sig_all[sub])
        slopes_lo.append(lo); slopes_hi.append(hi)
    else:
        slopes_lo.append(np.nan); slopes_hi.append(np.nan)
    mm = (sig_of == s) & reached_of
    coll_frac.append(float(np.mean(avg_all[mm] < FLOOR)) if mm.any() else np.nan)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(sigmas, 100 * super_frac, "o-", color="#c1121f", label="supercritical runs")
ax[0].plot(sigmas, 100 * np.array(coll_frac), "^-", color="#e07a5f",
           label=r"collapsed firms ($\bar S<10^{-3}$)")
ax[0].plot(sigmas, 100 * drop_frac, "s--", color="#888", label="dropped runs")
ax[0].set(xlabel=r"disorder $\sigma$", ylabel="percent", ylim=(-3, 103),
          title=r"Breakdown fractions vs $\sigma$")
ax[0].legend(fontsize=8)
ax[1].semilogy(sigmas, tfin_med, "o-", color="#1d3557")
ax[1].set(xlabel=r"disorder $\sigma$", ylabel=r"median physical time reached (yr)",
          title=r"Physical-time clock dies as $M$ diverges")
ax[2].plot(sigmas, slopes_lo, "o-", label=r"descending $\beta_{\rm lo}$")
ax[2].plot(sigmas, slopes_hi, "s--", label=r"ascending $\beta_{\rm hi}$")
ax[2].axhline(0, color="#888", lw=1, ls=":")
ax[2].set(xlabel=r"disorder $\sigma$", ylabel=r"log--log slope $\beta$",
          title=r"MSB branch slopes (subcritical $\sigma$ only)")
ax[2].legend()
plt.tight_layout()
plt.savefig("volatility_sigma_sweep_summary.png", dpi=120)
plt.show()""")

md(r"""## Summary

_To be filled after the run (Task 4) from the per-sigma report and the two figures: the V at the
subcritical $\sigma$, the supercritical onset, how the physical-time clock and the MSB measurement
break down as $\sigma$ grows, and the collapse/dropped fractions._""")

nb.cells = cells
with open("notebooks/volatility_sigma_sweep.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/volatility_sigma_sweep.ipynb with", len(cells), "cells")
```

- [ ] **Step 2: Run the builder**

Run: `uv run --with nbformat python /tmp/build_volatility_sigma_sweep.py`
Expected: `wrote notebooks/volatility_sigma_sweep.ipynb with 8 cells` (2 markdown + 6 code)

- [ ] **Step 3: Validate the notebook is well-formed**

Run: `uv run --with nbformat python -c "import nbformat; nb=nbformat.read('notebooks/volatility_sigma_sweep.ipynb', as_version=4); nbformat.validate(nb); print('valid', len(nb.cells), 'cells')"`
Expected: `valid 8 cells`

- [ ] **Step 4: Commit the unexecuted notebook**

```bash
git add notebooks/volatility_sigma_sweep.ipynb
git commit -m "feat(notebooks): fixed-mu high-sigma volatility-vs-size sweep (unexecuted)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Execute the notebook (full sweep, embed outputs)

**Files:**
- Modify (executed in place): `notebooks/volatility_sigma_sweep.ipynb`
- Create (artifacts): `notebooks/volatility_sigma_sweep_results.npz`, `notebooks/volatility_sigma_sweep.png`, `notebooks/volatility_sigma_sweep_summary.png`

- [ ] **Step 1: Ensure the kernel is registered (idempotent)**

Run: `uv run python -m ipykernel install --sys-prefix --name python3 --display-name "glv (uv)"`
Expected: `Installed kernelspec python3 in .../.venv/share/jupyter/kernels/python3`

- [ ] **Step 2: Execute the notebook**

Run:
```bash
uv run --with nbconvert python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 \
  notebooks/volatility_sigma_sweep.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes`, no traceback (a `MissingIDFieldWarning` is harmless). The probe says full run ≈15 s. If interrupted, re-run — the sweep cell loads the partial `.npz`.

- [ ] **Step 3: Verify artifacts and outputs**

Run:
```bash
cd /Users/ludovicofurlanetto/Code/glv && ls -la notebooks/volatility_sigma_sweep.png notebooks/volatility_sigma_sweep_summary.png notebooks/volatility_sigma_sweep_results.npz && uv run python -c "
import json, numpy as np
d = np.load('notebooks/volatility_sigma_sweep_results.npz')
print('sigmas   :', d['sigmas'])
print('n_sub    :', d['n_sub'])
print('n_super  :', d['n_super'])
print('dropped  :', d['dropped'])
print('tfin_med :', d['tfin_med'])
nb = json.load(open('notebooks/volatility_sigma_sweep.ipynb'))
nout = sum(1 for c in nb['cells'] if c.get('cell_type')=='code' and c.get('outputs'))
ncode = sum(1 for c in nb['cells'] if c.get('cell_type')=='code')
errs = [o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']
print(f'code cells with outputs: {nout}/{ncode}; errors: {len(errs)}')
assert len(errs)==0, 'notebook has error outputs'
"
```
Expected: all three files exist; `n_sub` nonzero for σ=0.2 and 0.5 and ≈0 for σ≥1.0; `n_super` the reverse; `tfin_med` ~hundreds for low σ, ~1 for high σ; `code cells with outputs: 4/6` (the two pure-definition cells emit no output — that's fine); `errors: 0`.

- [ ] **Step 4: Eyeball the figures**

Read `notebooks/volatility_sigma_sweep.png` and `notebooks/volatility_sigma_sweep_summary.png`. Confirm: σ=0.2/0.5 show solid V-curves; supercritical σ appear dashed/faded (or absent); the summary's "physical-time clock dies" panel drops from ~100s of yr to ~1 yr, and supercritical-% rises to 100. Note the specifics for Task 4.

- [ ] **Step 5: Commit the executed notebook + artifacts**

```bash
git add notebooks/volatility_sigma_sweep.ipynb notebooks/volatility_sigma_sweep_results.npz notebooks/volatility_sigma_sweep.png notebooks/volatility_sigma_sweep_summary.png
git commit -m "feat(notebooks): execute high-sigma volatility sweep (figures + cache)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Fill the summary and finalize

**Files:**
- Modify: `/tmp/build_volatility_sigma_sweep.py` (the summary `md(...)` cell only)
- Modify (rebuilt + re-executed): `notebooks/volatility_sigma_sweep.ipynb`

- [ ] **Step 1: Write the findings into the summary cell**

In `/tmp/build_volatility_sigma_sweep.py`, replace the final `md(r"""## Summary … """)` block with concrete observations from Task 3 Steps 3–4: the V at σ=0.2/0.5 (slopes), the supercritical onset (~σ0.6, where `n_sub`→0), the physical-time collapse (`tfin_med` numbers), and the collapse/dropped fractions. State plainly that the MSB physical-time measurement breaks above the onset (descriptive; no cause attribution). No placeholder text.

- [ ] **Step 2: Rebuild the notebook**

Run: `uv run --with nbformat python /tmp/build_volatility_sigma_sweep.py`
Expected: `wrote notebooks/volatility_sigma_sweep.ipynb with 8 cells` (keeps the `.npz`, so the next execute is fast).

- [ ] **Step 3: Re-execute (cache hit)**

Run:
```bash
uv run --with nbconvert python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 \
  notebooks/volatility_sigma_sweep.ipynb
```
Expected: finishes in seconds (`loaded cache ...`); no traceback.

- [ ] **Step 4: Verify no error outputs and summary present**

Run:
```bash
uv run python -c "
import json
nb = json.load(open('notebooks/volatility_sigma_sweep.ipynb'))
errs = [o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']
summ = ''.join(nb['cells'][-1]['source'])
assert len(errs)==0, 'error outputs present'
assert 'To be filled' not in summ, 'summary placeholder still present'
print('ok: no errors, summary filled')
"
```
Expected: `ok: no errors, summary filled`

- [ ] **Step 5: Commit**

```bash
git add notebooks/volatility_sigma_sweep.ipynb
git commit -m "docs(notebooks): write findings summary for high-sigma volatility sweep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:**
- Descriptive sweep, μ=0.2 fixed, σ=[0.2…4.0] (7 pts) → Task 2 setup cell. ✓
- Inline float64 shape-based `run_one` (no library), **state built internally** (N+2) → Task 2. ✓
- Physical-time kept; **subcritical/supercritical tag via `sol.status`** → Task 2 run_one + sweep. ✓
- Drop + count failed runs; per-σ sub/super counts, median physical-time, collapse → Task 2 report. ✓
- `.npz` cache (load if present) with `reached_of`/`n_sub`/`n_super`/`tfin_med` → Task 2 sweep cell. ✓
- Headline: solid subcritical V-curves + dashed supercritical "invalid" curves → Task 2 plot 1. ✓
- Summary: breakdown fractions, physical-time-dies, MSB slopes (subcritical) → Task 2 plot 2. ✓
- Inline `vbin`, `two_slopes` (hardened) ; LaTeX mathtext, `apply_style`, seeded RNG → Task 2. ✓
- Executed notebook with outputs → Tasks 3–4; caveats in intro markdown → Task 2 cell 0. ✓

**Placeholder scan:** summary cell ships a placeholder only between Task 2 and Task 4; Task 4 Step 4 asserts it is gone before the final commit. No placeholders in plan code. ✓

**Type/name consistency:** `run_one`→`(avg_size, sigma_i, reached, t_final)`; `_job`→`(sigma, res)`; sweep stores/loads `sigmas, avg_all, sig_all, sig_of, reached_of, dropped, n_sub, n_super, tfin_med`; every plot/report reads those exact names; `MIN_FIRMS`/`FLOOR`/`two_slopes`/`vbin` defined once, used consistently. ✓
