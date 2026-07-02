# Relative-GLV Presentation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `../relative-glv/story.ipynb` into a self-contained, GitHub-facing presentation of the relative GLV model where every figure uses one consistent palette.

**Architecture:** Enhance the existing notebook in place. Add one `PALETTE` + `rcParams` setup so matplotlib's `C0..C3` cycle maps to the palette automatically (the lazy unification); fix the handful of hard-coded off-palette hexes by hand. Drop the retired finite-economy/β-vs-N narrative. Add a D2-multiscaling figure and a two-time-DMFT appendix figure (the two-time solver is ported into the package so the standalone repo stays reproducible).

**Tech Stack:** Python, numpy, scipy, matplotlib, Jupyter; `uv` for execution; the `relative_glv` package.

## Global Constraints

- All commands run from `/Users/ludovicofurlanetto/Code/relative-glv` unless stated.
- Always use `uv run python` / `uv run pytest`, never bare `python`.
- **Own-degree only.** Do not reintroduce mean-degree / finite-economy / β-scaling framing.
- One palette everywhere: `navy #1d3557`, `blue #457b9d`, `coral #e76f51`, `teal #2a9d8f`; fits/empirical = coral, analytic/null lines = black dashed/dotted, neutral = grays.
- Notebook must run top-to-bottom with no errors under `nbconvert --execute`.
- Edit notebook cells with the NotebookEdit tool. Reference cells by their content marker (a distinctive first line), not by a fixed index — indices shift as cells are added/removed. **Do all deletions before additions.**

---

### Task 1: Palette + rcParams in the setup cell

**Files:**
- Modify: `../relative-glv/story.ipynb` — code cell 0 (starts `import numpy as np`)

**Interfaces:**
- Produces: module-level names `PALETTE` (dict), `MODEL`, `EMPIRICAL`, `REFERENCE`, `NEUTRAL`, `REGIME` — used by every later figure cell.

- [ ] **Step 1: Replace the setup cell body** with:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from matplotlib.lines import Line2D
from cycler import cycler

%matplotlib inline

# --- one palette for the whole notebook -------------------------------------
PALETTE = {"navy": "#1d3557", "blue": "#457b9d", "coral": "#e76f51", "teal": "#2a9d8f"}
MODEL      = PALETTE["navy"]     # primary model series / C0
EMPIRICAL  = PALETTE["coral"]    # fits and empirical reference
REFERENCE  = "black"             # analytic / null distribution lines (dashed/dotted)
NEUTRAL    = "0.6"               # scatter clouds, floors, grids
REGIME     = ["#e8e8e8", PALETTE["blue"], PALETTE["coral"], PALETTE["navy"]]  # frozen,fluct,shrink,diverg

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.prop_cycle": cycler(color=[PALETTE["navy"], PALETTE["blue"],
                                     PALETTE["coral"], PALETTE["teal"]]),
})

import pathlib
DATA = pathlib.Path("data")
```

- [ ] **Step 2: Verify the setup cell runs** — execute only cell 0 in Jupyter (or `uv run python -c "from cycler import cycler"` to confirm the dep is available). Expected: no error. If `cycler` import fails, use `plt.cycler` instead (bundled with matplotlib).

- [ ] **Step 3: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "feat(notebook): single palette + prop_cycle in setup"
```

---

### Task 2: Drop the β-vs-N section

**Files:**
- Modify: `../relative-glv/story.ipynb` — remove markdown cell "### $\beta$ vs $N$: sustained..." and code cell 8 (starts `# beta vs N for the OWN-DEGREE model`)

**Interfaces:**
- Consumes: nothing.
- Produces: removes the only use of `data/beta_limit.npz` in the notebook.

- [ ] **Step 1: Delete the two cells** (the `### β vs N` markdown header and the `# beta vs N ...` code cell) with NotebookEdit.

- [ ] **Step 2: Verify** no remaining reference — search the notebook JSON:

```bash
cd ../relative-glv && uv run python -c "import json; s=json.dumps(json.load(open('story.ipynb'))); assert 'beta_limit' not in s and 'beta vs N' not in s and 'N-robust' not in s; print('clean')"
```
Expected: `clean`

- [ ] **Step 3: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "docs(notebook): drop beta-vs-N (finite-economy story retired)"
```

---

### Task 3: Restyle the live MSB cells (4 and 6) to the palette

**Files:**
- Modify: `../relative-glv/story.ipynb` — code cell 4 (`from relative_glv import coupling, integrate, growth_rate`) and code cell 6 (`# Compute the two MSB facts LIVE`)

**Interfaces:**
- Consumes: `MODEL`, `EMPIRICAL`, `REFERENCE`, `NEUTRAL` from Task 1.

The `C0` references now already resolve to navy via the prop_cycle, so only the explicit `"r--"`, `"k:"`, and `"0.6"` markers need aligning. Make the changes below in each cell.

- [ ] **Step 1: Cell 4** — the aggregate/relative-size plot uses `color="C0"` (line 1). Leave it (cycle → navy). No other color literals. No change needed; skip to cell 6.

- [ ] **Step 2: Cell 6 replacements** (exact string swaps):
  - `ax[0].scatter(sv["Sbar"], sv["vol"], s=5, alpha=0.15, color="0.6")` → `color=NEUTRAL`
  - `ax[0].plot(bx, by, "o-", color="C0", ms=5, label="binned median")` → `color=MODEL`
  - the fit line `ax[0].plot(xx, 10 ** np.polyval(c, np.log10(xx)), "r--", lw=2,` → `color=EMPIRICAL, ls="--", lw=2,`
  - `ax[1].hist(z, bins=160, density=True, color="C0", alpha=0.6, label="growth rates")` → `color=MODEL`
  - `ax[1].plot(xx, laplace.pdf(...), "r--", lw=1.5, label="Laplace")` → `color=EMPIRICAL, ls="--", lw=1.5,`
  - `ax[1].plot(xx, norm.pdf(xx), "k:", lw=1.2, label="Gaussian")` → `color=REFERENCE, ls=":", lw=1.2,`

- [ ] **Step 3: Verify** the cell parses — run cell 6 in Jupyter. Expected: figure renders, no `NameError`.

- [ ] **Step 4: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "style(notebook): palette for live MSB-facts figures"
```

---

### Task 4: Restyle the discriminating-tests cell (10) to palette names

**Files:**
- Modify: `../relative-glv/story.ipynb` — code cell 10 (`# MSB's three DISCRIMINATING tests`)

**Interfaces:**
- Consumes: `PALETTE`, `REFERENCE` from Task 1.

This cell already uses the right hexes; swap the literals for names so there is one source of truth.

- [ ] **Step 1: Replace** the two color tuples:
  - `("#1d3557", "#457b9d", "#e76f51")` (used in D1 and D3 loops) → `(PALETTE["navy"], PALETTE["blue"], PALETTE["coral"])`
  - `("#1d3557", "#457b9d", "#e76f51", "#2a9d8f")` (D2 loop) → `(PALETTE["navy"], PALETTE["blue"], PALETTE["coral"], PALETTE["teal"])`
  - the Gaussian null `ax[2].semilogy(xx, norm.pdf(xx), "k--", lw=1, label="Gaussian")` → `color=REFERENCE, ls="--", lw=1,`

- [ ] **Step 2: Verify** — run cell 10. Expected: three-panel figure renders identically to before.

- [ ] **Step 3: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "style(notebook): palette names in discriminating-tests cell"
```

---

### Task 5: Add the D2 multiscaling figure

**Files:**
- Modify: `../relative-glv/story.ipynb` — add one markdown + one code cell immediately after code cell 10
- Reference: `scripts/plot_growing_multiscaling.py` (source of the normalized-exponent panel), `data/msb_conditional.npz`

**Interfaces:**
- Consumes: `PALETTE`, `NEUTRAL`, `DATA` from Task 1. npz keys `cq` (ζ_1..4).

- [ ] **Step 1: Insert a markdown cell** after cell 10:

```markdown
### Multiscaling (MSB test D2, sharpened)

The moment exponents $\zeta_q$ of $E[\sigma^q\mid S]\sim S^{-\zeta_q}$ rise with $q$ —
the anti-granular signature. Normalised by $\zeta_1$ and compared against the
Moran-Santos-Bouchaud data and the granular ($q$-independent) null:
```

- [ ] **Step 2: Insert the code cell** after that:

```python
# D2 sharpened: normalized moment exponents zeta_q/zeta_1 vs q, model vs MSB vs granular null.
d = np.load(DATA / "msb_conditional.npz")
cq = d["cq"]                                  # model zeta_1..4
MSB = np.array([0.20, 0.39, 0.51, 0.58])      # Moran-Santos-Bouchaud 2024
q = np.arange(1, 5)

fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.plot(q, cq / cq[0],  "o-", color=PALETTE["navy"], lw=2, ms=7, label=r"relative GLV (own-degree)")
ax.plot(q, MSB / MSB[0], "s--", color=EMPIRICAL, lw=2, ms=7, label="MSB data")
ax.plot(q, q / q[0],     ":",  color=NEUTRAL, lw=2, label="granular null ($\\zeta_q\\propto q$)")
ax.axhline(1.0, color=NEUTRAL, lw=0.6)
ax.set(xlabel="$q$", ylabel=r"$\zeta_q/\zeta_1$", xticks=q,
       title="D2: multiscaling — exponents rise sub-linearly with $q$")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
print(f"model zeta_q/zeta_1 = {np.round(cq/cq[0], 2)}  |  MSB = {np.round(MSB/MSB[0], 2)}  "
      "|  granular would be 1,2,3,4")
```

- [ ] **Step 3: Verify** — run the new cell. Expected: figure renders, print line shows model and MSB ratios.

- [ ] **Step 4: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "feat(notebook): add D2 multiscaling figure"
```

---

### Task 6: Restyle the DMFT cells (16, 18) to the palette

**Files:**
- Modify: `../relative-glv/story.ipynb` — code cell 16 (`from relative_glv import solve_fixed_point, sigma_c`) and code cell 18 (`dv = np.load(DATA / "dmft_validation.npz"`)

**Interfaces:**
- Consumes: `PALETTE`, `EMPIRICAL`, `REFERENCE` from Task 1.

Both cells use an off-palette ad-hoc set (`#1f6f8b`, `#c1121f`, `#e07a18`, `#3b5b8c`, `#2a9d8f`). Map consistently: growth/g-curve → `PALETTE["navy"]`; survival φ → `PALETTE["teal"]`; simulation points → `PALETTE["blue"]`; `σ_c` line and unstable-FP → `EMPIRICAL`; zero/grid lines → `REFERENCE` or `NEUTRAL`.

- [ ] **Step 1: Cell 16 replacements:**
  - `color="#1f6f8b"` (both `g_0` / `g0_arr` curves) → `color=PALETTE["navy"]`
  - `color="#c1121f"` (both `σ_c` axhline/axvline and its text) → `color=EMPIRICAL`
  - `color="#e07a18"` (φ curves) → `color=PALETTE["teal"]`
  - region label texts `color="#2a9d8f"` and `color="#3b5b8c"` → `color=PALETTE["teal"]` and `color=PALETTE["blue"]`

- [ ] **Step 2: Cell 18 replacements:**
  - `color="#2a9d8f"` (all three sim `"o-"` series) → `color=PALETTE["blue"]`
  - `color="#c1121f"` (DMFT `g*`, φ, and the `σ_c` axvline/text) → `color=EMPIRICAL`
  - `color="k"` / `color="grey"` guide lines → `color=REFERENCE` / `color=NEUTRAL`

- [ ] **Step 3: Verify** — run cells 16 and 18. Expected: both render; cell 18 prints the relaxed-phase mean errors.

- [ ] **Step 4: Commit**

```bash
cd ../relative-glv && git add story.ipynb && git commit -m "style(notebook): palette for DMFT phase + validation figures"
```

---

### Task 7: Recolor the phase-diagram regimes and replot the embedded PNG

**Files:**
- Modify: `../relative-glv/scripts/phase_owndeg_fine.py:111` (the `COLS` list)
- Regenerate: `data/phase_owndeg_fine.png` (and the tagged variant if the notebook shows it)

**Interfaces:**
- Consumes: nothing (standalone script). The notebook cell 12 embeds `data/phase_owndeg_fine.png`.

- [ ] **Step 1: Confirm which PNG the notebook embeds** — cell 12 does `Image(filename=str(DATA / "phase_owndeg_fine.png"))`. Recolor + replot that file.

- [ ] **Step 2: Edit the regime colors** at `scripts/phase_owndeg_fine.py:111`:

```python
    COLS = ["#e8e8e8", "#457b9d", "#e76f51", "#1d3557"]   # frozen, fluctuating, shrinking, divergent
```
(was `["#e8e8e8", "#4878a8", "#c0744f", "#3a3a3a"]`.)

- [ ] **Step 3: Replot from the existing npz (no recompute):**

```bash
cd ../relative-glv && uv run python scripts/phase_owndeg_fine.py --plot
```
Expected: rewrites `data/phase_owndeg_fine.png` in seconds (reads the npz; no 2-3h sim). If it errors that the default npz is missing, replot the tagged grid instead: `uv run python scripts/phase_owndeg_fine.py --grid 20 --seeds 10 --plot` and point cell 12 at the produced PNG.

- [ ] **Step 4: Verify** — re-run notebook cell 12. Expected: phase diagram shows blue fluctuating / coral shrinking / navy divergent regions.

- [ ] **Step 5: Commit**

```bash
cd ../relative-glv && git add scripts/phase_owndeg_fine.py data/phase_owndeg_fine.png && git commit -m "style(phase): palette regime colors + replot"
```

---

### Task 8: Port the two-time DMFT solver into the package

**Files:**
- Read: `../glv/growing_glv/dmft_twotime.py` (source of `solve_twotime`)
- Modify: `../relative-glv/relative_glv/dmft.py` (add `solve_twotime`), `../relative-glv/relative_glv/__init__.py` (export it)
- Test: `../relative-glv/tests/test_dmft_solver.py`

**Interfaces:**
- Consumes: `solve_fixed_point`, `sigma_c` (already in `relative_glv/dmft.py`).
- Produces: `solve_twotime(mu, sigma, *, seed=0, ...) -> dict` with keys at least `Ctau` (1-D autocorrelation), `dt` (float), `surv` (float), `g_eff` (float). Signature and return keys MUST match what `growing_glv/dmft_twotime_fig.py` consumes (`r["Ctau"]`, `r["dt"]`, `sol[s]["surv"]`, `sol[s]["g_eff"]`).

- [ ] **Step 1: Copy `solve_twotime` and its helpers** (`cholesky_psd`, `draw_noise`, `integrate`-for-ensemble, and whatever `solve_twotime` calls) from `growing_glv/dmft_twotime.py` into `relative_glv/dmft.py`. Rename the local ensemble `integrate` to `_twotime_integrate` to avoid clashing with the model's `integrate`. Keep the `from .dmft import solve_fixed_point, sigma_c` references as intra-module calls (they are already in this file).

- [ ] **Step 2: Export it** — in `relative_glv/__init__.py` add `solve_twotime` to the imports from `.dmft` and to `__all__`.

- [ ] **Step 3: Write the failing test** in `tests/test_dmft_solver.py`:

```python
def test_solve_twotime_freezes_below_and_decorrelates_above_sigma_c():
    from relative_glv import solve_twotime, sigma_c
    sc = sigma_c(0.0)
    relaxed = solve_twotime(-0.5, 1.0, seed=0)   # sigma < sqrt(2): autocorr should stay high
    fluct   = solve_twotime(-0.5, 2.0, seed=0)   # sigma > sqrt(2): autocorr should decay
    for r in (relaxed, fluct):
        assert {"Ctau", "dt", "surv", "g_eff"} <= set(r)
        assert r["dt"] > 0 and r["Ctau"][0] > 0
    # connected autocorrelation at the last lag: near 1 (frozen) vs clearly decayed
    conn = lambda r: (r["Ctau"][-1] - 1.0) / (r["Ctau"][0] - 1.0)
    assert conn(relaxed) > conn(fluct)
```

- [ ] **Step 4: Run it to confirm it fails first** (before Steps 1-2 are done) or, if implementing in order, run and confirm it passes:

```bash
cd ../relative-glv && uv run pytest tests/test_dmft_solver.py::test_solve_twotime_freezes_below_and_decorrelates_above_sigma_c -v
```
Expected after the port: PASS. (If the solver is slow, keep the test at these two σ only.)

- [ ] **Step 5: Commit**

```bash
cd ../relative-glv && git add relative_glv/dmft.py relative_glv/__init__.py tests/test_dmft_solver.py && git commit -m "feat(dmft): port two-time solver into the package"
```

---

### Task 9: Precompute two-time data and add the appendix figure

**Files:**
- Modify: `../relative-glv/scripts/compute.py` (add a two-time block writing `data/dmft_twotime.npz`)
- Create: `../relative-glv/data/dmft_twotime.npz`
- Modify: `../relative-glv/story.ipynb` — add a markdown + code cell in the appendix (after the DMFT-validation cell 18)

**Interfaces:**
- Consumes: `solve_twotime` (Task 8), `solve_fixed_point`, `sigma_c`, `data/dmft_validation.npz`.
- npz layout: `sig_2t` (array of σ), `Ctau` (object array of per-σ autocorr curves), `dt` (array), `surv` (array), `g_eff` (array).

- [ ] **Step 1: Add to `scripts/compute.py`** a block that mirrors `growing_glv/dmft_twotime_fig.py`'s computation and saves an npz:

```python
# --- two-time DMFT (fluctuating phase), for the notebook appendix ---
from relative_glv import solve_twotime
MU_2T = -0.5
sig_2t = [0.7, 1.0, 1.2, 1.6, 2.0, 2.5]
sol = [solve_twotime(MU_2T, s, seed=0) for s in sig_2t]
np.savez(DATA / "dmft_twotime.npz",
         sig_2t=np.array(sig_2t),
         Ctau=np.array([r["Ctau"] for r in sol], dtype=object),
         dt=np.array([r["dt"] for r in sol]),
         surv=np.array([r["surv"] for r in sol]),
         g_eff=np.array([r["g_eff"] for r in sol]),
         mu=MU_2T)
print("wrote dmft_twotime.npz")
```
(Match `DATA` to however `compute.py` already defines its data path.)

- [ ] **Step 2: Run it** to produce the npz:

```bash
cd ../relative-glv && uv run python scripts/compute.py
```
Expected: prints `wrote dmft_twotime.npz`; `data/dmft_twotime.npz` exists. (If `compute.py` runs many heavy blocks, temporarily guard the others or run just the two-time block in a `python -c`; do not recompute the multi-hour phase grid.)

- [ ] **Step 3: Add the appendix markdown cell** after cell 18:

```markdown
### Two-time DMFT: the fluctuating phase (fully-connected Gaussian limit)

The static fixed point is only valid below $\sigma_c=\sqrt2$. Above it the shares
never freeze and the theory must solve the whole autocorrelation $C(\tau)$
self-consistently. This appendix figure is the **fully-connected Gaussian limit**
(not the own-degree sparse network used above): it shows $C(\tau)$ freezing below
$\sigma_c$ and decorrelating above it, and confirms the two-time survival/growth
track direct simulation where the static FP overshoots.
```

- [ ] **Step 4: Add the appendix code cell:**

```python
# Two-time DMFT (FC-Gaussian limit): autocorrelation freeze/decorrelation + tracked observables.
d2 = np.load(DATA / "dmft_twotime.npz", allow_pickle=True)
sig_2t = list(d2["sig_2t"]); sc = sigma_c(0.0)
dv = np.load(DATA / "dmft_validation.npz", allow_pickle=True)
sim = {float(s): r for s, r in zip(dv["sigmas"], dv["sim"])}
idx = {s: i for i, s in enumerate(sig_2t)}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# left: connected autocorrelation, four sigmas across the transition
cyc = [PALETTE["teal"], PALETTE["blue"], PALETTE["coral"], PALETTE["navy"]]
for s, col in zip([1.0, 1.6, 2.0, 2.5], cyc):
    r = d2["Ctau"][idx[s]]; tau = np.arange(len(r)) * float(d2["dt"][idx[s]])
    conn = (r - 1.0) / (r[0] - 1.0)
    ax[0].plot(tau, conn, color=col, lw=2, label=fr"$\sigma={s}$" + (" (relaxed)" if s < sc else ""))
ax[0].axhline(0, color=NEUTRAL, lw=0.6)
ax[0].set(xlabel=r"lag $\tau$", ylabel=r"$[C(\tau)-1]/[C(0)-1]$",
          title="Share autocorrelation", ylim=(-0.05, 1.05)); ax[0].legend(fontsize=8)

# centre: survival, two-time vs sim vs static FP
sg = np.linspace(0.4, 2.5, 120)
phi_stat = np.array([solve_fixed_point(0.5, float(s))["phi"] for s in sg])
ax[1].plot(sg[sg < sc], phi_stat[sg < sc], color=EMPIRICAL, lw=2, label=r"static FP $\varphi$")
ax[1].plot(sg[sg >= sc], phi_stat[sg >= sc], color=EMPIRICAL, lw=2, ls=":", label="static FP (unstable)")
ax[1].plot(sig_2t, d2["surv"], "s", color=PALETTE["navy"], ms=8, label="two-time DMFT")
ss = [s for s in sig_2t if s in sim]
ax[1].plot(ss, [sim[s][1] for s in ss], "o", color=PALETTE["blue"], ms=7, label="direct sim")
ax[1].axvline(sc, color=REFERENCE, ls=":", lw=1)
ax[1].set(xlabel=r"$\sigma$", ylabel=r"surviving fraction $\varphi$",
          title="Survival: two-time tracks the sim", ylim=(0, 1.02)); ax[1].legend(fontsize=8)

# right: growth rate
g_stat = np.array([solve_fixed_point(0.5, float(s))["gstar"] for s in sg])
ax[2].plot(sg[sg < sc], g_stat[sg < sc], color=EMPIRICAL, lw=2, label=r"static FP $g^*$")
ax[2].plot(sg[sg >= sc], g_stat[sg >= sc], color=EMPIRICAL, lw=2, ls=":", label="static FP (unstable)")
ax[2].plot(sig_2t, d2["g_eff"], "s", color=PALETTE["navy"], ms=8, label="two-time DMFT")
ax[2].plot(ss, [sim[s][0] for s in ss], "o", color=PALETTE["blue"], ms=7, label="direct sim")
ax[2].axvline(sc, color=REFERENCE, ls=":", lw=1); ax[2].axhline(0, color=NEUTRAL, lw=0.6)
ax[2].set(xlabel=r"$\sigma$", ylabel=r"growth rate $g$",
          title="Growth: two-time tracks the sim"); ax[2].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()
```

Note on the `solve_fixed_point` sign: `dmft_twotime_fig.py` calls it with `-MU` where `MU=-0.5`, i.e. `solve_fixed_point(0.5, ...)`. Use `0.5` as above.

- [ ] **Step 5: Verify** — run the appendix cell. Expected: three-panel figure, autocorrelation for σ=1.0 stays near 1 and σ=2.5 decays toward 0.

- [ ] **Step 6: Commit**

```bash
cd ../relative-glv && git add scripts/compute.py data/dmft_twotime.npz story.ipynb && git commit -m "feat(notebook): two-time DMFT appendix figure (FC-Gaussian limit)"
```

---

### Task 10: Full-notebook execution + README sync

**Files:**
- Modify: `../relative-glv/story.ipynb` (executed output), `../relative-glv/README.md`

- [ ] **Step 1: Execute top-to-bottom:**

```bash
cd ../relative-glv && uv run jupyter nbconvert --to notebook --execute --inplace story.ipynb
```
Expected: completes with no error. If the two-time compute inside a cell is slow, confirm the appendix cell only *loads* `data/dmft_twotime.npz` (it should, per Task 9) rather than calling `solve_twotime`.

- [ ] **Step 2: Palette spot-check** — confirm no stray defaults remain:

```bash
cd ../relative-glv && uv run python -c "import json; s=json.dumps(json.load(open('story.ipynb'))); import re; bad=[h for h in ['#1f6f8b','#c1121f','#e07a18','#3b5b8c','#4878a8','#c0744f','#3a3a3a'] if h in s]; print('stray hexes:', bad or 'none')"
```
Expected: `stray hexes: none` (the recolored phase script is a `.py`, not in the notebook JSON; these are the old notebook literals).

- [ ] **Step 3: Update `README.md`** — in "Key Results"/notebook description, remove any β-vs-N / finite-economy mention if present, and add the D2-multiscaling and two-time-DMFT-appendix figures to the notebook's contents list. Add `solve_twotime` to the `__init__.py` API line and `data/dmft_twotime.npz` to the Repo Layout data list.

- [ ] **Step 4: Verify tests still pass:**

```bash
cd ../relative-glv && uv run pytest -q
```
Expected: all pass (including the new two-time test).

- [ ] **Step 5: Commit**

```bash
cd ../relative-glv && git add story.ipynb README.md && git commit -m "chore(notebook): execute end-to-end + README sync"
```

---

## Self-Review

**Spec coverage:**
- Palette single-source → Task 1. ✓
- Restyle every figure → Tasks 3,4,6 (live MSB, D-tests, DMFT), Task 7 (phase PNG). ✓
- D2 multiscaling extra → Task 5. ✓
- Two-time DMFT appendix (FC-Gaussian, labelled) → Tasks 8,9. ✓
- Drop β-vs-N / finite-economy → Task 2 (+ README in Task 10). ✓
- Live-light compute → cheap cells live; heavy via npz; two-time precomputed to npz. ✓
- Runs top-to-bottom → Task 10. ✓

**Placeholder scan:** No TBD/TODO; every code step shows the code or exact string swaps. Task 8 Step 1 says "copy the helpers" rather than pasting ~150 lines verbatim — the source file and required return keys are named exactly, which is the honest instruction here (the port is a mechanical copy of an existing, working solver).

**Type consistency:** `solve_twotime` return keys (`Ctau`, `dt`, `surv`, `g_eff`) are used identically in Task 8 test, Task 9 compute, and Task 9 plot. `PALETTE`/`MODEL`/`EMPIRICAL`/`REFERENCE`/`NEUTRAL`/`REGIME` defined in Task 1, consumed by name thereafter. Cell markers used instead of indices per the global constraint.

**Known risk:** two-time solver speed. Mitigated by precomputing to `data/dmft_twotime.npz` (Task 9) so the notebook only loads it; the test (Task 8) exercises just two σ values.
