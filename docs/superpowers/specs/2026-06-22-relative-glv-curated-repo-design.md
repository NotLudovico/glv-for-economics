# Curated repository for the relative GLV model

**Date:** 2026-06-22
**Status:** approved, ready for build plan

## Purpose

A standalone, shareable repository that explains the **relative GLV** model to
scientifically-interested readers. Curated, not exploratory: no dead-end
scripts, no scratch plots, only the code and figures that carry the story.

The new model lives today in `glv/growing_glv/` mixed with exploration (much of
it already archived to `growing_glv/_attic/`). This repo extracts the clean
core and presents it as a single narrative.

## The model (what we are explaining)

The relative (scale-invariant) GLV:

$$\dot x_i = x_i\Big[\,1 - \frac{x_i}{m} - \frac{(\alpha x)_i}{m}\,\Big],\qquad m=\langle x\rangle = M/N.$$

Interactions act on the **mean** firm, so the competitive/cooperative term is
$O(M)$ not $O(M^2)$: the aggregate $M$ grows exponentially in physical time with
no finite-time blow-up. Integrated in a **share + log-scale split** — shares
$w_i=x_i/M$ on the simplex, $\ln M$ separate — so nothing overflows and physical
time is unbounded:

$$f_i = 1 - N w_i - N(\alpha w)_i,\quad \langle f\rangle=\sum_i w_i f_i = g_{\mathrm{eff}},$$
$$\dot w_i = w_i\big(f_i - \langle f\rangle\big)\ \text{(replicator on the simplex)},\qquad \frac{d\ln M}{dt}=g_{\mathrm{eff}}.$$

This model simultaneously produces: exponential growth, coexistence, and a
**symmetric fat-tailed (tent-shaped) growth-rate distribution** with a
size–variance exponent $\beta$ in the empirical Moran–Santos–Bouchaud (MSB)
band $0.15$–$0.20$. It is a disordered replicator; its DMFT is derived in
`docs/superpowers/specs/2026-06-21-relative-glv-dmft-derivation.md` and gives a
$\mu$-independent chaos onset at $\sigma_c=\sqrt 2$.

## Decisions (from brainstorming)

- **Form:** narrative notebook (`story.ipynb`) backed by a small clean Python module.
- **Location:** `~/Code/relative-glv`, sibling to `glv/`, its own git repo.
- **Theory depth:** results-first. Lead with model + MSB empirics + phase
  diagram; DMFT compressed to a brief appendix (stated + validated, full
  derivation linked, not re-derived).
- **License:** MIT.
- **Documentation contract:** every science-bearing function carries its
  mathematics (governing equation / estimator / rationale) in both its
  docstring and the notebook prose. Pure plumbing (plotting, npz I/O, CLI) stays
  comment-light.

## Layout

```
relative-glv/
  relative_glv/
    __init__.py        # public exports
    model.py           # coupling() + integrate() — the model
    msb.py             # rescale(), size_volatility(), tent_stats() — the empirics
    dmft.py            # solve_fixed_point(), sigma_c() — the theory
  story.ipynb          # the explainer (results-first)
  scripts/
    compute.py         # regenerate the heavy big-N figure data (minutes)
  data/                # committed precomputed npz (compute-once, replot-free)
  figures/             # committed publication PNGs the notebook embeds
  tests/
    test_dmft.py       # DMFT-vs-simulation validation as real pytest
  README.md
  pyproject.toml
  LICENSE
  .gitignore
```

## Module design

One clean implementation of each piece of physics. Removes the triplicated RHS
(`explore.integrate`, `ch4_data.simulate`, `dmft_validate.simulate`) and the
copy-pasted MAD-rescaling.

### `relative_glv/model.py`

**`coupling(N, mu, sigma, *, kind="fc", gamma=0.0, seed=0)`**
Builds the interaction matrix $\alpha$.
- `kind="fc"`: fully-connected Gaussian, $\alpha_{ij}=\mu/N+(\sigma/\sqrt N)z_{ij}$,
  zero diagonal. The clean ensemble the DMFT is derived for.
- `kind="powerlaw"`: power-law configuration-model graph (degree exponent 2.5,
  mean degree ~100), Roy disorder per edge with mean $\mu/C$, variance
  $\sigma^2/C$, symmetric/antisymmetric split by $\gamma$.

*Math in docstring:* the disorder scaling (mean $\mu/C$, std $\sigma/\sqrt C$ —
the $1/C$ / $1/\sqrt C$ that keeps the mean field finite as $C$ grows) and the
$\gamma$ correlation construction $\sqrt{1+\gamma}\,S\pm\sqrt{1-\gamma}\,V$.

**`integrate(alpha, *, tmax, n_eval, lam=0.0, seed=0, method="LSODA")`**
Integrates the relative GLV in the share+lnM split. Returns a small result
object/dict: `t`, `W` (shares, $N\times n$, renormalised), `lnM`, `success`.
The single RHS. Optional `lam` is the persistent share floor
$\lambda(1/N - w_i)$ that fights condensation (mass-conserving on the simplex).

*Math in docstring:* the split equations above and **why it cannot overflow**
($w$ stays on the simplex, $\ln M$ grows linearly, $x=Mw$ is never formed).

Convenience: `growth_rate(t, lnM, window)`, `survivors(W, window, floor=1e-6)`.

**Sign convention:** competition-positive — $\mu>0$ means competition (matches
`coupling` and the model RHS). Documented once in `model.py` and the README.
(The thesis "chapter $\mu$" flips the sign for presentation; we do not.)

### `relative_glv/msb.py`

**`rescale(g)`** — centre and divide by $\sqrt{\pi/2}\,\mathrm{MAD}$, **not** std.
*Math in docstring:* std is inflated by the fat tails, so it mis-normalises the
tent; $\sqrt{\pi/2}\,\mathrm{MAD}$ equals the std for a Gaussian but is robust to
the tails. This is the crux of reading the MSB distribution correctly.

**`size_volatility(W, t, *, window, dt)`** — relative size $S_i=N w_i$, growth
$g_i=\Delta\ln S_i$ on a fixed $\Delta t$ grid; returns mean size $\bar S_i$,
MAD-volatility, binned-median curve, and the decline-branch exponent $\beta$.
*Math in docstring:* the size–variance relation $\sigma(\bar S)\sim\bar S^{-\beta}$
and the binned-median log–log fit on the large-$S$ decline branch (past the
floor plateau).

**`tent_stats(g)`** — Bowley skewness, excess kurtosis of the rescaled growth.

### `relative_glv/dmft.py`

Lifted ~verbatim from the validated `dmft_solver.py`.
**`solve_fixed_point(mu, sigma, gamma=0.0)`**, **`sigma_c(gamma=0.0)`**.
*Math in docstring:* the relaxed-phase self-consistency for $(\Delta,q,\chi,g^*)$,
and that $\sigma_c=\sqrt 2$ is $\mu$-independent because $\mu$ is a uniform shift
of every fitness and cancels in the replicator's relative dynamics ($g^*=g_0-\mu$).

## `story.ipynb` — results-first

1. **The phenomenon.** MSB stylized facts being explained: the tent-shaped
   growth-rate PDF (fatter than Gaussian, symmetric) and the size–variance
   exponent $\beta\approx0.15$–$0.20$. State them as the target.
2. **The model.** Equations + one live simulation (`coupling` + `integrate`,
   small $N$): $\log_{10}M(t)$ rising while relative sizes $S_i$ churn.
3. **MSB in the model.** `size_volatility` + `tent_stats`: the symmetric
   fat-tailed tent vs Laplace/Gaussian, and $\beta$ in the empirical band.
   Show the committed big-$N$ figures (size–volatility, tent, $\beta$-vs-$N$,
   $\beta_\infty$ extrapolation toward ~0.2 as $N\to\infty$).
4. **Phase diagram.** Where growing + chaotic + coexisting co-occur (committed
   figure; light recompute optional).
5. **Appendix: DMFT in brief.** Single-site = disordered replicator; run
   `solve_fixed_point`/`sigma_c`; show the validation figure (growth rate,
   survival, chaos onset at $\sigma_c=\sqrt2$) from `tests`/`scripts` output.

Live cells use small $N$ (~800–1600, ~1–2 min total). Heavy big-$N$ results are
embedded as committed PNGs and reproduced by `scripts/compute.py`.

## Compute split

- **Light:** notebook live demos, small $N$, run on open.
- **Heavy:** `scripts/compute.py` regenerates the big-$N$ datasets ($N$ up to
  ~12800, 8 seeds, parallel via joblib) into `data/*.npz`; a plotting pass turns
  those into `figures/*.png`. Source: the locked operating point and procedures
  in `ch4_data.py` / `ch4_plots.py`, cleaned. Dataset-oriented: compute once,
  replot freely. Committed `data/` + `figures/` mean no recompute is needed to
  read the repo.

## Tests

`tests/test_dmft.py` — the `dmft_validate.py` assertions as real pytest against a
matched fully-connected simulation: relaxed-phase $g_{\mathrm{eff}}\approx g^*$
and survival $\approx\varphi$; chaos onset near $\sigma_c=\sqrt2$;
$\mu$-independence (shape fixed, $g_{\mathrm{eff}}$ shifts by $-\Delta\mu$). Plus
the `dmft.py` solver self-check ($\sigma_c(0)=\sqrt2$). Uses small $N$/few seeds
so it runs in CI-reasonable time.

## Provenance map (source → curated)

| Curated | Source in `glv/growing_glv/` |
|---|---|
| `model.coupling` (fc) | `dmft_validate.fc_alpha` |
| `model.coupling` (powerlaw) | `explore.build_alpha` |
| `model.integrate` | the shared RHS in `explore.integrate` / `ch4_data.simulate` |
| `msb.rescale` | the repeated `sqrt(pi/2)*MAD` blocks |
| `msb.size_volatility` | `measure_msb` / `ch4_data._decline_beta` |
| `dmft.*` | `dmft_solver.py` (~verbatim) |
| heavy figures | `ch4_data.py` + `ch4_plots.py` |
| phase diagram | `phase_diagram.py` / `phase_space_compute.py` |
| `tests/test_dmft.py` | `dmft_validate.py` assertions |

## Out of scope

`_attic/`, the original sparse non-relative GLV (`glv/` package), `criticality_msb/`,
all exploratory notebooks, the thesis chapters. This repo is only the relative-GLV
story. The thesis remains the authoritative long-form derivation; the new
README links back to it.

## Open question for build

- Whether to add a one-paragraph README link to the thesis PDF / DMFT derivation
  doc, or copy the DMFT derivation doc into the repo. Default: link, do not copy
  (single source of truth stays in `glv/`).
