# MSB observables in the slowly-diverging regime at the MA → Unbounded boundary — design

**Date:** 2026-06-20
**Status:** Draft (pending user review)
**Deliverable:** new self-contained notebook `notebooks/slow_divergence_msb.ipynb` + figures `slow_divergence_*.png`

## Goal

Measure the two MSB firm-growth stylized facts — the size–volatility exponent β and the pooled
growth-rate distribution (skew, tail) — in the **slowly-diverging regime** that sits just above the
boundary between Roy's chaotic Multiple-Attractors (MA) phase and the Unbounded-Growth (UB) phase.
Deliver a verdict: do clean, empirically-shaped MSB observables (β ≈ 0.2, symmetric fat-tailed growth
tent) emerge in this regime, or is the regime a condensation transient that does not support a
well-defined MSB law?

## Regime and why it is the interesting one

The base model is Roy's disordered GLV at the canonical chaotic-MA spec, already implemented in
`notebooks/chaotic_glv.ipynb`:

```
ẋ_i = x_i (1 − x_i − (α x)_i) + λ        (Roy sign convention: positive α mean = competition)
```

with (μ, σ, γ, λ) = (4, 2, 0, ~10⁻¹⁰..0) on a power-law configuration-model graph (degree exponent
ALPHA = 2.5, mean degree ⟨k⟩ ≈ 100), per-edge mean μ/C and variance σ²/C. Per realization there is an
own critical disorder σ_c (≈ 2–2.5 here) above which total biomass M = Σx_i diverges in finite time.

"Slowly diverging" means σ just **above** σ_c, where the blow-up time t_div is large (critical slowing:
t_div → ∞ as σ → σ_c⁺). Inside a long pre-blowup window the system is a **growing chaotic state**:
M(t) climbs roughly exponentially (a GDP-like aggregate trend) while the cross-sectional shares
S_i = N x_i / Σ_j x_j keep turning over chaotically.

This regime is the natural MSB analog, and is distinct from the bounded f < 1 side already studied in
`chaotic_glv.ipynb` / `criticality_msb_findings.ipynb`:

- Measuring in **shares** subtracts the common growth automatically:
  g_i = Δln S_i = Δln x_i − Δln(M/N). The Δln(M/N) term is the market/GDP component; shares isolate
  the **idiosyncratic** firm growth that MSB actually measures.
- The bounded f < 1 side has a ~stationary M (no aggregate growth) and there β floors at ≈ 0.3–0.5,
  never the empirical 0.15–0.20 (`project_chaotic_glv_dt_sampling`). The slowly-diverging side has
  genuine aggregate growth **plus** idiosyncratic turnover — physically closer to a growing economy.

**The known risk this design must police:** approaching σ_c, extinction *increases* — a winner-take-all
**condensation**, not coexistence (`project_chaotic_glv_dt_sampling`, finding E). If condensation
completes inside the measurement window the shares degenerate (one S_i → N, the rest → 0) and β / the
growth tent become artifacts. So the make-or-break step is verifying a **quasi-stationary growing
window** exists before measuring anything.

## Approach

### 1. Dataset generation (compute once, store raw, slice many ways)

Per realization: build a fresh power-law graph + frozen disorder + IC (reuse the `build_realization`
construction from `chaotic_glv.ipynb` cell 31, copied inline per the project's no-library-abstraction
preference), `locate_sigma_c` via bisection with the **per-capita** divergence threshold (a fixed total
threshold false-fires at large S since bounded biomass ~ S), then integrate at σ = f·σ_c for a band
straddling the boundary:

- **bounded reference:** f ∈ {0.90, 0.97}
- **diverging:** f ∈ {1.02, 1.05, 1.10, 1.20, 1.50}

f → 1⁺ gives long t_div, f ≫ 1 gives fast blow-up, so the f-band *is* the slowness controller — no
separate bisection-on-t_div needed. Integration: `solve_ivp` RK45, `rtol=1e-8`, `atol=1e-10`, on a
fixed output grid `t_eval = linspace(0, tmax, n_eval)`; a diverging run keeps the points before
blow-up (`r.success == False`, t_div = `r.t[-1]`).

Store per run (`.npz`, float32 for the bulk arrays): `seed, sigma_c, f, sigma, C_eff, status, t_div`,
the aggregate `M(t)` on the fixed grid, and **per-firm log-size on a fixed Δt = 1 grid truncated at
t_div**. That per-firm log-size array is the raw-enough payload to recompute MSB in any sub-window at
any Δt ≥ 1 later, without re-integrating.

Scale: S = 2000, ~150 realizations × the 7 f-values. σ_c and the MSB observables are expected
~S-independent (mean-field property); spot-check on a handful of S = 10000 runs. Prototype at
S = 1000, ~20 realizations to validate the pipeline and the self-similarity diagnostic before scaling.

### 2. Slow-diverger selection (post-hoc)

Among diverging runs, keep those with t_div ≥ T_min (default 150) **and** for which a quasi-stationary
window of ≥ W = 50 snapshots exists (see §4). The slowness axis is t_div (equivalently f). Fast
divergers and bounded runs are retained in the dataset for contrast but excluded from the headline
slow-diverger pool.

### 3. MSB measurement

In the window [t_burn, ρ·t_div] with defaults t_burn = 0.2·t_div (skip IC relaxation) and ρ = 0.8
(skip the terminal condensation spike), on a **fixed Δt = 1** grid common to all runs, via the existing
`_msb_from_grid` kernel (Moran cross-sectional size S_i = N x_i/Σx, MAD volatility
σ_i = √(π/2)·mean|g_i − ⟨g_i⟩|, binned-median log-log β). Report, pooled across the slow-diverger
ensemble at the common Δt:

- **β** (size–volatility), computed on **all firms** *and* on **persistent firms** — the two diverge
  near σ_c (`criticality_msb_findings`, finding F: persistent-firm conditioning ≈ empirical
  survivorship recovers the symmetric tent; all-firms β is crash/floor-sensitive).
- **growth distribution** g = Δln S: skew, excess kurtosis, **and** the Δt-robust 99.9pct/MAD quantile
  tail (kurtosis is not Δt-robust, the quantile metric is — `criticality_msb_findings`, finding B).

### 4. The make-or-break diagnostic — is it a self-similar growing state?

Before any β is trusted, for each slow-diverger:

1. **Exponential aggregate growth.** Fit ln M(t) vs t in the window; confirm ~linear and report the
   growth rate ⟨d ln M/dt⟩. A self-similar growing state has steady exponential M; curvature signals an
   accelerating run-up toward finite-time blow-up (not stationary).
2. **Stationary cross-section.** Split the window early/late; check the cross-sectional share
   distribution (e.g. its log-share CDF or top-share), β, and the growth moments do **not** drift
   between halves.

If (1) and (2) hold, the slow-diverger MSB is a genuine phase property. If they fail (condensation
completes inside the window), the MSB there is a transient. **Per the user decision: a failure is the
finding** — document "the slow-diverger MSB is a condensation transient, not a phase property" and
stop; do not force a regime that is not there.

### 5. Comparisons

Put the slow-diverger β and growth distribution next to:

- the bounded f < 1 side from this same dataset (expected β ≈ 0.3–0.5, left-skewed all-firms growth);
- the empirical MSB anchors (β ≈ 0.15–0.20, symmetric fat-tailed tent).

## What is stored / notebook structure

Self-contained `notebooks/slow_divergence_msb.ipynb`:

1. params + imports
2. inline helpers copied from `chaotic_glv.ipynb`: `build_realization`, `locate_sigma_c` /
   `_diverged_at` (per-capita threshold), `_msb_from_grid`, plus a new `run_and_store` that integrates
   one (realization, f) and returns the stored payload
3. dataset-generation ensemble cell (`joblib.Parallel`), writing `slow_divergence_dataset.npz`
4. slow-diverger selection + t_div / f distribution diagnostics
5. self-similarity diagnostic cell (the make-or-break check) + figure
6. MSB measurement cell (β all/persistent, growth distribution) + figures
7. comparison cell (slow-diverger vs bounded side vs empirical) + figure
8. markdown verdict

## Constraints and caveats

- **Δt robustness.** Use a single fixed Δt = 1 across all pooled runs; never the adaptive grid (the
  adaptive grid manufactures fat tails by horizon-mixing — `project_chaotic_glv_dt_sampling`). Δt = 1
  must stay above the trajectory-resolution floor (`tmax/(n_eval−1)`); set n_eval so the fixed grid is
  not interpolated finer than the integrator output.
- **Divergence detection / σ_c locator.** Per-capita (mean-abundance) threshold, not total biomass.
- **Condensation.** The central risk; the §4 diagnostic exists precisely to catch it. The persistent-
  firm β is the floor-robust read; the all-firms β near σ_c is crash-sensitive.
- **S-independence.** Assumed for σ_c and MSB; verified by spot-check at S = 10000, not taken on faith.
- **Immigration.** Default λ small/0. λ is **out of scope** for the headline study (memory D: λ
  narrows the distribution, does not symmetrize). If the self-similarity check fails it is *not* the
  first fallback — the decision is to report the negative finding.

## Success criterion

A quantified verdict with figures: (a) the self-similarity diagnostic outcome (does a quasi-stationary
growing window exist, and over what t_div / f range), and (b) if it does, the slow-diverger β (all and
persistent) and growth-distribution skew + quantile tail at a common fixed Δt, compared against the
bounded f < 1 side and the empirical MSB anchors — with a one-paragraph statement of whether the
slowly-diverging boundary regime yields a cleaner MSB law than the bounded side.

## Out of scope

- Immigration λ sweep (deferred; not the first fallback on failure).
- (μ, σ) phase-diagram mapping, Lyapunov-exponent chaos quantification, runaway-cluster mechanism
  (these were the other AskUserQuestion branches the user did not pick).
- Any change to `glv/` library code or to `chaotic_glv.ipynb` (this study is additive and isolated).
