# Langevin GLV at the critical point — MSB observables

**Date:** 2026-06-17

## Goal

Investigate a Langevin (multiplicative-noise) generalized Lotka–Volterra model sitting at the
cooperative critical point μ = μ_c, with interaction-matrix disorder σ = 0.2, and compute the
Moran–Secchi–Bouchaud (MSB) observables: the size–volatility relation and the growth-rate
distribution.

This replaces all prior Langevin GLV work (deleted), which sat deep in the competitive regime
(μ = −2, σ_mat = 1.0) and found the headline MSB signatures to be largely trivial. The new regime
is the opposite corner: low disorder, sitting exactly on the divergence edge.

## Key physics decision

At μ = μ_c the total abundance M = Σx diverges by definition (cooperative-runaway onset). Absolute
abundances therefore cannot reach a steady state — integrating them overflows. But the MSB
observables depend only on the *relative* size S_i = N·x_i / Σ_j x_j, which has a well-defined
stationary fluctuating state even while the total diverges.

So the model is integrated in the **relative / rescaled formulation** (Approach B): the simplex-
renormalized composition, which is bounded by construction and never overflows. This is exactly
what the repo's rescaled GLV formulation exists for — it factors out the M-divergence. MSB is
measured in rescaled-τ time; at σ = 0.2 this is benign and only sets the overall volatility level,
not the size-dependence exponent.

## Configuration

- N = 400.
- Power-law graph: ALPHA = 1.5, KMIN = 5 (⟹ ⟨k⟩ ≈ 11), KMAX = 120. Built inline via
  `nx.configuration_model` (same builder as prior work; kept inline, not abstracted).
- Interaction disorder σ_mat = 0.2.
- Weights W_ij = A_ij · (μ/C + (σ_mat/√C)·z_ij), z_ij ~ N(0,1), frozen per realization.
- Noise sweep s ∈ {0.05, 0.1, 0.2, 0.3} (Langevin multiplicative-noise strength, independent of σ).
- n_runs graph realizations pooled per s.
- EM step dτ, horizon τmax, burn-in fraction, sample stride — concrete values chosen in the plan;
  start from the prior notebook's scales (dτ ~ 0.02, burn ~ 0.4) and adjust.
- FLOOR for the live-size mask.

## Pipeline

1. **Build graph** (inline power-law configuration model), get binary adjacency A and mean degree C.
2. **Locate μ_c on the realized graph** via `find_mu_c_shape_scalar(A, C, sigma=0.2, mus, seed)`,
   with the μ-grid centered on `calculate_mu_c_regular(0.2)` and widened (the power-law graph is not
   regular, so the regular value is only a sweep center).
3. **Rebuild W at μ = μ_c** using the *same frozen disorder draw* (same `seed`, so the same z over
   edges) → the Langevin run sits exactly on the located critical point.
4. **Integrate (Approach B, EM in log-space, simplex-renormalized):**
   - State u = ln y, with y the relative composition (Σy = 1).
   - Drift from the rescaled ODE: `du = (F − φ − y + Σy²) dτ + s·√dτ·ξ`, ξ ~ N(0, I), F = W·y,
     φ = yᵀF.
   - Renormalize y = softmax(u) each step (keeps the composition on the simplex, bounded).
   - Record S = N·y at the sample stride after burn-in.
   - **Inline assertions:** Σy ≈ 1, all values finite, no overflow, throughout integration.
5. **Observables** (same definitions as MSB / prior work):
   - Per-species mean relative size S̄_i.
   - Growth-rate volatility σ_i = √(π/2) · mean|g − ḡ|, with g_i = Δ ln S_i.
   - Pooled growth-rate distribution g.
6. **Cache** pooled results to `.npz`.

## Outputs

- `notebooks/langevin_glv_critical_msb.ipynb` — self-contained notebook.
- `notebooks/langevin_glv_critical_results.npz` — cached pooled observables.
- `notebooks/langevin_glv_critical_volatility_size.png` — size–volatility loglog, one V-curve per s,
  with two-branch power-law slopes annotated.
- `notebooks/langevin_glv_critical_growth_dist.png` — pooled growth distribution at a representative
  s, vs same-σ Gaussian and same-scale Laplace.

## Out of scope

- No frozen-field (decoupling) adversarial null this round — observables only.
- No separate pytest file — inline assertions only.
- No absolute-x cross-check (Approach A).

## Deletions (prior Langevin GLV work)

- `notebooks/langevin_glv_msb.ipynb`
- `notebooks/langevin_glv_results.npz`
- `notebooks/langevin_glv_growth_dist.png`
- `notebooks/langevin_glv_volatility_size.png`
- Memory file `project_langevin_glv_msb.md` and its `MEMORY.md` index line; fix any back-links
  (`[[project_langevin_glv_msb]]`) in other memory files.
