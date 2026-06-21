# σ_c / multiple-attractor investigation — findings

**Date:** 2026-06-17
**Status:** Consolidated, paused (no further probing pending a decision)
**Origin:** Started as "fix μ, sweep σ, find the critical regime, then measure volatility-vs-size and
the growth-rate distribution at the critical σ" — the σ-analog of the μ_c workflow.

## What was built

`notebooks/sigma_critical.ipynb` (uncommitted): at fixed μ=0.2 on the **rescaled** model, locates a
per-realization critical σ_c via the shape-scalar zero-crossing (σ-analog of `find_mu_c_shape_scalar`),
reproducible (graph wiring seeded), then measures volatility-vs-size and the growth-rate distribution at
each realization's own σ_c on the physical-time MSB clock.
- σ_c ≈ 0.61 (median 0.633, range 0.52–0.84 over realizations).
- The size–volatility **V survives** (β ≈ −0.8 / +0.4).
- The growth distribution is **one-sided** (decay tail only, strongly left-skewed).

## What the investigation revealed (the important part)

1. **That σ_c is the unbounded-growth onset, not the multiple-attractor (MA) transition.** The
   shape-scalar c=0 marks where dM/dτ=1+cM changes sign, i.e. where total mass M diverges — a
   cooperative-side transition, distinct from the MA transition.

2. **The one-sided growth is a relaxation transient, not an overshoot.** Backing off σ subcritically did
   not add a right tail; it only lengthened the physical window. Firms relax from a dispersed IC onto a
   fixed point (few survive at g≈0, many decay) — consistent with the existing "the V is a transient"
   result (`volatility_v_cause.ipynb`).

3. **The rescaled model misrepresents the competitive/multistable regime.** At μ=−2, σ=1.6 the rescaled
   `rescaled_glv_sparse` collapses to ~4 survivors (M hits the cap) where the **original** model
   `dx=x(1−x+Ax)` keeps ~67% coexistence. Use the original model for anything competitive/multistable.
   (Also: RK45 + max_step=1e2 silently fails at high σ — use LSODA + a divergence event.)

4. **A bounded multiple-attractor phase exists, but only as a thin sliver.** On the original model at
   competitive μ, sweeping σ gives: unique-FP coexistence → a narrow bounded MA band (different ICs →
   different communities, cross-IC overlap dips to ~0.90) → unbounded growth. Making μ more competitive
   (−2 → −5) just shifts the sliver to higher σ; it never widens. The heavy-tailed exponential degree law
   lets cooperative runaway clusters diverge before a broad MA phase can open.

5. **The MA states are STATIC fixed points, and γ<0 does not change that.** Late-window Var(ln x) ≈ 1e−29
   (machine zero) throughout the MA band — no persistent fluctuations. A direct γ scan
   (correlated A_ij,A_ji, γ ∈ [−0.9, +0.8]) showed γ<0 still settles to static FPs and γ>0 diverges early.
   So anti-symmetry alone does not open a fluctuating phase here.

## Conclusion

In this **sparse, exponential-degree (C=5), finite-N** GLV, **unbounded divergence is the dominant
instability** — it preempts both a wide multiple-attractor phase and a persistent-fluctuation phase at
every (μ, σ, γ) scanned. A two-tailed MSB growth-rate distribution (which needs persistent fluctuations)
is therefore **not reachable by tuning interaction statistics** on this graph. The untested lever is
interaction **structure**: denser or regular connectivity (toward the mean-field limit where the MA and
fluctuating phases are established in Bunin/Galla) and/or larger N.

## Open next steps (not started)

- Probe denser (large C) / regular-degree graphs at fixed competitive μ, sweeping σ, for a bounded
  fluctuating or wide-MA phase.
- Re-test persistent fluctuations at larger N (e.g. N=2000) and γ near −1, in case the fluctuating phase
  is finite-size-suppressed at N=300.
- If pursuing γ: add a symmetry parameter to `glv.generate_matrix` (currently entries are independent ⇒
  γ=0) with a test.

## FINAL UPDATE — the fluctuating phase / two-tailed growth is not robustly reachable here

After mapping the multiple-attractor sliver, the search continued for a *persistent-fluctuation* phase
(the regime that gives a genuinely two-tailed growth distribution, vs the static MA fixed points):

- A **dense power-law graph** (⟨k⟩≈12) *did* show a bounded fluctuating window (late-window Var≈1e−2) with a
  two-tailed growth distribution (skew≈−1.4) at N=400, σ≈2.0 — the first positive signal. Density, not
  the tail exponent, was the apparent lever.
- But **stationarity testing** (Var of ln x across successive late windows) revealed this is delicate: a
  single late-window variance conflates slow relaxation with genuine fluctuation. The genuinely-fluctuating
  states (flat Var across windows) exist only in a **narrow, realization-dependent sliver pinned to each
  graph's divergence onset**.
- Attempts to robustify failed: **density** (⟨k⟩ 11→23) did not widen it; **immigration** `+λ` (1e−5,1e−4),
  the literature regularizer, did not either. Across μ∈[0.2,−5], σ, γ∈[−0.9,0.8], exp/power-law graphs,
  ⟨k⟩∈{2…23}, N∈{300,600}, λ∈{0,1e−5,1e−4}, the fluctuating phase is always a fragile sliver where the
  σ-disorder's cooperative runaway clusters diverge before a wide bounded fluctuating phase can open.
- **Whole-window** growth is transient-swamped (one-sided) even where fluctuations are real; only a
  post-transient window shows the two tails, and only in the sliver.

**Conclusion:** a robust two-tailed MSB growth law is not obtainable in this GLV variant (logistic
self-limitation + sign-indefinite Gaussian interactions on sparse/moderate finite graphs). Reaching it
needs a more fundamental change: a true mean-field limit (dense, large N, σ/√N scaling) or
bounded/structured interactions that forbid cooperative runaway (competitive-only α<0, or capped positive
entries). The project's solid positive results remain the rescaled-model relaxation-transient findings.

## Status of artifacts

- `notebooks/sigma_critical.ipynb` — valid study of the **unbounded-growth onset** at μ=0.2 (rescaled
  model). Uncommitted. Keep/commit only if that transition is wanted; it is NOT the MA/fat-tail study.
- Design spec: `docs/superpowers/specs/2026-06-17-sigma-critical-design.md`.
- Findings persisted in memory: `project_sigma_critical`, `project_ma_phase_needs_gamma`.
