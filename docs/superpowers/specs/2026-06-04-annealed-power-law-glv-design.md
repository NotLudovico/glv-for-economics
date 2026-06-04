# Annealed power-law graph → GLV dynamics — design

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Deliverable:** `notebooks/annealed_power_law_glv.ipynb`

## Goal

A notebook that builds the **annealed adjacency matrix** of a power-law graph from a
continuous degree (hidden-variable) sequence, forms the GLV interaction matrix
$W = A\cdot\alpha$, and runs the rescaled GLV dynamics to produce three observables:
rescaled trajectories, volatility-vs-size scaling, and the growth-rate distribution.

It is the **degree-based, Chung–Lu-normalized** counterpart of the existing
`notebooks/power_graph.ipynb`, which uses a fitness parameterization with an arbitrary
$\beta=1$ that makes the graph nearly complete (mean degree $\approx 745/1000$). Here a
single mean-degree constant $C=\bar k$ keeps the graph genuinely sparse
(mean degree $\approx\bar k$) and self-consistent.

## Construction (single realization)

All quantities below are built **inline in the notebook** (project convention: do not
wrap degree-sequence construction in the library). Continuous hidden degrees — no
rounding to integers, no configuration-model graph; the annealed matrix is built
directly.

1. **Degrees** — power law $P(k)\propto k^{-(1+\alpha)}$ for $k\ge k_{\min}$, by inverse-CDF:
   $$k_i = k_{\min}\,U_i^{-1/\alpha},\qquad U_i\sim\text{Unif}(0,1),\quad i=1,\dots,N.$$

2. **Mean-degree constant** — the single $C$ used everywhere, recomputed per realization:
   $$C=\bar k=\frac1N\sum_{i} k_i.$$

3. **Connection probability** — soft (Fermi) form, applied to **every** entry including
   the diagonal:
   $$p_{ij}=\frac{k_ik_j}{k_ik_j+NC},\qquad p_{ii}=\frac{k_i^2}{k_i^2+NC}.$$
   Since $NC=\sum_l k_l$, this is the Chung–Lu normalization: $k_i$ is the expected
   degree of node $i$, and the realized mean degree of $A$ is $\approx C$. The condition
   $k_ik_j\le NC$ holds for typical pairs; rare hub–hub (and hub self-) pairs saturate
   softly with $p<1$ (the denominator guarantees $p\in[0,1)$ always), so no degree cutoff
   and no diagonal removal.

4. **Annealed adjacency** — Gaussian matching a Bernoulli$(p_{ij})$ in mean and variance,
   symmetric, **diagonal kept**:
   $$A_{ij}=p_{ij}+\sqrt{p_{ij}(1-p_{ij})}\,B_{ij},\qquad B_{ij}=B_{ji}\sim\mathcal N(0,1).$$
   ($\mathbb E[A_{ij}]=p_{ij}$, $\operatorname{Var}[A_{ij}]=p_{ij}(1-p_{ij})$, for all $i,j$
   including $i=j$.) The symmetric noise $B$ must have **unit-variance marginals on every
   entry including the diagonal** — note $(G+G^\top)/\sqrt2$ leaves the diagonal at
   variance 2, so the diagonal is overwritten with independent unit normals. This is the
   one place the inline `sym_noise` differs from `power_graph.ipynb` (which zeroed the
   diagonal).

5. **Interaction weights** — same $C$ as the mean-field scale, independent symmetric
   Gaussian noise $B^{(\alpha)}$ (also unit-variance diagonal), **diagonal kept**:
   $$\alpha_{ij}=\frac{\mu}{C}+\frac{\sigma}{\sqrt C}\,B^{(\alpha)}_{ij},\qquad
     W_{ij}=A_{ij}\,\alpha_{ij},\quad W_{ii}=A_{ii}\alpha_{ii}.$$
   $W$ is dense and symmetric ($N=1000\Rightarrow$ fine). **Consequence for the dynamics:**
   the rescaled RHS computes $F=Wy$ with a *separate* $-y_i$ self-term, so a nonzero
   $W_{ii}$ shifts node $i$'s effective self-regulation from $-y_i$ to $(W_{ii}-1)y_i$ —
   matrix self-coupling on top of the baseline logistic term. With $\mu=0$ this is small
   ($|W_{ii}|\lesssim\sigma/\sqrt C$) and concentrated on the few saturated hubs, so it
   stays self-regulating (no blow-up). This departs from the usual GLV convention
   $W_{ii}=0$, per the instruction to keep the diagonal.

## Dynamics and observables

Reuse the library: `glv.rescaled_glv_sparse` (RHS) and `glv.sweep_observables`
(parallel multi-run). Integration: `solve_ivp(..., method="LSODA", max_step=1)`,
state $(y_1,\dots,y_N,M,t)$ with $y_i=x_i/M$, $M=\sum_i x_i$ — identical to
`power_graph.ipynb`.

- **Rescaled trajectories** — single run: $y_i(\tau)$ and unscaled $x_i(t)$ for ~100
  sampled nodes, max-degree hub highlighted.
- **Volatility-vs-size scaling** — yearly resample on real time $t$; cross-section
  $S_{iy}=N x_{iy}/\sum_j x_{jy}$; growth $g_{iy}=\ln(S_{i,y+1}/S_{iy})$; per-species
  $\sigma_i=\sqrt{\pi/2}\,\overline{|g_i-\bar g_i|}$ vs $\langle S_i\rangle$; log–log with
  equal-count bins and power-law slope $\beta$. Single run and multi-run pooled.
- **Growth-rate distribution** — pooled histogram of $g$ on a log-density axis.

## Notebook layout

| # | Section | Content |
|---|---------|---------|
| 0 | Setup & parameters | imports, `glv.apply_style()`, parameter block, `default_rng` |
| 1 | Construction | sample $k$, $C=\bar k$, $p_{ij}$, $A$; **summary prints only** (no diagnostics figure — that track was not selected; one small sanity figure is easy to add later) |
| 2 | Interaction matrix | build $W=A\,\alpha$, print stats |
| 3 | Single rescaled-GLV run | `solve_ivp` + status/survivor prints |
| 4 | Trajectories | $y_i(\tau)$, $x_i(t)$, hub in accent color |
| 5 | Single-run observables | volatility-vs-size (binned + slope) **and** growth-rate distribution |
| 6 | Multi-run pooled | `glv.sweep_observables` over $n_\text{runs}$ fresh realizations → pooled volatility-vs-size + growth-rate distribution |

Inline helpers (adapted from `power_graph.ipynb`): `sym_noise` (symmetric standard normal,
unit-variance diagonal — **not** zeroed), a degree-based `build_power_graph`
(returns `k, P, A, C`, diagonal retained throughout), `build_W`, `loglog_binfit`.

## Default parameters (top-of-notebook, all adjustable)

| Symbol | Value | Note |
|--------|-------|------|
| $N$ | 1000 | nodes |
| $\alpha$ | 1.5 | tail exponent $\tau=1+\alpha=2.5$ (canonical scale-free) |
| $k_{\min}$ | 1 | degree lower cutoff |
| $\mu$ | 0 | marginal case (as in `power_graph.ipynb`) |
| $\sigma$ | 1 | disorder |
| $\tau_{\max}$ | $10^3$ single / $10^4$ pooled | rescaled-time horizon |
| $n_\text{runs}$ | 50 | pooled realizations |
| $n_\text{years}$ | 100 | yearly resample grid |

## Conventions

- **Plots:** `glv.apply_style()` (editorial palette); all labels/titles/legends in LaTeX
  mathtext via raw strings ($\mu$, $\sigma$, $\langle S_i\rangle$, $\sigma_i$, $\beta$,
  $\tau$, …). Volatility/growth plots may enable a faint log grid for readability.
- **Reproducibility:** seeded `np.random.default_rng`; per-run seeds in the pooled sweep.
- **No new library code.** Construction stays inline; the existing
  `glv.generate_annealed_matrix` is a *different* model ($A_{ij}=k_ik_j/(NC)\cdot$weight)
  and is left untouched.

## Out of scope (not selected)

Eigenvalue spectrum of $A$; construction-diagnostics figures (degree-distribution fit,
$p_{ij}$ heatmap, $A$-entry histogram); annealed-vs-quenched comparison; fixed-point /
survivor analysis.
