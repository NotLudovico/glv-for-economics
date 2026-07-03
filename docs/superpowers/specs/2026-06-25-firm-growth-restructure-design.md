# Firm-growth chapter restructure + DMFT appendix completion

Date: 2026-06-25. Approved design (brainstorming). Scope: thesis prose only (`thesis/chapters/`),
plus the two figures already generated in this session.

## Decision

Drop the **mean-degree** model from the thesis. Its headline claim — beta approx 0.15 at the
empirical economy size, via a beta(N) decline read at the empirical N — is the least defensible
(a finite-economy coincidence, no plateau). The spine becomes the **own-degree** model plus the
**DMFT**, and the beta claim becomes **qualitative**.

## Settled beta story (the framing everything serves)

- The model reproduces the MSB phenomenology qualitatively: symmetric fat-tailed tent (D1),
  multiscaling (D2), size-volatility decline beta>0 (D3).
- beta is **not a universal constant**: it is a crossover set by the measurement horizon h
  relative to the firm memory time tau. Steep at short h, flat at long h.
- The model's distinguishing success is a genuine **multi-year memory / persistence**, exactly
  the horizon signature in the data (Moran-Bouchaud: growth distributions Gaussianize with
  horizon yet keep fat tails). An iid granular model cannot produce this.
- The **magnitude** of beta cannot be sharply calibrated from available data: the deciding
  quantity (firm-memory timescale, defined consistently) is not in the sources, and is both
  convention-dependent (1/e vs integrated time) and fat-tail-dependent. So: right physics,
  magnitude not pinned. Do NOT headline "reproduces 0.2" nor a hard "2x too steep".

## Firm-growth chapter (`firm_growth.tex`) — new arc

1. Phenomenology in the fluctuating state: own-degree model gives the D1/D2/D3 stylized facts.
   Keep figures `growing_conditional`, `growing_multiscaling`.
2. beta is a crossover, not a constant: horizon h vs memory time tau; no single value.
   New main-text figure: the h/tau crossover (`explain_htau` -> rename for thesis).
3. The real content is memory: multi-year persistence matches the data's horizon signature
   (Gaussianization + persistent fat tails); iid models cannot.
4. Honest limitation: beta magnitude not pinnable from available data; claim is qualitative.
5. Discussion: updated to this framing.

**Cut:** mean-degree finite-size sections + figures (`growing_finite_economy`,
`growing_beta_extrapolation`, the beta(N)->0.15 argument). Audit `growing_stationary_msb`,
`growing_trajectories`, `growing_growth_churn`, `phase_diagram` (mean-degree) — keep only if
re-castable as own-degree or model-agnostic; otherwise cut.

## Appendix D (`appendix_dmft.tex`) — complete the deferral

Replace the closing "natural next step" paragraph with the **two-time DMFT** result:
- validation: relaxed phase recovered exactly; growth rate matches the matched sim;
- aging -> entry -> stationary: at lambda=0 the fluctuating phase ages (no stationary beta);
  firm entry (lambda>0, the regime the runs use) makes it stationary; this is the mechanism
  that sets the memory time tau behind the crossover.
- New figure: `dmft_twotime` (validation). Keep `dmft_phase`, `dmft_validation`.

## Constraints

- Prose style (per user memory): plain M2-student voice; no \emph; no em-dashes; keep
  en-dashes and math. Edit prose only, match surrounding style.
- Figures already exist in `growing_glv/` (`dmft_twotime.png`, `explain_htau.png`); copy +
  rename into `thesis/` per the FIGURES.md convention, and record provenance in FIGURES.md.
- Do not overclaim. The cross-sectional size-growing-correlation question (M-B) is a SEPARATE
  open issue from the temporal-memory point; keep them distinct.

## Order of work (low-risk first)

1. Appendix two-time section (additive, safe).
2. FIGURES.md provenance update + figure copy/rename.
3. Firm-growth chapter reframe (higher-risk: cuts user prose + rewrites). Show cut/keep list
   before slashing.
