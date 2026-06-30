# Thesis rewrite — design

**Date:** 2026-06-30
**Branch:** thesis-restructure

## Goal

Rewrite the thesis in the author's own voice with a clean skeleton. **Keep all
results and figures**; rebuild the chapter structure and the prose. This is a
restructure + voice rewrite, not new science.

## Voice

Extracted from the author's bachelor's thesis (`thesis/Modello di tipo dimero e
applicazione al modello di Ising.pdf`) and confirmed. See memory
`feedback_thesis_voice` + `feedback_thesis_prose_style`.

- **Motivate, then define.** Introduce machinery only when needed, stating *why*
  first, then the definition. Never a definition dropped cold.
- **Explicit signposting**, forward and back ("we will see…", "from now on we
  assume…", "as in §X").
- **Plain why-asides** — explain each choice in ordinary words. No flourish, no hype.
- **Abstraction then a concrete worked example.**
- **Quiet impersonal "we".** Calm, rigorous, textbook-steady.
- Plain M2 voice: no `\emph`, no em-dashes; keep en-dashes and math. Edit prose only.

## Model scope

Own-degree relative-GLV is the **only** model. The mean-degree normalisation and
the finite-economy-size β reframe are dropped and must not resurface. See memory
`feedback_owndegree_only`.

## Skeleton

**Ch1 — Introduction**
- The empirical puzzle: Moran–Bouchaud firm-growth laws (size–volatility exponent,
  symmetric fat-tailed growth tent, multiscaling) and why they have resisted
  explanation.
- The ecosystem question: can disordered interactions generate them? One line
  flagging that the obvious GLV does not, motivating what follows.
- Roadmap.

**Ch2 — The model**
- GLV background.
- *Why not the obvious GLV* — the critical-GLV failure (blow-up / one-sided
  growth) stated as the problem the relative interaction fixes. (brief motivating
  section; the substantive version lands here, Ch1 only flags it.)
- The relative (mean-normalised) interaction → steady growth; the own-degree
  normalisation (`eq:normalizations`).
- Phase structure + phase diagram (growing/fluctuating chaotic phase).
- Figures: `phase_diagram.png`, `growing_trajectories.png`.

**Ch3 — Firm-growth statistics**
- Size–volatility β and the growth tent. Figs: `growing_stationary_msb.png`,
  `growing_growth_churn.png`.
- Finite size → mean-field limit.
- Higher moments / multiscaling + conditional tests D1/D2/D3. Figs:
  `growing_multiscaling.png`, `growing_conditional.png`.
- The clock: β as a function of h/τ. Fig: `growing_clock.png`.
- Role of the interactions (ablation).
- Discussion: β≈0.8, 4× too steep → a missing ingredient (size-dependent
  correlations of a firm built from many parts).

**Ch4 — Conclusion** — what is established, the open magnitude problem, outlook.

**Appendix A — DMFT** of the relative model (content unchanged). Figs: `dmft_*.png`.
**Appendix B — Numerical methods and measurement protocol.**

## Out of scope

- New simulations, new figures, new results. Figures are reused as-is. (Exception:
  `growing_clock.png` is an orphan with no generator; regenerate from the
  own-degree model only if the author later wants its numbers aligned — not part
  of this rewrite.)
- The mean-degree model and the finite-size-β reframe.
- The critical-GLV "stage-1" appendices (stay dropped; survives only as the brief
  Ch2 motivating section).

## Process

Rewrite chapter by chapter, in order, reusing the existing figures and equations.
Each chapter: agree the section beats (already outlined above), draft in the
voice, author reviews before moving to the next. `main.tex` includes stay as they
are (intro, model, firm_growth, conclusion, appendix_dmft, appendix_numerics).
`FIGURES.md` provenance stays the source of truth for figure → generator.
