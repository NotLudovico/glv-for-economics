# Thesis Restructure (article-style) Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a LaTeX prose migration, not code — each task's "test" is **`bash build.sh` compiles clean + no content lost + all `\ref`/`\eqref`/`\cite` resolve (no `??` in the PDF)**, followed by a commit.

**Goal:** Reorganize the thesis from a 5-chapter two-stage narrative into a lean, article-style spine (Intro → Model → Results → Conclusion) with a larger appendix, demoting the Stage-1 critical model to an appendix and bringing the orphaned DMFT into the thesis.

**Architecture:** Lead with the relative-GLV model. Stage 1 (absolute critical GLV + rescaling machinery) survives only as (a) a ~½-page motivation inside the Model chapter and (b) a full appendix. DMFT results go in the Model chapter; its derivation goes to an appendix.

**Tech Stack:** LaTeX (report class, biblatex), `build.sh` (latexmk).

## Global Constraints

- Prose style: plain M2-student voice; **no `\emph`, no em-dashes**; keep en-dashes and math. (Edit prose only.)
- **Lose no content.** Migrated prose moves verbatim unless a step explicitly rewrites it. When in doubt, relocate rather than delete.
- Every task ends compiling: `bash build.sh` succeeds and the PDF has no `??` unresolved references.
- Preserve every `\label`. When a chapter becomes an appendix, repoint its `\ref`s, do not orphan them.
- Commit after each task. Work on a branch, not `main`.

## Target structure

```
main.tex
  \include introduction          (EXPAND)
  \include model                 (NEW: relative GLV + motivation + phase diagram + DMFT results)
  \include firm_growth           (NEW: results — split from growing_economy)
  \include conclusion            (EXPAND)
  \appendix
    A absolute_model   (Stage 1: blow-up, rescaling, critical relaxation-transient results)
    B dmft             (DMFT derivation + validation)
    C numerics         (solver, immigration floor, MAD estimator, listing/fit protocol)
```

## Content-migration map (where every existing block goes)

| Source | Destination |
|--------|-------------|
| `introduction.tex` | stays → **Intro** (expanded; roadmap paragraph rewritten for new structure) |
| `growing_economy.tex` §relative-model + phase diagram | **model.tex** |
| `growing_economy.tex` §§firm-growth-stats → discussion | **firm_growth.tex** |
| `growing_economy.tex` solver/floor/MAD/fit asides | **appendix C** (pointer left behind) |
| `dynamical_equations.tex` (all) | **appendix A** |
| `results.tex` (Stage-1 results) | **appendix A** |
| `main.tex` inline appendix (rescaled-eq derivation) | **appendix A** (consolidate) |
| `dmft_theory.tex` derivation | **appendix B** |
| `dmft_theory.tex` σ_c=√2 + validation result/figure | summarized into **model.tex** |
| Stage-1 motivation (absolute model is wrong) | **distilled** into model.tex (~½ page) |

## Reference-repointing checklist (the breakage risk)

- `\ref{chap:rescaling}` → `\ref{app:absolute}`
- `\ref{chap:results}` → `\ref{app:absolute}` (or a `\label{sec:stage1-transient}` inside it)
- `\label{eq:glv}` (absolute GLV) must be **re-homed into model.tex's motivation** — it is referenced before the appendix. Appendix A then refers back to it.
- `growing_economy.tex` opening ("The previous chapter left one of the two facts unexplained…") rewritten to stand alone — there is no previous chapter anymore.
- After every task: grep the build log / PDF for `??` and fix.

---

### Task 1: Branch + split the spine (Model / Results)

**Files:** Create `chapters/model.tex`, `chapters/firm_growth.tex`; Modify `main.tex`; (source) `chapters/growing_economy.tex`.

- [ ] **Step 1:** `git checkout -b thesis-restructure`.
- [ ] **Step 2:** Read `growing_economy.tex` in full. Cut at `\section{Firm-growth statistics in the fluctuating state}` (`\label{sec:growing-observables}`).
- [ ] **Step 3:** `model.tex` = `\chapter{The model}\label{chap:model}` + everything **before** that cut (relative-model section, phase diagram, the two early figures). Verbatim.
- [ ] **Step 4:** `firm_growth.tex` = `\chapter{Firm-growth statistics}\label{chap:results}` + everything **from** the cut to the end (stats, finite-size, multiscaling, conditional tests, normalization, ablation, discussion). Verbatim. (Reuse `chap:results` label so surviving refs to "results" still point at the empirical results — the *static* Stage-1 results move to the appendix under a new label.)
- [ ] **Step 5:** In `main.tex`, replace `\include{chapters/growing_economy}` with `\include{chapters/model}` then `\include{chapters/firm_growth}`. Leave the old Stage-1 includes for now.
- [ ] **Verify:** `bash build.sh` compiles. **Commit:** `refactor(thesis): split growing-economy into model + firm-growth chapters`.

### Task 2: Appendix A — absolute model & Stage-1 (relocate, don't delete)

**Files:** Create `chapters/appendix_absolute.tex`; Modify `main.tex`; (sources) `dynamical_equations.tex`, `results.tex`, inline appendix in `main.tex`.

- [ ] **Step 1:** Read `dynamical_equations.tex` and `results.tex` in full.
- [ ] **Step 2:** `appendix_absolute.tex` (`\section{...}\label{app:absolute}` under `\appendix`): the absolute GLV, the blow-up, the stretched-time rescaling, μ_c location (from `dynamical_equations.tex`), then the Stage-1 critical results — heavy tails recovered, size–variance is a relaxation transient (from `results.tex`). Move the inline rescaled-eq derivation now in `main.tex` here too, removing the duplicate. Add `\label{sec:stage1-transient}` at the transient result.
- [ ] **Step 3:** Remove `\include{chapters/dynamical_equations}` and `\include{chapters/results}` from the main chain; add `\include{chapters/appendix_absolute}` after `\appendix`. Delete the now-empty inline appendix block in `main.tex`.
- [ ] **Verify:** `bash build.sh` compiles (expect `??` for the chap refs — fixed in Task 5). **Commit:** `refactor(thesis): move stage-1 + rescaling machinery to appendix A`.

### Task 3: Appendix B — DMFT derivation

**Files:** Create `chapters/appendix_dmft.tex`; Modify `main.tex`; (source) `dmft_theory.tex`.

- [ ] **Step 1:** Read `dmft_theory.tex` in full. Identify the derivation (single-site reduction, FP/Bunin map, σ_c=√2) vs the headline results + validation figure.
- [ ] **Step 2:** `appendix_dmft.tex` (`\label{app:dmft}`): the full derivation, verbatim, adapted to appendix sectioning. Keep the validation figure here (and/or reference it from the model chapter).
- [ ] **Step 3:** Add `\include{chapters/appendix_dmft}` after Appendix A.
- [ ] **Verify:** `bash build.sh` compiles. **Commit:** `feat(thesis): add DMFT derivation as appendix B`.

### Task 4: Appendix C — numerics & protocol

**Files:** Create `chapters/appendix_numerics.tex`; Modify `model.tex`, `firm_growth.tex`.

- [ ] **Step 1:** `appendix_numerics.tex` (`\label{app:numerics}`): the explicit-RK solver choice + timing, immigration floor λ=10⁻³, the √(π/2) MAD volatility estimator, the listed-firm protocol (>1% mean, ≥20 rates), the plateau/R² fit rule. Pull these from the methodological paragraphs in `model.tex`/`firm_growth.tex`.
- [ ] **Step 2:** In the chapters, replace each moved aside with a one-line pointer (e.g. "(solver and protocol details in Appendix~\ref{app:numerics})"), keeping the result sentences.
- [ ] **Step 3:** Add `\include{chapters/appendix_numerics}` last.
- [ ] **Verify:** `bash build.sh` compiles. **Commit:** `refactor(thesis): collect numerics + measurement protocol into appendix C`.

### Task 5: Re-home `eq:glv` + write the Stage-1 motivation in the Model chapter

**Files:** Modify `chapters/model.tex`, `chapters/appendix_absolute.tex`.

- [ ] **Step 1:** At the top of `model.tex`, write the motivating passage (~½ page, new prose, plain voice): GLV as interacting firms; the absolute model has a fixed size scale and blows up in finite time, and even rescaled to criticality gives only a relaxation transient, not a stationary size–variance law — full treatment in Appendix~\ref{app:absolute}. Define the absolute GLV here with `\label{eq:glv}`.
- [ ] **Step 2:** In `appendix_absolute.tex`, change its local redefinition of the absolute GLV to a reference back to `\eqref{eq:glv}` (avoid a duplicate equation/label).
- [ ] **Step 3:** Rewrite the old opening of the Model chapter ("The previous chapter left one of the two facts unexplained…") to flow from the new motivation and stand alone.
- [ ] **Step 4:** Repoint refs per the checklist: `\ref{chap:rescaling}`→`\ref{app:absolute}`; Stage-1-specific `\ref{chap:results}`→`\ref{sec:stage1-transient}` (leave references to the *empirical* results chapter pointing at `chap:results`).
- [ ] **Verify:** `bash build.sh` compiles, **zero `??` in PDF**. **Commit:** `refactor(thesis): re-home eq:glv and rewrite stage-1 as model-chapter motivation`.

### Task 6: DMFT results into the Model chapter

**Files:** Modify `chapters/model.tex`.

- [ ] **Step 1:** After the phase-diagram material, add a short subsection: σ_c=√2 derived from DMFT (not just cited from Diederich), the FP/Bunin correspondence, and the ~1% validation against matched simulation, referencing Appendix~\ref{app:dmft} and the validation figure. Soften the current bare `\citep{diederich1989}` for σ_c to "we derive (Appendix~\ref{app:dmft}) … consistent with \citep{diederich1989}".
- [ ] **Verify:** `bash build.sh` compiles. **Commit:** `feat(thesis): summarize DMFT results in the model chapter`.

### Task 7: Expand the Introduction

**Files:** Modify `chapters/introduction.tex`.

- [ ] **Step 1:** Deepen the problem + literature treatment (the user's main ask): more on why the two laws matter, the granular/sub-unit lineage, and a sharper account of Morán's three discriminating tests and how the granular picture fails them. Keep the existing strong material; add depth, don't replace.
- [ ] **Step 2:** Rewrite the final roadmap paragraph for the new structure (Model / Results / Conclusion + appendices). Remove references to the Stage-1 chapters as main chapters.
- [ ] **Verify:** `bash build.sh` compiles, refs resolve. **Commit:** `docs(thesis): expand introduction problem + literature, update roadmap`.

### Task 8: Expand & realign the Conclusion

**Files:** Modify `chapters/conclusion.tex`.

- [ ] **Step 1:** Rewrite for the new spine; drop the "two steps / Stage-1" framing, foreground the relative model as the thesis's model. Reconcile the qualifications count (chapter says "four", conclusion says "three") — pick one and make both agree.
- [ ] **Step 2:** Expand from ~700 words to a fuller close (contribution, honest limits, next steps), without repeating the chapter discussion verbatim.
- [ ] **Verify:** `bash build.sh` compiles. **Commit:** `docs(thesis): expand and realign conclusion to new structure`.

### Task 9: Whole-thesis coherence pass

**Files:** All chapters, `main.tex`, abstract.

- [ ] **Step 1:** Read the built PDF end to end. Fix any remaining `??`, stale "previous/next chapter" phrasing, ToC order, and figure placement.
- [ ] **Step 2:** Reconcile the abstract's two-stage description with the new article structure (it currently narrates Stage 1 → Stage 2; trim to match, or keep as honest scientific arc — author's call).
- [ ] **Step 3:** Confirm `git status` clean, branch ready. **Commit:** `docs(thesis): coherence pass after restructure`.

---

## Self-review notes
- **Coverage:** Intro expand (T7), Stage-1→appendix (T2,T5), DMFT main+appendix (T3,T6), article spine (T1), bigger appendix (T2,T3,T4), conclusion (T8) — all three user decisions covered.
- **Ordering rationale:** mechanical relocations (T1–T4) before creative rewrites (T5–T8) so the risky prose work happens on a structure that already compiles; coherence pass last.
- **Biggest risk:** ref/narrative breakage from demoting Stage 1 — isolated to T5 and verified (`??`-free) there and again in T9.
