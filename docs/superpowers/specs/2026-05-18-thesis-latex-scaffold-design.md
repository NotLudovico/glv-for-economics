# Thesis LaTeX Scaffold — Design

**Date:** 2026-05-18
**Author:** Ludovico Furlanetto
**Status:** Approved

## Goal

Set up a structured LaTeX folder for a Master's thesis ("Can a simple model explain economics?") inside the existing GLV research repository. Migrate the user's current single-file `article`-class draft into a modular, chapter-based `report`-class layout with automated builds and a bibliography pipeline ready to use.

## Non-Goals

- Writing new thesis content. Migration preserves existing text verbatim (chapter/section levels are remapped, but prose is unchanged).
- Producing the figure PNGs referenced in the source. The figures directory will be created with a `.gitkeep`; figure files are generated outside this scaffold.
- Choosing a final bibliography style. `numeric-comp` is the default; user can swap later.
- Adding any university-specific branding or class file.

## Location

`/Users/ludovicofurlanetto/Code/glv/thesis/`

Lives alongside `papers/`, `images/`, `notebooks/`, etc. No nesting under `docs/`.

## Directory Layout

```
thesis/
├── main.tex                  # \documentclass{report}, \input preamble, \include chapters
├── preamble.tex              # packages, math macros, hyperref, biblatex, minted
├── metadata.tex              # \title, \author, \date, abstract text
├── chapters/
│   ├── 01-introduction.tex
│   ├── 02-rescaling.tex
│   ├── 03-results.tex        # placeholder chapter
│   └── 04-conclusions.tex    # placeholder chapter
├── appendix/
│   └── A-rescaled-equations.tex
├── figures/
│   └── .gitkeep              # PNGs live here; \graphicspath set to this dir
├── bibliography/
│   └── references.bib        # empty stub; biber backend
├── .latexmkrc                # pdflatex -shell-escape, biber
└── .gitignore                # LaTeX aux files, _minted-*; main.pdf is committed
```

## Class and Top-Level Structure

- **Document class:** `report` (matches the abstract's references to "Chapter 2/3/4").
- **Heading remap from current source:**
  - `\section{Introduction}` → `\chapter{Introduction}`
  - `\section{Dynamical equations rescaling}` → `\chapter{Dynamical equations rescaling}`
  - All `\subsection` inside become `\section`
  - All `\subsubsection` inside become `\subsection`
  - Appendix `\section{Rescaled Equations}` stays a chapter-level heading under `\appendix`

## File Responsibilities

- **`main.tex`** — `\documentclass`, `\input{preamble}`, `\input{metadata}`, `\begin{document}`, `\maketitle`, abstract block, `\tableofcontents`, `\include` chapter files, `\appendix`, `\include` appendix files, `\printbibliography`, `\end{document}`.
- **`preamble.tex`** — all `\usepackage` lines, `\graphicspath{{figures/}}`, `\addbibresource{bibliography/references.bib}`, minted setup, hyperref colors, theorem environments (if any).
- **`metadata.tex`** — `\title`, `\author`, `\date`, and the abstract paragraph as a macro `\abstracttext` (so `main.tex` can drop it into an `abstract` environment).
- **Chapter files** — content only. No `\documentclass`, no `\begin{document}`. Each begins with `\chapter{...}` and contains the relevant sections.
- **Appendix file** — content only, begins with `\chapter{Rescaled Equations}` (rendered as "Appendix A" once `\appendix` is active).

## Migration Plan for Existing Source

The user's current `\documentclass{article}` source is split as follows:

| Current source block | Destination |
|----------------------|-------------|
| `\title`, `\author`, `\date` | `metadata.tex` |
| `\begin{abstract}` content | `metadata.tex` (as `\abstracttext`) |
| `\section{Introduction}` body | `chapters/01-introduction.tex` (as `\chapter{Introduction}`) |
| `\section{Dynamical equations rescaling}` block (all subsections including the minted code, plot subsections, and $\mu_c$ subsections) | `chapters/02-rescaling.tex` |
| `\appendix \section{Rescaled Equations}` | `appendix/A-rescaled-equations.tex` |

Prose is not rewritten; only structural commands change. The minted Python block is preserved unchanged (the `\begin{minipage}{\textwidth}` wrapper remains).

Placeholder chapters `03-results.tex` and `04-conclusions.tex` contain only `\chapter{...}` lines and a comment marking them as TODO.

## Preamble Contents

Math and typography:

- `amsmath`, `amssymb`, `amsthm`, `mathtools` — math typesetting
- `bm` — bold math symbols
- `cancel` — already used in the existing source
- `siunitx` — units

Layout and graphics:

- `geometry` — page margins (default ~1 inch on all sides)
- `graphicx` with `\graphicspath{{figures/}}`
- `caption` — better caption formatting

Code listings:

- `minted` — Python syntax highlighting (matches existing usage)
- No global `\setminted` config; `frame=lines, fontsize=\small` may be added if requested later

Cross-references and links:

- `hyperref` — colored links (`colorlinks=true`, sensible link/cite/url colors)
- `cleveref` — `\cref{eq:foo}` for nicer references; loaded last among reference packages

Bibliography:

- `biblatex` with `backend=biber, style=numeric-comp, sorting=none`
- `\addbibresource{bibliography/references.bib}` in the preamble
- `\printbibliography` at the end of `main.tex`

## Build Configuration

**`.latexmkrc`:**

```perl
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
$biber = 'biber %O %S';
$clean_ext = '_minted-main bbl run.xml synctex.gz';
```

This enables shell-escape for minted, runs biber for biblatex, and adds minted artifacts to `latexmk -c`.

Single build command: `latexmk -pdf main.tex` (run from `thesis/`).

## `.gitignore`

Per-folder `.gitignore` covering LaTeX intermediates:

```
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc
*.lof
*.lot
_minted-*/
```

`main.pdf` is **not** ignored — user requested it be committed for easy sharing.

## Build / Verification

After scaffolding, `latexmk -pdf main.tex` from `thesis/` must produce `main.pdf` with:

- A title page (title, author, date)
- An abstract page
- A table of contents listing the chapters and appendix
- Chapter 1 (Introduction) with the migrated intro prose
- Chapter 2 (Dynamical equations rescaling) with all the math, the minted Python block, and the figure placeholders
- Chapters 3 and 4 as one-line placeholders
- Appendix A with the rescaled-equations derivation
- An empty bibliography section

Figure `\includegraphics` calls referencing files that don't yet exist on disk (`rescaled_unscaled_mu_minus_2.png`, etc.) will cause build warnings/errors. Mitigation: comment those `\begin{figure}` blocks out in `02-rescaling.tex` with a `% FIXME: figure file pending` marker, so the first build succeeds. User uncomments once PNGs are placed in `figures/`.

Requires `pygments` (`pip install Pygments`) on PATH for minted. If not present, build will fail with a clear minted error.

## Risks and Open Questions

- **Pygments availability.** Minted requires Python's `pygments`. The user's environment uses `uv`. If `pygments` is not globally available, `latexmk` will fail. Implementation step will verify and document this.
- **Figure compilation on first build.** Resolved by commenting out figure blocks with a TODO marker (see Build / Verification above).
- **Bibliography style.** `numeric-comp` chosen as a generic default; trivial to change later in `preamble.tex`.
