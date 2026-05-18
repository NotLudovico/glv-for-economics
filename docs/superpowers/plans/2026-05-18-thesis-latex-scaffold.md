# Thesis LaTeX Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the existing single-file `article`-class thesis draft into a modular, chapter-based `report`-class LaTeX project at `/Users/ludovicofurlanetto/Code/glv/thesis/`, with `latexmk` + `biblatex` + `minted` wired up and a successful first PDF build.

**Architecture:** A `thesis/` folder containing `main.tex`, `preamble.tex`, `metadata.tex`, per-chapter files under `chapters/`, an appendix file under `appendix/`, an empty `bibliography/references.bib`, and a `figures/` placeholder. Existing prose is preserved verbatim; only structural commands (`\section` → `\chapter`, etc.) are remapped. A `.latexmkrc` enables `-shell-escape` for minted and configures `biber` as the bibliography backend.

**Tech Stack:** LaTeX (`report` class), `latexmk`, `biber`, `biblatex`, `minted` (requires Pygments), `hyperref`, `cleveref`, `amsmath`/`amssymb`/`mathtools`, `graphicx`, `geometry`, `siunitx`.

**Spec:** [docs/superpowers/specs/2026-05-18-thesis-latex-scaffold-design.md](../specs/2026-05-18-thesis-latex-scaffold-design.md)

---

## Notes for the Implementing Engineer

- This is a documentation/scaffold task, not a code-feature task. There is no unit-test framework; "tests" here are **build verifications**: `latexmk -pdf main.tex` succeeds and the resulting PDF has the expected structure.
- Work entirely inside `/Users/ludovicofurlanetto/Code/glv/thesis/`. Do not modify files outside this directory except where explicitly noted.
- Each task is committed separately. Commit messages use conventional commits (`feat:`, `chore:`, `docs:`).
- The user's existing `main.tex` content (the long source they pasted into the brainstorming session) is the source of truth for prose. It is reproduced verbatim inside the relevant tasks below — do not paraphrase, summarize, or fix typos in the prose.
- The current working directory is `/Users/ludovicofurlanetto/Code/glv`. All bash commands assume this CWD unless specified.

---

## Task 1: Verify LaTeX toolchain

**Files:** none (verification only)

- [ ] **Step 1: Check for required binaries**

Run:
```bash
which pdflatex biber latexmk pygmentize
```

Expected: all four paths printed. If any are missing, **stop and inform the user**. Install hints:

- macOS: `brew install --cask mactex-no-gui` (provides `pdflatex`, `biber`, `latexmk`), then `pip install Pygments` for `pygmentize`.
- Or `brew install basictex` + `sudo tlmgr install latexmk biber minted` + `pip install Pygments`.

Do not invent a workaround. Wait for the user to install and re-run this check.

- [ ] **Step 2: Confirm versions are recent enough**

Run:
```bash
pdflatex --version | head -1
biber --version | head -1
latexmk --version | head -1
pygmentize -V
```

Expected: pdflatex from any TeX Live ≥ 2020; biber ≥ 2.15; latexmk ≥ 4.70; Pygments ≥ 2.0. Anything older — flag to user but proceed.

- [ ] **Step 3: No commit**

Verification step only.

---

## Task 2: Create empty folder structure

**Files:**
- Create: `thesis/` (directory)
- Create: `thesis/chapters/` (directory)
- Create: `thesis/appendix/` (directory)
- Create: `thesis/figures/` (directory)
- Create: `thesis/figures/.gitkeep` (empty file)
- Create: `thesis/bibliography/` (directory)

- [ ] **Step 1: Make directories and the figures placeholder**

Run:
```bash
mkdir -p thesis/chapters thesis/appendix thesis/figures thesis/bibliography
touch thesis/figures/.gitkeep
```

- [ ] **Step 2: Verify layout**

Run:
```bash
find thesis -type d
ls thesis/figures
```

Expected output:
```
thesis
thesis/chapters
thesis/appendix
thesis/figures
thesis/bibliography
.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add thesis/figures/.gitkeep
git commit -m "chore(thesis): create thesis folder structure"
```

Note: empty directories aren't tracked by git, but `.gitkeep` materializes `thesis/` + `thesis/figures/`. The other empty subdirs will be tracked when files land in them in subsequent tasks.

---

## Task 3: Write `.gitignore`

**Files:**
- Create: `thesis/.gitignore`

- [ ] **Step 1: Write the file**

Create `thesis/.gitignore` with:

```gitignore
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

Note: `main.pdf` is intentionally **not** ignored — the user wants the rendered PDF committed.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/.gitignore
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/.gitignore
git commit -m "chore(thesis): add gitignore for LaTeX aux files"
```

---

## Task 4: Write `.latexmkrc`

**Files:**
- Create: `thesis/.latexmkrc`

- [ ] **Step 1: Write the file**

Create `thesis/.latexmkrc` with:

```perl
$pdf_mode = 1;
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
$biber = 'biber %O %S';
$clean_ext = '_minted-main bbl run.xml synctex.gz';
```

What this does:
- `$pdf_mode = 1` — produce PDF via pdflatex.
- `$pdflatex` — enables `-shell-escape` (required by minted) and non-interactive mode.
- `$bibtex_use = 2` — always run bibliography step when a `.bcf` file exists.
- `$biber` — explicit biber invocation (biblatex's default backend).
- `$clean_ext` — extra extensions to remove on `latexmk -c`.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/.latexmkrc
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/.latexmkrc
git commit -m "chore(thesis): add latexmkrc with shell-escape and biber"
```

---

## Task 5: Write `preamble.tex`

**Files:**
- Create: `thesis/preamble.tex`

- [ ] **Step 1: Write the file**

Create `thesis/preamble.tex` with:

```latex
% ---- Math ----
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{bm}
\usepackage{cancel}
\usepackage{siunitx}

% ---- Layout & graphics ----
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{caption}

% ---- Code listings ----
\usepackage{minted}

% ---- Bibliography ----
\usepackage[backend=biber,style=numeric-comp,sorting=none]{biblatex}
\addbibresource{bibliography/references.bib}

% ---- Links & cross-references (load hyperref before cleveref) ----
\usepackage[colorlinks=true,
            linkcolor=blue!60!black,
            citecolor=green!50!black,
            urlcolor=blue!60!black]{hyperref}
\usepackage{cleveref}
```

Notes for the engineer:
- `\graphicspath{{figures/}}` makes `\includegraphics{foo.png}` resolve to `figures/foo.png`. The double braces are intentional — `graphicx` syntax.
- `hyperref` must be loaded after most packages and before `cleveref`. Order matters.
- `sorting=none` for biblatex preserves citation order in the bibliography; change later if user prefers alphabetical.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/preamble.tex
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/preamble.tex
git commit -m "feat(thesis): add preamble with math, biblatex, minted, hyperref"
```

---

## Task 6: Write `metadata.tex`

**Files:**
- Create: `thesis/metadata.tex`

- [ ] **Step 1: Write the file**

Create `thesis/metadata.tex` with:

```latex
\title{Can a simple model explain economics?}
\author{Ludovico Furlanetto}
\date{May 2026}

\newcommand{\abstracttext}{%
Probably not%
}
```

The abstract content is wrapped in a macro so `main.tex` can drop it inside the `abstract` environment. The trailing `%` on `Probably not%` and on the line above prevent stray whitespace.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/metadata.tex
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/metadata.tex
git commit -m "feat(thesis): add title/author/date and abstract macro"
```

---

## Task 7: Write empty `bibliography/references.bib`

**Files:**
- Create: `thesis/bibliography/references.bib`

- [ ] **Step 1: Write the file**

Create `thesis/bibliography/references.bib` with:

```bibtex
% Bibliography entries go here.
% Example:
% @article{example2026,
%   author  = {Doe, Jane},
%   title   = {An Example Paper},
%   journal = {Journal of Examples},
%   year    = {2026},
% }
```

Biber requires the `.bib` file to exist (even if empty) when `\addbibresource` references it. Comment-only files are valid.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/bibliography/references.bib
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/bibliography/references.bib
git commit -m "chore(thesis): add empty bibliography stub"
```

---

## Task 8: Write `chapters/01-introduction.tex`

**Files:**
- Create: `thesis/chapters/01-introduction.tex`

Prose is migrated verbatim from the user's existing source. Only the heading level changes (the original `\section{Introduction}` becomes `\chapter{Introduction}`).

- [ ] **Step 1: Write the file**

Create `thesis/chapters/01-introduction.tex` with:

```latex
\chapter{Introduction}

The dynamics of firm growth and economic fluctuations have long been modeled through the lens of complex interacting systems. One of the most robust empirical observations in quantitative economics—often referred to as a "stylized fact"—is the scaling relationship between the size of a firm ($S$) and the standard deviation of its growth rate ($\sigma$). Empirical data consistently demonstrates a power-law decay of the form $\sigma(S) \propto S^{-\beta}$, where the scaling exponent $\beta$ is typically found in the range of $0.15$ to $0.20$. From a statistical physics perspective, this exponent is highly anomalous. If a firm were composed of independent, uncorrelated sub-units, the Central Limit Theorem (CLT) would strictly dictate an exponent of $\beta = 0.5$. The empirically observed suppression of this exponent implies the existence of deep, systemic, intra- and inter-firm correlations that cannot be explained by standard additive random walks.

To uncover the microscopic origins of these macroscopic correlations, agent-based network models have become a primary theoretical tool. Specifically, the Generalized Lotka-Volterra (GLV) model—adapted from theoretical ecology—provides a deterministic framework to study the competitive and cooperative interactions between distinct economic nodes. However, extracting continuous, non-decaying volatility from a deterministic model requires situating the system at a very specific thermodynamic boundary: the critical phase transition ($\mu = \mu_c$). At topological criticality, the restoring forces of the system vanish, and long-lived transient macroscopic fluctuations naturally emerge.

Simulating interacting systems exactly at this divergence point introduces a severe computational bottleneck. As the interaction strength approaches the critical threshold, the macroscopic variables diverge, causing standard continuous ordinary differential equation (ODE) solvers to fail due to infinitesimally small step sizes or machine precision overflow.

The primary objective of this thesis is twofold. First, we engineer a robust numerical framework to bypass this critical divergence. By applying a coordinate transformation that maps the unbounded abundances onto a bounded probability simplex, coupled with a dynamic time-rescaling, we establish a method to integrate the GLV equations seamlessly through the critical singularity. Second, we deploy this computational tool to formally test whether the transient critical dynamics of the static GLV model are sufficient to break the Central Limit Theorem and recover the empirical volatility scaling exponent of $\beta \approx 0.2$.

Applying this stabilized numerical framework, we conduct a rigorous stress-test of the GLV model across different network topologies to evaluate its capacity to generate realistic economic fluctuations. We first investigate the system on networks with exponential degree distributions. Because the variance of an exponential degree distribution is finite and well-behaved, a macroscopic critical threshold ($\mu_c > 0$) exists, allowing the system to be simulated at criticality. However, our findings indicate that the resulting volatility scaling exponent is tightly bound around $\beta \approx 0.5$. This mathematically demonstrates that, in the absence of heavy-tailed topological hierarchies or multiplicative external shocks, the deterministic interactions within the static GLV model effectively average out as mean-field noise. Consequently, the system remains strictly governed by the Central Limit Theorem, failing to produce the deep correlations required to match empirical economic data.

Furthermore, we extend our analysis to scale-free (power-law) networks, which more accurately reflect the topology of real-world economic and financial systems. In this regime, the theoretical limits of the continuous GLV model are laid bare. Because the second moment of a power-law degree distribution diverges in the thermodynamic limit ($\langle g^2 \rangle \to \infty$), the theoretical critical threshold vanishes ($\mu_c \to 0$). We demonstrate that attempting to simulate such a system does not merely result in a failure of the algorithm, but reflects a physical reality of the model: an instantaneous "monopoly collapse." A single massive hub node exponentially dominates the system, forcing the relative fractions of all other nodes in the simplex to drop so violently that they trigger a 64-bit machine precision underflow. This confirms that without structural cutoffs or artificial bounds, continuous GLV interactions are incompatible with the scale-free topologies required to break the Central Limit Theorem.

Ultimately, this thesis maps the exact boundaries of where static topological criticality ceases to describe economic reality, providing a definitive falsification of the simple GLV model as a standalone mechanism for empirical volatility scaling.

The remainder of this thesis is structured as follows. Chapter 2 details the mathematical derivation of the coordinate transformation, formally mapping the diverging GLV equations to the probability simplex, and discusses the limits of the analytical mean-field approximations. Chapter 3 presents the numerical methodology and the empirical results, illustrating the successful bypass of the computational divergence and analyzing the resulting scaling behaviors on both exponential and power-law networks. Finally, Chapter 4 concludes the work, summarizing the implications of these findings and outlining potential theoretical extensions, such as the introduction of multiplicative stochastic noise, required to recover realistic macroeconomic dynamics.
```

What changed from the original source:
- `\section{Introduction}` → `\chapter{Introduction}`.
- The original was a single block of prose with no paragraph breaks. Paragraph breaks have been inserted at the natural transitions identified by the user's wording ("To uncover...", "Simulating...", "The primary objective...", "Applying this stabilized...", "Furthermore...", "Ultimately...", "The remainder..."). No words were changed.

- [ ] **Step 2: Verify**

Run:
```bash
wc -l thesis/chapters/01-introduction.tex
head -5 thesis/chapters/01-introduction.tex
```

Expected: file exists, first line is `\chapter{Introduction}`.

- [ ] **Step 3: Commit**

```bash
git add thesis/chapters/01-introduction.tex
git commit -m "feat(thesis): add introduction chapter"
```

---

## Task 9: Write `chapters/02-rescaling.tex`

**Files:**
- Create: `thesis/chapters/02-rescaling.tex`

This is the largest file. Original prose preserved verbatim. The user's `\section{Dynamical equations rescaling}` becomes `\chapter`, original `\subsection` → `\section`, original `\subsubsection` → `\subsection`. The figure blocks are **commented out** with a TODO marker so the first build succeeds before figure PNGs land in `figures/`.

- [ ] **Step 1: Write the file**

Create `thesis/chapters/02-rescaling.tex` with:

```latex
\chapter{Dynamical equations rescaling}

Starting from the standard Generalized Lotka-Volterra (GLV) equations, which govern the dynamics of interacting nodes in our network:
$$\dot{x}_i = x_i \left( 1 - x_i + \sum_j W_{ij} x_j \right)$$
we encounter severe computational divergences as the interaction strength $\mu$ approaches and exceeds the critical threshold $\mu_c$. To study the system across this critical regime without triggering numerical overflow, we perform a coordinate transformation that maps the unbounded trajectories onto a bounded probability space.

\section{Mapping to the Probability Simplex and Time Rescaling}

We introduce a macroscopic scaling factor $M(t)$ representing the total system abundance, and map the absolute abundances $x_i(t)$ to relative fractions $y_i(t)$:
$$\begin{cases}
M(t) = \sum_j x_j(t) \\
y_i(t) = \frac{x_i(t)}{M(t)}
\end{cases}$$
By definition, the rescaled variables are bounded on the probability simplex such that $y_i(t) \ge 0 \quad \forall t$ and $\sum_j y_j(t) = 1$.

While this transformation bounds the variables $y_i$, the resulting dynamical equations still depend multiplicatively on the macroscopic envelope $M(t)$, which diverges for $\mu > \mu_c$. To stabilize the integration, we introduce a non-linear stretching of time by defining a new time variable $\tau$:
$$\frac{d\tau}{dt} = M(t)$$
Applying this change of variables and time rescaling to the original GLV system (the full step-by-step algebraic derivation is provided in Appendix A), we obtain the following stable, bounded set of dynamical equations:
$$\frac{dy_i}{d\tau} = y_i \left[ \sum_{j=1}^N W_{ij} y_j - y^T W y - y_i + y^T y \right]$$
$$\frac{dM}{d\tau} = 1 + M \left[ y^T W y - y^T y \right]$$
$$\frac{dt}{d\tau} = \frac{1}{M}$$
We can also express the dynamics of the simplex directly in a fully vectorized form, utilizing element-wise multiplication ($\odot$) and a vector of ones ($\mathbf{1}$):
$$\frac{dy}{d\tau} = y \odot \left[ Wy - (y^T W y)\mathbf{1} - y + (y^T y)\mathbf{1} \right]$$
Crucially, the evolution of the relative fractions $y(\tau)$ is now completely decoupled from the divergent macroscopic envelope $M$. However, we concurrently integrate $M(\tau)$ and $t(\tau)$ alongside $y(\tau)$ to allow for the exact reconstruction of the unscaled physical trajectories $x(t)$ whenever necessary.

\section{Theoretical Limitations and the Need for Numerical Integration}

While closed-form analytical solutions for the equilibrium state $(M^*, y^*)$ can be derived mathematically (see Appendix B), they are valid only under strictly cooperative assumptions ($\mu < \mu_c$) where an interior fixed point exists and all species survive.

However, as the interaction strength reaches and exceeds the critical threshold ($\mu \ge \mu_c$), the theoretical mean-field behavior predicts the onset of relative extinctions. In this divergent regime, hub nodes grow exponentially faster than the rest of the network, forcing the relative fractions of lower-degree nodes toward zero. Consequently, the full $N \times N$ matrix $(I-W)$ is no longer invertible, and the true equilibrium transitions into a Linear Complementarity Problem (LCP) restricted to surviving active nodes.

Because this analytical framework breaks down precisely at the critical topological boundary we aim to study, we abandon the search for closed-form static equilibria. Instead, our methodology relies entirely on the numerical integration of the exact, bounded dynamical equations derived above to capture the critical transient fluctuations of the system.

\section{Numerical Integration}

The function that will be fed into the solver is the following:

\noindent
\begin{minipage}{\textwidth}
\begin{minted}{python}
def rescaled_glv_sparse(tau, state, N, W_sparse):
    """
    The numerically stable, rescaled ODE system.
    state = [y_1, y_2, ..., y_N, M, t]
    """
    y = state[:N]
    M = state[N]
    # t = state[N+1] is tracked by the solver, but doesn't affect the derivatives

    # fast sparse matrix multiplication
    F = W_sparse @ y

    # vectorized scalars
    phi = np.dot(y, F)     # The average network fitness: y^T W y
    sq_sum = np.sum(y**2)  # The self-competition penalty: y^T y

    # differential equations
    dydtau = y * (F - phi - y + sq_sum)
    dMdtau = 1.0 + M * (phi - sq_sum)
    dtdtau = 1.0 / M

    return np.concatenate((dydtau, [dMdtau], [dtdtau]))
\end{minted}
\end{minipage}

\vspace{10pt}
The only thing we must careful about is to set a max step of about $10^3$ otherwise the trajectories will reach a plateau but a big max step could introduce noise that would then destroy the unscaled trajectory.

\section{M(t) Scaling ($\sigma = 0$, $0<\mu<\mu_c$)}

In true time the evolution of $M(t)$ is governed by:
\begin{equation}
    \frac{dM}{dt} = M - M^2(y^Ty - y^T Wy)
\end{equation}

\subsection{$\mu < \mu_c$}

Effectively $(I - W)$ is positive definite, hence the growth is logistic with a dynamic carry capacity and $M^* = 0$ or $M^* = \left[(y^*)^T(I-W)y^*\right]^{-1} = 1/c$

Now onto the solution for $y^*$ (solution for $y^*=0$ is inadmissible since $\sum_j y_j = 1$):
\begin{equation}
    \begin{aligned}
    0 &= -(I - W)y^* + c\mathbf{1} \\
    y^* &= c (I - W)^{-1} \mathbf{1}
    \end{aligned}
\end{equation}
we can now use the normalization condition $\mathbf{1}^T y^* = 1$
\begin{equation}
\begin{aligned}
        \mathbf{1}^T y^* = c \Big( \mathbf{1}^T (I - W)^{-1} \mathbf{1} \Big) \\
        1 = c \Big( \mathbf{1}^T (I - W)^{-1} \mathbf{1} \Big)
\end{aligned}
\end{equation}
hence:
\begin{equation}
    c = \frac{1}{\mathbf{1}^T (I - W)^{-1} \mathbf{1}}
\end{equation}
then
\begin{equation}
   y^* = \frac{(I - W)^{-1} \mathbf{1}}{\mathbf{1}^T (I - W)^{-1} \mathbf{1}}
\end{equation}
finally
\begin{equation}
    M^* = \frac{1}{c} = \mathbf{1}^T (I - W)^{-1} \mathbf{1}
\end{equation}

\section{Plots}

Here are the results of the integration of the rescaled variables compared with the unscaled trajectories, a sample of 100 trajectories is shown. For all the plots the exponential degree distribution has been chosen.

% FIXME: figure file pending — uncomment once rescaled_unscaled_mu_minus_2.png lands in figures/
% \begin{figure}[htbp]
%     \centering
%     \begin{minipage}{1\linewidth}
%         \centering
%         \includegraphics[width=\linewidth]{rescaled_unscaled_mu_minus_2.png}
%         \caption{$N=10^4$, $C=50$, $\mu = -2 < \mu_c \approx 0.5458$, $\sigma=0.5$}
%         \label{fig:stable-unscaled}
%     \end{minipage}
%
%     \vspace{1.5em}
%
%     \begin{minipage}{1\linewidth}
%         \centering
%         \includegraphics[width=\linewidth]{rescaled_unscaled_mu_05.png}
%         \caption{$N=10^4$, $C=50$, $\mu = 0.5 \approx \mu_c \approx 0.5458$, $\sigma=0.5$}
%         \label{fig:divergin-unscaled}
%     \end{minipage}
% \end{figure}

\section{Finding $\mu_c$}

The theoretical result for $\mu_c$ fails in the context of finite graphs, hence we must find a new way to calculate the critical value for $\mu$. Here is where the rescaled equations play their fundamental role since they are blind to the crossing of $\mu_c$ we can integrate the rescaled equations for different values of $\mu$ inputting the same max rescaled time, then converting back to true time we can clearly see where the $\mu_c$ actually happened.

% FIXME: figure file pending — uncomment once true_mu_c.png lands in figures/
% \begin{figure}[h]
%     \centering
%     \includegraphics[width=1\linewidth]{true_mu_c.png}
%     \caption{Enter Caption}
%     \label{fig:placeholder-true-mu-c}
% \end{figure}

This method is very robust because true time is an integrated quantity of the inverse of $M(\tau)$.

We can characterize the distance of the actual value for $\mu_c$ over different realizations of the system looking at the distribution of the distance between the empirical and the theoretical one yielding this histogram. Histogram has been built on 100 different graph realizations due to constraints in computation time.

% FIXME: figure file pending — uncomment once theoretical-vs-actual.png lands in figures/
% \begin{figure}[h]
%     \centering
%     \includegraphics[width=0.75\linewidth]{theoretical-vs-actual.png}
%     \caption{Enter Caption}
%     \label{fig:placeholder-theoretical-vs-actual}
% \end{figure}
```

Notes for the engineer:
- The three figure blocks are commented out wholesale. They are uncommented by the user once the PNGs are dropped into `thesis/figures/`.
- Original `\subsection{Mapping to the Probability Simplex...}` → `\section{...}`. Original `\subsection{Numerical Integration}` → `\section{...}`. Original `\subsection{$\mu < \mu_c$}` (which was nested inside `\subsection{M(t) Scaling...}`) → `\subsection{...}` under the new `\section{M(t) Scaling ...}`. Heading-level depth in `report` class: chapter → section → subsection → subsubsection.
- Duplicate `\label{fig:placeholder}` in the original source was problematic; renamed to `fig:placeholder-true-mu-c` and `fig:placeholder-theoretical-vs-actual` to avoid a LaTeX "multiply-defined label" warning when uncommented later.

- [ ] **Step 2: Verify**

Run:
```bash
head -3 thesis/chapters/02-rescaling.tex
grep -c "FIXME: figure file pending" thesis/chapters/02-rescaling.tex
```

Expected: first line is `\chapter{Dynamical equations rescaling}`. The grep returns `3`.

- [ ] **Step 3: Commit**

```bash
git add thesis/chapters/02-rescaling.tex
git commit -m "feat(thesis): add rescaling chapter (figures commented for first build)"
```

---

## Task 10: Write placeholder `chapters/03-results.tex`

**Files:**
- Create: `thesis/chapters/03-results.tex`

- [ ] **Step 1: Write the file**

Create `thesis/chapters/03-results.tex` with:

```latex
\chapter{Results}

% TODO: numerical methodology and empirical results, scaling behaviors on
% exponential and power-law networks.
```

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/chapters/03-results.tex
```

Expected: exactly the two lines above (chapter heading + TODO comment).

- [ ] **Step 3: Commit**

```bash
git add thesis/chapters/03-results.tex
git commit -m "feat(thesis): add results chapter placeholder"
```

---

## Task 11: Write placeholder `chapters/04-conclusions.tex`

**Files:**
- Create: `thesis/chapters/04-conclusions.tex`

- [ ] **Step 1: Write the file**

Create `thesis/chapters/04-conclusions.tex` with:

```latex
\chapter{Conclusions}

% TODO: summarize implications and outline theoretical extensions
% (multiplicative stochastic noise, etc.).
```

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/chapters/04-conclusions.tex
```

Expected: exactly the two lines above.

- [ ] **Step 3: Commit**

```bash
git add thesis/chapters/04-conclusions.tex
git commit -m "feat(thesis): add conclusions chapter placeholder"
```

---

## Task 12: Write `appendix/A-rescaled-equations.tex`

**Files:**
- Create: `thesis/appendix/A-rescaled-equations.tex`

Original `\section{Rescaled Equations}` (under `\appendix`) becomes `\chapter{Rescaled Equations}`. Inside `\appendix`, `\chapter` is auto-labeled "Appendix A".

- [ ] **Step 1: Write the file**

Create `thesis/appendix/A-rescaled-equations.tex` with:

```latex
\chapter{Rescaled Equations}

Starting from the Generalized Lotka-Volterra equations:
$$
\dot{x}_i = x_i \left(1-x_i + \sum_j A_{ij}\alpha_{ij}x_j\right) = x_i \left(1-x_i +\sum_j W_{ij}x_j\right)
$$
we can perform a change of variables that removes divergences from the trajectories of the system, this is useful to study the system across the critical $\mu$ regime.
We introduce the following rescaled variables:

\begin{equation}
    \left\{
        \begin{aligned}
        M(t) &= \sum_j x_j (t)\\
        y_i(t) &= \frac{x_i(t)}{M(t)} \\
        \end{aligned}
    \right.
\end{equation}
This means we'll be working with new rescaled variables such that $y_i(t) \ge 0, \forall t$ and $\sum_j y_j(t) = 1$.
Now to the dynamical equations:
$$
\frac{dM}{dt}= \sum_j \frac{dx_j}{dt} = \sum_j x_j\left(1-x_j+\sum_jW_{jk}x_k\right)
$$
but $x_j(t) = M(t)y_j(t)$ hence:

\begin{equation}
    \begin{aligned}
    \frac{dM}{dt} &= M \sum_k y_k - M^2 \sum_k y_k^2 + M^2 \sum_{k,j} y_k W_{kj} y_j \\
                  &= M - M^2 y^T y + M^2 y^T Wy
    \end{aligned}
\end{equation}
where we used the fact that $\sum_j y_j(t) = 1$
Now onto the dynamical equation for $y_j(t)$:

\begin{equation}
    \begin{aligned}
        \frac{dy_i}{dt} &= \frac{\frac{dx_i}{dt} M - x_i \frac{dM}{dt}}{M^2} = \frac{1}{M} \frac{dx_i}{dt} - \frac{x_i}{M^2} \frac{dM}{dt} \\
        &= \frac{1}{M} \frac{dx_i}{dt} - \frac{y_i}{M} \frac{dM}{dt} \\
        &= \frac{1}{M} \left[ y_i M \left( 1 - y_i M + \sum_{j=1}^N W_{ij} y_j M \right) \right] - \frac{y_i}{M} \left[ M - M^2 \sum_{k=1}^N y_k^2 + M^2 y^T W y \right] \\
        &= \cancel{y_i} - y_i^2 M + y_i M \sum_{j=1}^N W_{ij} y_j - \cancel{y_i} + y_i M \sum_{k=1}^N y_k^2 - y_i M \  y^T W y \\
        &= M(t) y_i \left[ \sum_{j=1}^N W_{ij} y_j - y^T W y - y_i +  y^T y\right]
    \end{aligned}
\end{equation}

Now the results would still be diverging for $\mu > \mu_c$ due to the $M(t)$ term which multiplies the right-hand side, we then rescale time, introducing:
\begin{equation}
    \frac{d\tau}{dt} = M(t)
\end{equation}

This leads to the final equations:
\begin{equation}
    \left\{
        \begin{aligned}
           \frac{dy_i}{d\tau} &= y_i \left[ \sum_{j=1}^N W_{ij} y_j - y^T W y  -  y_i + y^T y\ \right] \\
           \frac{dM}{d\tau} &= 1 + M \left[ y^T W y - y^T y \right] \\
           \frac{dt}{d\tau} &= \frac{1}{M(\tau)}
        \end{aligned}
    \right.
\end{equation}
We could also obtain a fully vectorial equation also for $y$ by using the element wise multiplication ($\odot$)  and a vector of ones ($\mathbf{1}$):

\begin{equation}
\frac{dy}{d\tau} = y \odot \left[ W y - (y^T W y)\mathbf{1} - y + (y^T y)\mathbf{1} \right]
\end{equation}

Finally we notice that the equation for $y$ is not dependent on $M$, nevertheless we will integrate also $M$ and $t$ in order to be able to reconstruct the original trajectories in true time and abundances from the rescaled solutions.
```

- [ ] **Step 2: Verify**

Run:
```bash
head -3 thesis/appendix/A-rescaled-equations.tex
wc -l thesis/appendix/A-rescaled-equations.tex
```

Expected: first line is `\chapter{Rescaled Equations}`. Line count > 50.

- [ ] **Step 3: Commit**

```bash
git add thesis/appendix/A-rescaled-equations.tex
git commit -m "feat(thesis): add appendix A (rescaled equations derivation)"
```

---

## Task 13: Write `main.tex`

**Files:**
- Create: `thesis/main.tex`

- [ ] **Step 1: Write the file**

Create `thesis/main.tex` with:

```latex
\documentclass[11pt,a4paper,oneside]{report}

\input{preamble}
\input{metadata}

\begin{document}

\maketitle

\begin{abstract}
\abstracttext
\end{abstract}

\tableofcontents

\include{chapters/01-introduction}
\include{chapters/02-rescaling}
\include{chapters/03-results}
\include{chapters/04-conclusions}

\appendix
\include{appendix/A-rescaled-equations}

\printbibliography

\end{document}
```

Notes for the engineer:
- `oneside` matches typical thesis printing for a draft; switch to `twoside` later if needed.
- `\include` (not `\input`) is used for chapters so each starts on a new page and `latexmk` can manage per-chapter aux files.
- `\printbibliography` will render an empty bibliography section header for now (no entries yet). That's expected.

- [ ] **Step 2: Verify**

Run:
```bash
cat thesis/main.tex
```

Expected: exactly the contents above.

- [ ] **Step 3: Commit**

```bash
git add thesis/main.tex
git commit -m "feat(thesis): add main.tex entry point"
```

---

## Task 14: First build

**Files:** none modified; produces `thesis/main.pdf` and aux files (aux files are gitignored).

- [ ] **Step 1: Run latexmk from inside thesis/**

Run:
```bash
cd thesis && latexmk -pdf main.tex
```

Expected: exit code 0. Last lines of output include `Latexmk: All targets (main.pdf) are up-to-date` or `Latexmk: ... fully made`. A `main.pdf` file appears in `thesis/`.

If pdflatex reports a minted error about `pygmentize` not being found, install Pygments (`pip install Pygments` or `uv tool install pygments`) and re-run.

If biber complains about an empty `.bib` (it shouldn't — comment-only files are accepted), the comment in `references.bib` is sufficient. If it does error, add a single trivial entry:
```bibtex
@misc{placeholder, note = {placeholder}}
```
and re-run.

- [ ] **Step 2: Inspect the PDF structure**

Run:
```bash
cd thesis && pdfinfo main.pdf | grep -E "Pages|Title"
```

Expected: `Title` contains "Can a simple model explain economics?" (or empty — hyperref may or may not propagate it depending on options). `Pages` is at least 8 (title + abstract + TOC + 4 chapter pages + appendix + bib).

Run:
```bash
cd thesis && pdftotext main.pdf - | head -40
```

Expected: title, author, "Probably not" (abstract), "Contents" (TOC), then "Chapter 1 Introduction" and the start of the intro prose.

If `pdfinfo`/`pdftotext` are unavailable, just open `main.pdf` manually and confirm the structure visually.

- [ ] **Step 3: Verify TOC contents**

Run:
```bash
cd thesis && pdftotext main.pdf - | sed -n '/^Contents/,/Chapter 1/p'
```

Expected output includes lines for:
- "1 Introduction"
- "2 Dynamical equations rescaling"
- "3 Results"
- "4 Conclusions"
- "A Rescaled Equations"

- [ ] **Step 4: Commit the PDF**

```bash
git add thesis/main.pdf
git commit -m "build(thesis): first successful PDF build"
```

If the build failed, **do not commit a partial PDF**. Fix the underlying error, re-run latexmk, then commit.

---

## Task 15: Sanity-check `latexmk -c` clean

**Files:** none

- [ ] **Step 1: Run clean**

Run:
```bash
cd thesis && latexmk -c
```

Expected: removes aux files (`*.aux`, `*.log`, `*.toc`, `*.fdb_latexmk`, `*.fls`, `*.bbl`, `*.bcf`, `*.run.xml`, `*.synctex.gz`, `_minted-*`). Leaves `main.pdf` and all source `.tex`/`.bib`/`.gitignore`/`.latexmkrc` files alone.

- [ ] **Step 2: Verify with git status**

Run:
```bash
git status thesis/
```

Expected: clean (no modified or untracked files). Aux files that were briefly created during the build are gone after `latexmk -c`, and the ones still on disk are ignored by `.gitignore`.

- [ ] **Step 3: Rebuild once more to confirm reproducibility**

Run:
```bash
cd thesis && latexmk -pdf main.tex
```

Expected: builds again cleanly. `main.pdf` may or may not be byte-identical (timestamps embedded by pdflatex can differ); that's normal.

- [ ] **Step 4: No commit**

Verification only.

---

## Self-Review Notes (already applied to the plan above)

- **Spec coverage:** Every item in the design (`thesis/` layout, class change, preamble packages, latexmkrc, gitignore-with-PDF-committed, modular file split, figure placeholders, minted shell-escape, biblatex+biber, build verification) maps to a task.
- **Placeholder scan:** No "TBD"/"fill in later" content in the plan itself. The TODO comments in placeholder chapters 3/4 are intentional content of those files, not plan placeholders.
- **Type/name consistency:** File paths (`thesis/preamble.tex`, `thesis/metadata.tex`, etc.) are used identically across tasks. Macro names (`\abstracttext`) are defined in Task 6 and referenced in Task 13. Figure labels are unique.
- **Pygments install hint** appears in Task 1 (toolchain check) and again in Task 14 (build) for the engineer who reads tasks out of order.
