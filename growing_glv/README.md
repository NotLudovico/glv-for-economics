# growing_glv

Relative-GLV explorations and the thesis figure pipeline (mean-degree + DMFT).

> **The curated, authoritative relative-GLV model is `../../relative-glv`**
> (`relative_glv/model.py`). The **own-degree** model (`α_ij = μ/k_i + (σ/√k_i) z_ij`,
> `kind="powerlaw_owndeg"`) and the multiscaling results live there, **not here**.

This folder is the older working version, kept because it still generates several thesis
figures (see `../thesis/FIGURES.md` for the full map):

- `ch4_data.py` + `ch4_plots.py` — the **mean-degree** main-text figures
  (`growing_stationary_msb`, `growing_meanfield`, `growing_trajectories`,
  `growing_growth_churn`, `growing_beta_extrapolation`), written straight into `../thesis/`.
- `dmft_phase.py`, `dmft_validate.py` — the DMFT figures (`dmft_phase`, `dmft_validation`).
- `phase_diagram.py` / `phase_space_plot.py` — the phase diagram.

`explore.py` builds the **mean-degree** coupling only (`σ/√C_eff`); `hdmft_*.py` are
degree-resolved *analyses* of that same mean-degree model — neither builds the own-degree
coupling. For own-degree, use `../../relative-glv`.

Exploratory one-offs are archived in `_attic/`.
