# _attic — archived exploration scripts

One-off exploration and superseded variant scripts from the relative-GLV work,
set aside to keep the main `growing_glv/` listing focused on the library core
(`explore.py`, `measure_msb.py`, `dmft_solver.py`) and the canonical drivers
that feed the thesis (`phase_*`, `dmft_*`, `ch4_*`, `moran_protocol.py`).

Nothing here is imported by the live code. Kept for reference / reproducibility.

Their `sys.path` bootstrap was repointed to the parent dir, so they still import
`explore` etc. and run in place, e.g.:

    uv run python growing_glv/_attic/persistence_scan.py

Clusters: `persistence_*` (regime persistence), `robustness*` (dt/N robustness),
`beta_*` (beta scaling/shape), `find_*` (regime search), plus assorted
single-shot diagnostics and scans.
