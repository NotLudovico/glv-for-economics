"""Criticality–MSB numerical campaign: build big parameter-sweep datasets once, analyze many ways.

Layout:
  config.py      — simulation settings + parameter grid (edit here)
  core.py        — disordered-GLV machinery (build, integrate w/ LSODA+event, locate sigma_c)
  simulate.py    — dataset builder (sweep -> one npz per run, resumable)
  dataset.py     — loader / manifest over the npz runs
  observables.py — analysis toolkit (Moran size, beta, growth dist, tails, extinction)
  analyze_*.py   — analysis scripts producing figures (read the dataset, never re-simulate)
  data/, figures/ — outputs (created by the scripts)
"""
