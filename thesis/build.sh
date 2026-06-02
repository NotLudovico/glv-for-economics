#!/usr/bin/env bash
# Compile the thesis. minted needs -shell-escape + pygmentize (lives in ../.venv/bin).
set -euo pipefail
cd "$(dirname "$0")"
PATH="$(cd ../.venv/bin && pwd):$PATH" latexmk -pdf -shell-escape "$@" main.tex
