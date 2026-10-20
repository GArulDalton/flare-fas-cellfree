#!/usr/bin/env bash
# Reproduce every figure and table of the paper from scratch.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/diagrams.py          # Figs. 1-2
if [[ "${RETRAIN:-0}" == "1" ]]; then
  python scripts/train.py           # ~12 min on one CPU core; overwrites weights/
fi
python scripts/evaluate.py          # Figs. 3-9, Tables 2-4 (~5 min)
python scripts/overhead.py          # Table 5, Fig. 10 (~2 min)
echo "Done. Figures in figures/, numbers in results/."
