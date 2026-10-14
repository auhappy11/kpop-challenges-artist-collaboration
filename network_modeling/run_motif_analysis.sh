#!/usr/bin/env bash
# Compile 02_motif_analysis.c and run the three-node motif analysis on every
# yearly network (2022, 2023, 2024) built by 01_network_construction.py.
# Requires igraph (C library, 0.10.x), e.g. `brew install igraph`.
# Results: network_modeling/networks/motif_results/fanmod_motif_analysis_<year>.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NET="$HERE/networks"
OUT="$NET/motif_results"
mkdir -p "$OUT"
# Set IGRAPH_PREFIX to override; otherwise try the usual install locations.
for PREFIX in ${IGRAPH_PREFIX:-} /opt/homebrew /usr/local /usr; do
  if cc -O2 -w -o "$OUT/motif_analysis" "$HERE/02_motif_analysis.c" \
        -I"$PREFIX/include" -L"$PREFIX/lib" -ligraph -lm 2>/dev/null; then
    break
  fi
done
[ -x "$OUT/motif_analysis" ] || { echo "Could not compile: igraph not found (set IGRAPH_PREFIX)"; exit 1; }
cd "$OUT"   # the program writes its results to the working directory
for year in 2022 2023 2024; do
  ./motif_analysis "$NET/$year/edges_$year.txt" "$year" > "log_$year.txt"
  echo "done: $year"
done
