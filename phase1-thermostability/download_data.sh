#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
for f in mixed_split human human_cell; do
  wget -nc -q --show-progress -P data/raw \
    "http://data.bioembeddings.com/public/FLIP/fasta/meltome/${f}.fasta"
done
