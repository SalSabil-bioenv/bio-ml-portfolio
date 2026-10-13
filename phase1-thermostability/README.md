# Phase 1: Protein Thermostability Predictor (FLIP meltome)
**Data:** FLIP meltome FASTA (`mixed_split`, `human`, `human_cell`); re-download with `./download_data.sh` (raw/processed data is gitignored).
**Pipeline:** `src/flip_data.py` (parse) → `src/eda.py` → `src/baseline_aac.py`.
**Baselines:** see `results/day01_baselines.md`.
**Next:** ESM-2 embeddings (cached) + Ridge/MLP head.
