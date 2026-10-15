import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from embed import MAX_RES, embed_batch, embed_seqs, load

MODEL = "facebook/esm2_t12_35M_UR50D"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch threads:", torch.get_num_threads())
tok, model = load(MODEL, device)
assert tok.padding_side == "right", "pooling logic assumes right padding"

df = pd.read_parquet("data/processed/human_cell.parquet")
rng = np.random.default_rng(0)
sample = df.iloc[rng.choice(len(df), 200, replace=False)].seq.tolist()

# Check 1: padding / batch invariance
probe = sorted(sample, key=len)[::25]
batched = embed_batch(tok, model, probe, device)
single = np.vstack([embed_batch(tok, model, [s], device) for s in probe])
d1 = float(np.abs(batched - single).max())
print(f"[1] batch-invariance max|diff| = {d1:.2e}")
assert np.allclose(batched, single, atol=1e-3)

# Check 2: manual pooling reference
enc = tok([probe[0]], return_tensors="pt").to(device)
with torch.inference_mode():
    h = model(**enc).last_hidden_state[0]
ref = h[1:-1].mean(0).cpu().numpy()
d2 = float(np.abs(ref - single[0]).max())
print(f"[2] manual-pooling max|diff| = {d2:.2e}")
assert np.allclose(ref, single[0], atol=1e-5)

# Check 3: truncation path
e = embed_batch(tok, model, ["ACDEFGHIKLMNPQRSTVWY" * 75], device)
print("[3] truncation ok, embedding shape:", e.shape)

# Benchmark
embed_seqs(tok, model, sample[:8], device, 4096)
t0 = time.time()
embed_seqs(tok, model, sample, device, 8192)
dt = time.time() - t0
n_tok = sum(min(len(s), MAX_RES) + 2 for s in sample)
rate = n_tok / dt
report = {"device": device, "model": MODEL, "seq_per_s": round(len(sample) / dt, 2),
          "tok_per_s": round(rate), "projected_min": {}}
for split in ["human_cell", "human", "mixed_split"]:
    d = pd.read_parquet(f"data/processed/{split}.parquet")
    toks = float((d.length.clip(upper=MAX_RES) + 2).sum())
    report["projected_min"][split] = round(toks / rate / 60, 1)
print(json.dumps(report, indent=2))
Path("results").mkdir(exist_ok=True)
Path("results/day02_benchmark.json").write_text(json.dumps(report, indent=2))
