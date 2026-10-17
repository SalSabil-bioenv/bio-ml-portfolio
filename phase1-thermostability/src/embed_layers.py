"""Per-layer masked-mean ESM-2 embeddings -> data/embeddings/<tag>/<split>_layers.npz"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Suppress sm_50 CUDA capability warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

sys.path.insert(0, "src")
# Removed make_batches to avoid signature conflicts
from embed import MAX_RES, load

@torch.inference_mode()
def embed_batch_layers(tok, model, seqs, device, layers):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_RES + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    hs = model(**enc, output_hidden_states=True).hidden_states   
    
    mask = enc["attention_mask"].clone()
    lens = mask.sum(1)
    mask[:, 0] = 0
    mask[torch.arange(mask.size(0), device=mask.device), lens - 1] = 0
    
    m = mask.unsqueeze(-1).to(hs[0].dtype)
    denom = m.sum(1).clamp(min=1)
    
    pooled = [((hs[l] * m).sum(1) / denom).float().cpu().numpy() for l in layers]
    return np.stack(pooled, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Explicitly force CPU to avoid GTX 750 Ti compatibility crashes
    device = torch.device("cpu")
    
    model_name = "facebook/esm2_t12_35M_UR50D"
    tok, model = load(model_name, device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    layers = list(range(1, num_layers + 1))

    tag = model_name.split("/")[-1]
    out_dir = Path(f"data/embeddings/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_layers.npz"

    if out_file.exists():
        print(f"already done: {out_file}")
        return

    df = pd.read_parquet(f"data/processed/{args.split}.parquet")
    seqs = df["seq"].tolist()

    all_embs = []
    t0 = time.time()

    # Direct linear batching
    for i in range(0, len(seqs), args.batch_size):
        batch = seqs[i:i + args.batch_size]
        batch_embs = embed_batch_layers(tok, model, batch, device, layers)
        all_embs.append(batch_embs)
        
        # Log progress to the nohup file
        if (i // args.batch_size) % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(seqs)} sequences...")

    final_embs = np.concatenate(all_embs, axis=0)
    np.savez_compressed(out_file, embeddings=final_embs)
    print(f"Processed {len(seqs)} sequences -> {final_embs.shape} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()