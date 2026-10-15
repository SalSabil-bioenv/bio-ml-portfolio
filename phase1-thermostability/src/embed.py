"""ESM-2 mean-pooled embeddings: length-sorted token-budget batching, chunked resumable cache."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

MAX_RES = 1022  # residues; +2 for <cls>/<eos> = 1024-token context
CHUNK = 1000

def load(model_name, device, fp16=False):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()
    if fp16 and device == "cuda":
        model = model.half()
    return tok, model

@torch.inference_mode()
def embed_batch(tok, model, seqs, device):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_RES + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    h = model(**enc).last_hidden_state                    # [B, L, D]
    mask = enc["attention_mask"].clone()                  # [B, L], 1 = real token
    lens = mask.sum(1)
    mask[:, 0] = 0                                        # drop <cls>
    mask[torch.arange(mask.size(0), device=mask.device), lens - 1] = 0   # drop <eos>
    m = mask.unsqueeze(-1).to(h.dtype)
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return pooled.float().cpu().numpy()

def make_batches(lengths, token_budget):
    """Group length-sorted indices so padded tokens per batch <= token_budget."""
    batches, cur, cur_max = [], [], 0
    for i in np.argsort(lengths, kind="stable"):
        L = min(int(lengths[i]), MAX_RES) + 2
        if cur and max(cur_max, L) * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(int(i))
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches

def embed_seqs(tok, model, seqs, device, token_budget):
    lengths = np.array([len(s) for s in seqs])
    out = None
    for b in make_batches(lengths, token_budget):
        e = embed_batch(tok, model, [seqs[i] for i in b], device)
        if out is None:
            out = np.zeros((len(seqs), e.shape[1]), dtype=np.float32)
        out[b] = e
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.model.split("/")[-1]
    df = pd.read_parquet(f"data/processed/{a.split}.parquet")
    seqs = df.seq.tolist()
    lengths = np.array([len(s) for s in seqs])

    out_dir = Path(f"data/embeddings/{tag}")
    cache = out_dir / f"{a.split}_chunks"
    cache.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{a.split}.npz"
    if final.exists():
        print("already done:", final)
        return

    tok, model = load(a.model, device, a.fp16)
    order = np.argsort(lengths, kind="stable")
    chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]
    t0 = time.time()
    for k, idx in enumerate(chunks):
        f = cache / f"chunk{k:03d}.npz"
        if f.exists():          # resume support
            continue
        emb = embed_seqs(tok, model, [seqs[i] for i in idx], device, a.token_budget)
        np.savez(f, idx=idx, emb=emb)
        print(f"[{a.split}] chunk {k + 1}/{len(chunks)} | {time.time() - t0:.0f}s | "
              f"max len {lengths[idx].max()}", flush=True)

    D = np.load(cache / "chunk000.npz")["emb"].shape[1]
    out = np.zeros((len(seqs), D), dtype=np.float32)
    for f in sorted(cache.glob("chunk*.npz")):
        z = np.load(f)
        out[z["idx"]] = z["emb"]
    np.savez(final, emb=out, ids=df.id.values.astype(str))
    print("saved", final, out.shape)

if __name__ == "__main__":
    main()
