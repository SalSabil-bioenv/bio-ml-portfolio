import argparse, torch, os
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True)
parser.add_argument("--token-budget", type=int, default=16384)
args = parser.parse_args()

print(f"Loading {args.split}...")
df = pd.read_parquet(f"data/processed/{args.split}.parquet")
model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval().cuda()

layers_to_extract = [2, 4, 6, 8, 10, 12]
all_embeddings = []

for i, seq in enumerate(df.seq.values):
    inputs = tokenizer(seq[:1022], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    seq_len = inputs["input_ids"].shape[1]
    layer_reps = []
    for layer_idx in layers_to_extract:
        rep = outputs.hidden_states[layer_idx][0, 1:seq_len-1, :].mean(dim=0).cpu().numpy()
        layer_reps.append(rep)
        
    all_embeddings.append(np.stack(layer_reps))
    if i > 0 and i % 1000 == 0:
        print(f"Processed {i}/{len(df)} sequences")

out_matrix = np.stack(all_embeddings)
out_dir = f"data/embeddings/esm2_t12_35M_UR50D"
os.makedirs(out_dir, exist_ok=True)
np.savez(f"{out_dir}/{args.split}_layers.npz", embeddings=out_matrix)
print(f"saved {out_dir}/{args.split}_layers.npz {out_matrix.shape}")
