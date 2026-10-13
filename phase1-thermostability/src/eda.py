from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROC, RES = Path("data/processed"), Path("results")
RES.mkdir(exist_ok=True)
ESM_MAX = 1022  # ESM-2 context minus <cls>/<eos>

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, name in enumerate(["mixed_split", "human", "human_cell"]):
    df = pd.read_parquet(PROC / f"{name}.parquet")
    g = df.groupby("subset")
    summary = pd.DataFrame({
        "n": g.size(),
        "tm_mean": g.target.mean(), "tm_std": g.target.std(),
        "tm_min": g.target.min(), "tm_max": g.target.max(),
        "len_med": g.length.median(), "len_p95": g.length.quantile(0.95),
        "len_max": g.length.max(),
        "pct_gt1022": g.length.apply(lambda s: 100 * (s > ESM_MAX).mean()),
    }).round(2)
    print(f"\n== {name}\n{summary.to_string()}")

    tr, te = df[df.subset == "train"], df[df.subset == "test"]
    print(f"exact train/test sequence overlap: {len(set(tr.seq) & set(te.seq))} | "
          f"duplicate seqs within train: {tr.seq.duplicated().sum()}")

    for s, c in [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]:
        sub = df[df.subset == s]
        if len(sub):
            axes[0, j].hist(sub.target, bins=50, alpha=0.5, density=True, label=s, color=c)
            axes[1, j].hist(sub.length.clip(upper=2000), bins=50, alpha=0.5,
                            density=True, label=s, color=c)
    axes[0, j].set(title=f"{name}: Tm (°C)")
    axes[1, j].set(title=f"{name}: length (clipped at 2000)")
    axes[0, j].legend()

plt.tight_layout()
plt.savefig(RES / "day01_eda.png", dpi=150)
print("\nsaved results/day01_eda.png")
