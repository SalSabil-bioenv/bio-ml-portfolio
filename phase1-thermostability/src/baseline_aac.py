import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA = "ACDEFGHIKLMNPQRSTVWY"


def featurize(seqs):
    """20-d amino-acid composition + log10(length)."""
    X = np.zeros((len(seqs), 21), dtype=np.float32)
    for i, s in enumerate(seqs):
        c, L = Counter(s), max(len(s), 1)
        for j, a in enumerate(AA):
            X[i, j] = c.get(a, 0) / L
        X[i, 20] = np.log10(L)
    return X


def metrics(y, p):
    rho = float("nan") if np.std(p) == 0 else float(spearmanr(y, p).statistic)
    return {"spearman": rho,
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p))}


MODELS = {
    "mean": lambda: DummyRegressor(strategy="mean"),
    "ridge_aac": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "rf_aac": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
}

rows = []
for split in ["mixed_split", "human", "human_cell"]:
    df = pd.read_parquet(f"data/processed/{split}.parquet")
    parts = {s: df[df.subset == s] for s in ["train", "val", "test"]}
    Xtr, ytr = featurize(parts["train"].seq.tolist()), parts["train"].target.values
    for mname, make in MODELS.items():
        model = make().fit(Xtr, ytr)
        for ev in ["val", "test"]:
            part = parts[ev]
            if part.empty:
                continue
            m = metrics(part.target.values, model.predict(featurize(part.seq.tolist())))
            rows.append({"split": split, "model": mname, "eval": ev, **m})
            print(split, mname, ev, {k: round(v, 3) for k, v in m.items()})

res = pd.DataFrame(rows).round(3)
Path("results").mkdir(exist_ok=True)
res.to_json("results/day01_baselines.json", orient="records", indent=2)
Path("results/day01_baselines.md").write_text(
    "# Day 1 baselines (AA composition + log length)\n\n```\n" + res.to_string(index=False) + "\n```\n"
)
print("\n" + res.to_string(index=False))
