import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

split = sys.argv[1] if len(sys.argv) > 1 else "human_cell"
tag = "esm2_t12_35M_UR50D"
df = pd.read_parquet(f"data/processed/{split}.parquet")
z = np.load(f"data/embeddings/{tag}/{split}.npz", allow_pickle=True)
assert (z["ids"] == df.id.values.astype(str)).all(), "row misalignment"
X, y = z["emb"], df.target.values
tr, te = (df.subset == "train").values, (df.subset == "test").values

model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12)))
model.fit(X[tr], y[tr])
p = model.predict(X[te])
print(f"ESM2-35M + Ridge | {split} test: spearman={spearmanr(y[te], p).statistic:.3f} "
      f"rmse={np.sqrt(((y[te] - p) ** 2).mean()):.2f} degC | "
      f"alpha={model[-1].alpha_:.2g}")
