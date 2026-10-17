import sys, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import spearmanr

split = sys.argv[1]
df = pd.read_parquet(f"data/processed/{split}.parquet")
npz = np.load(f"data/embeddings/esm2_t12_35M_UR50D/{split}.npz")
X = npz[npz.files[0]]
y = df.target.values

groups = pd.factorize(df.seq)[0]
cv = GroupKFold(5)

print(f"== MLP vs Ridge: {split} ==")
ridge_preds = cross_val_predict(Ridge(alpha=430.0), X, y, cv=cv, groups=groups)
print(f"Ridge | Spearman: {spearmanr(y, ridge_preds).statistic:.3f}")

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
mlp_preds = cross_val_predict(mlp, X, y, cv=cv, groups=groups)
print(f"MLP   | Spearman: {spearmanr(y, mlp_preds).statistic:.3f}")
