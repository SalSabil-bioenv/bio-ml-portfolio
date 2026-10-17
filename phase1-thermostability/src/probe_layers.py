import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import spearmanr

split = sys.argv[1]
df = pd.read_parquet(f"data/processed/{split}.parquet")
npz = np.load(f"data/embeddings/esm2_t12_35M_UR50D/{split}_layers.npz")
X_all = npz[npz.files[0]]
y = df.target.values

# GroupKFold applied to prevent duplicate sequence leakage
groups = pd.factorize(df.seq)[0]
cv = GroupKFold(5)

print(f"== Layer Sweep: {split} ==")
layer_indices = [2, 4, 6, 8, 10, 12]
best_rho, best_layer = -1, -1

for i, layer_name in enumerate(layer_indices):
    X = X_all[:, i, :]
    preds = cross_val_predict(Ridge(alpha=430.0), X, y, cv=cv, groups=groups)
    rho = spearmanr(y, preds).statistic
    print(f"Layer {layer_name:2d} | Spearman: {rho:.3f}")
    if rho > best_rho:
        best_rho, best_layer = rho, layer_name
        
print(f"Selected Layer: {best_layer}")
