import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def nan_replace_df(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

print("--- SUBIECT 25 ---")

# Citire
buget = pd.read_csv("dataIN/Buget.csv")
pop = pd.read_csv("dataIN/LocPopulation.csv") 
nan_replace_df(buget)

merged = pd.merge(buget, pop, on="Siruta")
# Identificare coloane
pop_col = [c for c in merged.columns if "pop" in c.lower()][0] 
county_col = [c for c in merged.columns if "jud" in c.lower() or "county" in c.lower() or "Judet" in c][0]

incomes = ["V1", "V2", "V3", "V4", "V5"]
expenses = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# === CERINTA 1 ===
# Venituri/Cheltuieli per locuitor
df_res1 = merged[["Siruta"]].copy()
for c in incomes + expenses:
    if c in merged.columns:
        df_res1[f"{c}_per_capita"] = merged[c] / merged[pop_col]

df_res1.to_csv("dataOUT/Requirement_1.csv", index=False)

# === CERINTA 2 ===
# Procentaj cheltuieli din total, pe judet
grp = merged.groupby(county_col)[expenses].sum()
total_exp = grp.sum(axis=1)

# Impartire vectorizata (fiecare coloana la total)
res2 = grp.div(total_exp, axis=0) * 100
res2.to_csv("dataOUT/Requirement_2.csv")

# === DATASET 25 (3, 4, 5) ===
ds25 = pd.read_csv("dataIN/DataSet_25.csv")
nan_replace_df(ds25)

numeric_cols = [c for c in ds25.columns if pd.api.types.is_numeric_dtype(ds25[c])]
X = ds25[numeric_cols].values

# 3. Matricea de covarianta standardizata (= corelatie)
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
cov_mat = np.cov(X_std.T)
pd.DataFrame(cov_mat, index=numeric_cols, columns=numeric_cols).to_csv("dataOUT/StdCov.csv")

# 4. Scoruri PCA
pca = PCA()
pca.fit(X_std)
scores = pca.transform(X_std)
pd.DataFrame(scores, index=ds25.index).to_csv("dataOUT/Scores.csv")

# 5. Scree Plot
eigenvalues = pca.explained_variance_
plt.figure("Scree Plot")
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'ro-', linewidth=2)
plt.axhline(1, color='b', linestyle='--')
plt.title("Scree Plot")
plt.savefig("dataOUT/ScreePlot.png")
