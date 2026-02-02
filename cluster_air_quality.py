import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def nan_replace_df(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

print("--- SUBIECT 35 ---")

# Citire
df = pd.read_csv("dataIN/AirQualityCountries.csv")
codes = pd.read_csv("dataIN/CountryCodes.csv")
nan_replace_df(df)

indicators = [c for c in df.columns if "Air_quality" in c]

# === CERINTA 1 ===
# Tara cu valoare maxima per indicator
res1 = []
for col in indicators:
    # idxmax returneaza indexul valorii maxime
    max_country = df.loc[df[col].idxmax(), "Country"]
    res1.append({"Indicator": col, "Country": max_country})

pd.DataFrame(res1).to_csv("dataOUT/Requirement_1.csv", index=False)

# === CERINTA 2 ===
# Tara cu valoare maxima per indicator pe continent
df_merged = pd.merge(df, codes, on="CountryID")

res2 = []
grouped = df_merged.groupby("Continent")
for continent, grp in grouped:
    # Pentru fiecare continent, gasim tara maxima la fiecare indicator
    row = {"Continent": continent}
    for col in indicators:
        row[col] = grp.loc[grp[col].idxmax(), "Country"]
    res2.append(row)

pd.DataFrame(res2).to_csv("dataOUT/Requirement_2.csv", index=False)

# === CLUSTERING (3, 4, 5, 6) ===
# Clusterizare variabile -> Transpusa
X = df[indicators].values.T
labels = indicators

# 3. Ierarhie (Average, Correlation)
Z = linkage(X, method='average', metric='correlation')
print("Hierarchy Matrix:\n", Z)

# 4. Threshold si diferenta maxima
diff = Z[1:, 2] - Z[:-1, 2]
idx_max = np.argmax(diff)
thresh = (Z[idx_max, 2] + Z[idx_max+1, 2]) / 2
print(f"Max diff at step {idx_max+1}, Threshold: {thresh}")

# 5. Dendrograma
plt.figure("Dendrogram")
dendrogram(Z, labels=labels, leaf_rotation=45)
plt.title("Dendrogram")
plt.savefig("dataOUT/Dendrogram.png")

# 6. Partitie optima
groups = fcluster(Z, t=thresh, criterion='distance')
res6 = pd.DataFrame({"Indicator": labels, "Cluster": groups})
res6.to_csv("dataOUT/OptPart.csv", index=False)
