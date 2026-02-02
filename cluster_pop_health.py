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

print("--- SUBIECT 65 ---")

nat_mov = pd.read_csv("dataIN/NatLocMovement.csv")
pop_loc = pd.read_csv("dataIN/PopulationLoc.csv")
nan_replace_df(nat_mov)

merged = pd.merge(nat_mov, pop_loc, on="Siruta")
pop_col = [c for c in merged.columns if "pop" in c.lower()][0]
county_col = [c for c in merged.columns if "jud" in c.lower() or "county" in c.lower() or "Judet" in c][0]

# === CERINTA 1 ===
# Rata spor natural judetean
# Sumam evenimentele si populatia per judet (atentie: pop e per localitate, suma lor = pop judet)
grp = merged.groupby(county_col)[["LiveBirths", "Deceased", pop_col]].sum()

# Rate la 1000 locuitori
grp["BirthRate"] = (grp["LiveBirths"] / grp[pop_col]) * 1000
grp["DeathRate"] = (grp["Deceased"] / grp[pop_col]) * 1000
grp["NatIncRate"] = grp["BirthRate"] - grp["DeathRate"]

grp[["NatIncRate"]].to_csv("dataOUT/Requirement_1.csv")

# === CERINTA 2 ===
# Localitati cu rate maxime per judet
indicators = ["Marriages", "Deceased", "DeceasedUnder1Year", "Divorces", "StillBirths", "LiveBirths"]

# Calculam rate per localitate
for ind in indicators:
    if ind in merged.columns:
        merged[f"{ind}_Rate"] = (merged[ind] / merged[pop_col]) * 1000

# Pentru fiecare judet, gasim localitatea cu maxim pt fiecare indicator
res2 = []
# Iteram prin judete
for county, sub in merged.groupby(county_col):
    row = {county_col: county}
    for ind in indicators:
        col_rate = f"{ind}_Rate"
        if col_rate in sub.columns:
            # Localitatea maxima
            loc_idx = sub[col_rate].idxmax()
            # Numele localitatii (City sau Siruta)
            # Folosim 'City' daca exista
            city_name = sub.loc[loc_idx, "City"] if "City" in sub.columns else sub.loc[loc_idx, "Siruta"]
            row[ind] = city_name
    res2.append(row)

pd.DataFrame(res2).to_csv("dataOUT/Requirement_2.csv", index=False)


# === DATASET 65 (3, 4, 5) ===
ds65 = pd.read_csv("dataIN/DataSet_65.csv")
nan_replace_df(ds65)

vars_cluster = ["POP", "PIB", "CHS_PUB", "CHS_PRIV", "RM", "RN", "SPV"]
available_vars = [v for v in vars_cluster if v in ds65.columns]
X = ds65[available_vars].values

# 3. Standardizare si Ierarhie (Ward, Euclidean)
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
pd.DataFrame(X_std, columns=available_vars, index=ds65.index).to_csv("dataOUT/Xstd.csv")

Z = linkage(X_std, method='ward', metric='euclidean')
print("Hierarchy Matrix:\n", Z)

# 4. Threshold
diff = Z[1:, 2] - Z[:-1, 2]
idx_max = np.argmax(diff)
thresh = (Z[idx_max, 2] + Z[idx_max+1, 2]) / 2
print(f"Max diff step {idx_max+1}, Threshold: {thresh}")

# 5. Dendrograma
plt.figure("Dendrogram")
dendrogram(Z, labels=ds65.index, leaf_rotation=45, color_threshold=thresh)
plt.axhline(thresh, c='r', linestyle='--')
plt.title("Dendrogram (Ward)")
plt.savefig("dataOUT/Dendrogram.png")
