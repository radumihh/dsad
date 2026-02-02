import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Functie inlocuire NaN (Seminar)
def nan_replace_df(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

print("--- SUBIECT 15 ---")

# Citire date
df = pd.read_csv("dataIN/GlobalIndicatorsPerCapita.csv")
countries = pd.read_csv("dataIN/ContriesIDs.csv") 
nan_replace_df(df)

# === CERINTA 1 ===
# Valoare adaugata = suma sectoarelor
sectors = ["AgrHuntForFish", "Construction", "Manufacturing", "MiningManUt", "TradeT", "TransportComm", "Other"]
df["ValueAdded"] = df[sectors].sum(axis=1)

req1 = df[["CountryID", "Country", "ValueAdded"]]
req1.to_csv("dataOUT/Requirement_1.csv", index=False)

# === CERINTA 2 ===
# Coeficienti de variatie la nivel de continent
df_merged = pd.merge(df, countries, on="CountryID")

# Identificare indicatori (de la GNI la final)
indicators = df.columns[df.columns.get_loc("GNI"):].tolist()
indicators = [c for c in indicators if pd.api.types.is_numeric_dtype(df[c])]

# Calcul CV per grup (Continent)
req2 = df_merged.groupby("Continent")[indicators].apply(lambda x: x.std() / x.mean())
req2.to_csv("dataOUT/Requirement_2.csv")

# === ACP (3, 4, 5) ===
X = df[indicators].values
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

pca = PCA()
pca.fit(X_std)

# 3. Varianta
print("Variances:", pca.explained_variance_)

# 4. Scoruri
scores = pca.transform(X_std)
df_scores = pd.DataFrame(scores, columns=["C"+str(i+1) for i in range(scores.shape[1])], index=df.index)
df_scores.to_csv("dataOUT/Scores.csv")

# 5. Grafic
plt.figure("PCA Scores")
plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel("C1")
plt.ylabel("C2")
plt.title("PCA Scores")
plt.savefig("dataOUT/PCA_Scores.png")
