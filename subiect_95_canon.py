import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import seaborn as sns

# --- CITIRE DATE ---
df_med = pd.read_csv("dataIN/Date_Medicale.csv")
df_sport = pd.read_csv("dataIN/Date_Sportive.csv")
df = pd.merge(df_med, df_sport, on="PacientID")

# --- CERINTA 1 ---
# Atleti: Viteza > 12 si Rezistenta > 40
res1 = df[(df["Viteza_Alergare"] > 12) & (df["Rezistenta"] > 40)][["PacientID", "Viteza_Alergare", "Rezistenta"]]
res1.to_csv("Atleti.csv", index=False)

# --- CERINTA 2 ---
# Corelatie BMI vs Viteza
corr = df["BMI"].corr(df["Viteza_Alergare"])
print(f"Corelatie BMI - Viteza: {corr:.4f}")

# --- CERINTA 3 (B) ---
# Analiza Canonica
vars_X = ["BMI", "Tensiune", "Puls", "Colesterol"]
vars_Y = ["Viteza_Alergare", "Rezistenta", "Forta"]

X = df[vars_X]
Y = df[vars_Y]

# Standardizare (Important pt CCA)
X_std = (X - X.mean()) / X.std()
Y_std = (Y - Y.mean()) / Y.std()

# Model
n_components = min(len(vars_X), len(vars_Y))
cca = CCA(n_components=n_components)
cca.fit(X_std, Y_std)

# 1. Calcul Scoruri (X_c, Y_c)
X_c, Y_c = cca.transform(X_std, Y_std)
# X_c sunt scorurile canonice pt setul X (variabilele U)
# Y_c sunt scorurile canonice pt setul Y (variabilele V)

# Salvam primele 2 componente
res_cca = pd.DataFrame({
    "U1": X_c[:, 0], "U2": X_c[:, 1],
    "V1": Y_c[:, 0], "V2": Y_c[:, 1]
})
res_cca.to_csv("ScoruriCanonice.csv", index=False)

# 2. Corelatii Canonice
# Sklearn nu da direct corelatiile canonice ca atribut simplu, trebuie calculate 
# ca si corelatia dintre scorurile pereche U_i si V_i
canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
print("Corelatii Canonice:", np.round(canonical_corrs, 4))

# 3. Plot (Biplot sau Cerc)
# Facem un plot simplu al primei perechi de variabile canonice (U1 vs V1)
plt.figure(figsize=(6, 6))
plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.7)
plt.title(f"Corelatie Canonica 1 (r={canonical_corrs[0]:.2f})")
plt.xlabel("Variabila Canonica U1 (din X)")
plt.ylabel("Variabila Canonica V1 (din Y)")
plt.grid()

# Adaugam linie de regresie pt vizualizare
m, b = np.polyfit(X_c[:, 0], Y_c[:, 0], 1)
plt.plot(X_c[:, 0], m*X_c[:, 0] + b, color='red')

plt.show()
