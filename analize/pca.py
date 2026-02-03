# => Analiza Componentelor Principale (PCA)

# Importuri necesare:
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb
import os

OUT_DIR = "grafice_out"
os.makedirs(OUT_DIR, exist_ok=True)

# citim datele
date = pd.read_csv("data_in/Teritorial_2022.csv", index_col=0)
# NAN
for c in date.columns:
    if date[c].isna().any():
        if pd.api.types.is_numeric_dtype(date[c]):
            date[c] = date[c].fillna(date[c].mean())
        else:
            date[c] = date[c].fillna(date[c].mode(dropna=True)[0])

id_cols = list(date.columns[:3])
date_id = date[id_cols]
date_numerice = date.drop(columns=id_cols) # pentru pca avem nevoie doar de coloanele numerice
print(date_numerice)

# standardizam datele
scaler = StandardScaler()
date_standard = scaler.fit_transform(date_numerice)

# aplicam PCA
pca = PCA()
date_pca = pca.fit(date_standard)
componente_principale = pca.components_

# Varianta componente
n = date_standard.shape[0]
varianta = pca.explained_variance_ * (n-1)/n
print("1.Varianta componentelor:\n", varianta)

# Plot varianta componente cu evidentierea criteriilor de relevanta
varianta_explicata = pca.explained_variance_ratio_
varianta_cumulativa = np.cumsum(varianta_explicata)
m = np.sum(varianta > 1)
plt.figure(figsize=(8,6))
plt.bar(range(1, len(varianta)+1), varianta_explicata, alpha=0.7, color="blue", label="Varianta explicata")
plt.plot(range(1, len(varianta)+1), varianta_cumulativa, marker="o", linestyle="--", color="red", label="Varianta cumulativa")
plt.xlabel("Numarul componentelor principale")
plt.ylabel("Varianta explicata")
plt.title("Plot varianta componente")
plt.legend()
plt.grid()
plt.savefig(f"{OUT_DIR}/varianta_componente.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# Calculul corelatii factoriale (corelatii variabile observate - componente)
corelatie_factoriala = componente_principale[:m, :].T * np.sqrt(varianta[:m])
print("3. Corelatii factoriale:\n", corelatie_factoriala)

# Trasare corelograma corelatii factoriale
plt.figure(figsize=(10,8))
sb.heatmap(corelatie_factoriala, annot=True, cmap='coolwarm', center=0,
           xticklabels=[f"PC{i+1}" for i in range(corelatie_factoriala.shape[1])])
plt.title("Corelograma corelatiilor factoriale")
plt.xlabel("Componente principale")
plt.ylabel("Variabile observate")
plt.savefig(f"{OUT_DIR}/corelatii_factoriale.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# Trasare cercul corelatiilor
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

circle = patches.Circle((0,0), 1, fill=False, color="gray")
ax.add_patch(circle)

for i, (x, y) in enumerate(corelatie_factoriala[:, :2]):
    ax.arrow(0,0,x,y, head_width=0.04, head_length=0.04, color="red")
    ax.text(x, y, date_numerice.columns[i], fontsize=10)

ax.axhline(0, color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")
ax.set_title("Cercul corelatiilor")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid()

plt.savefig(f"{OUT_DIR}/cercul_corelatiilor.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# Calcul componente si/sau scoruri
scoruri_componente = pca.transform(date_standard)
alpha = varianta          # la tine varianta = explained_variance_*(n-1)/n  ~ alpha din seminar
C = scoruri_componente    # C din seminar
S = C / np.sqrt(alpha)    # S din seminar (broadcast pe coloane)
print("Componente principale:\n", componente_principale)
print("Scoruri componente:\n", C)

# Trasare plot componente/scoruri
plt.figure(figsize=(8,6))
plt.scatter(C[:,0], C[:,1], alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scoruri - Componente principale")
plt.grid()

plt.savefig(f"{OUT_DIR}/scoruri_pca.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# Calcul cosinusuri
# cos = componente^2 / suma pe fiecare variabila
C2 = C**2
cosinusuri = (C2.T / np.sum(C2, axis=1)).T
print("Cosinusuri:\n", cosinusuri)

# Calcul contributii
contributii = C2 * 100 / np.sum(C2, axis=0)
print("Contributii:\n", contributii)

# Calcul comunalitati
R2 = corelatie_factoriala**2
comunalitati = np.cumsum(R2, axis=1)
df_comm = pd.DataFrame(
    comunalitati,
    index=date_numerice.columns,
    columns=[f"PC{i+1}" for i in range(comunalitati.shape[1])]
)
df_comm = df_comm.replace([np.inf, -np.inf], np.nan).astype(float)
print("Comunalitati:\n", comunalitati)

# Trasare corelograma comunalitati
plt.figure(figsize=(10,8))
sb.heatmap(df_comm, annot=True, cmap="viridis", fmt=".2f")
plt.title("Comunalitati")

plt.savefig(f"{OUT_DIR}/comunalitati.png", dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()
