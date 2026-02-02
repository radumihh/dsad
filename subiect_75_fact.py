import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

# --- CITIRE DATE ---
# Presupunem fisierele in "dataIN/"
df_ang = pd.read_csv("dataIN/Satisfactie_Angajati.csv")
df_dep = pd.read_csv("dataIN/Departamente.csv")

# Merge
df = pd.merge(df_ang, df_dep, on="AngajatID")

# --- CERINTA 1 ---
# Angajati cu salariu > media
media_salariu = df["Salariu"].mean()
rez1 = df[df["Salariu"] > media_salariu][["Nume", "Departament", "Salariu"]]
rez1.to_csv("Cerinta1.csv", index=False)

# --- CERINTA 2 ---
# Scor mediu satisfactie per angajat (Q1-Q5)
cols_q = ["Q1", "Q2", "Q3", "Q4", "Q5"]
df["ScorMediu"] = df[cols_q].mean(axis=1)

# Medie per departament
grp_dep = df.groupby("Departament")["ScorMediu"].mean()

# Departament cu max
dep_max = grp_dep.idxmax()
val_max = grp_dep.max()

rez2 = pd.DataFrame({
    "Departament": [dep_max],
    "Scor_Mediu_Agregat_Maxim": [val_max]
})
rez2.to_csv("Cerinta2.csv", index=False)

# --- CERINTA 3 (B.1) ---
# Analiza Factoriala
X = df[cols_q].values

# Initializare
fa = FactorAnalyzer(n_factors=min(X.shape[1], X.shape[0]), rotation="varimax")
fa.fit(X)

# Varianta (Eigenvalues)
ev, v = fa.get_eigenvalues()
# Putem lua si varianta explicata din get_factor_variance()
variance_info = fa.get_factor_variance()
# variance_info e un array 3xN (SS Loadings, Prop Var, Cumul Var)

tabel_varianta = pd.DataFrame({
    "Eigenvalues": ev, # Nota: adesea se cer doar primii factori retinuti, dar aici punem toti
    # Pentru varianta explicata dupa rotatie, folosim variance_info
})

# De obicei la examen se cere tabelul formatat specific.
# Vom salva variance_info pentru factorii extrasi
t_var = pd.DataFrame(data=variance_info, 
                     index=["SS Loadings", "Proportion Var", "Cumulative Var"],
                     columns=[f"Factor{i+1}" for i in range(variance_info[0].shape[0])])

t_var.T.to_csv("Varianta.csv")

# --- CERINTA 4 (B.2) ---
# Loadings (Corelatii variabile-factori)
loadings = fa.loadings_
t_loadings = pd.DataFrame(loadings, 
                          index=cols_q, 
                          columns=[f"Factor{i+1}" for i in range(loadings.shape[1])])
t_loadings.to_csv("FactorLoadings.csv")

# --- CERINTA 5 (B.3) ---
# Cercul corelatiilor (F1 vs F2)
plt.figure(figsize=(6, 6))
plt.title("Cercul Corelatiilor (F1 vs F2)")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

# Cerc unitate
cerc = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_artist(cerc)

# Plotare variabile
L = loadings[:, :2] # Luam doar primii 2 factori
for i, var in enumerate(cols_q):
    x_val = L[i, 0]
    y_val = L[i, 1]
    plt.arrow(0, 0, x_val, y_val, head_width=0.05, color='r')
    plt.text(x_val + 0.05, y_val + 0.05, var, color='r')

plt.grid()
plt.show()
