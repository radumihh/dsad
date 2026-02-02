import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# --- CITIRE DATE ---
df_cli = pd.read_csv("dataIN/Clienti_Banca.csv")
df_suc = pd.read_csv("dataIN/Sucursale.csv")
df = pd.merge(df_cli, df_suc, on="ClientID")

# --- CERINTA 1 ---
# Risc ridicat: Datorie > 20000 si Score < 600
mask = (df["Datorie"] > 20000) & (df["ScoredCredit"] < 600)
rez1 = df[mask][["ClientID", "VenitAnual", "Datorie"]]
rez1.to_csv("RiscRidicat.csv", index=False)

# --- CERINTA 2 ---
# Venit mediu pe sucursala
grp = df.groupby("Sucursala")["VenitAnual"].mean().reset_index()
grp.columns = ["Sucursala", "VenitMediu"]
grp.to_csv("VenitMediuSucursala.csv", index=False)

# --- CERINTA 3 (B.1, B.2, B.3) ---
# Analiza Discriminanta
vars_indep = ["VenitAnual", "Datorie", "ScoredCredit"]
X = df[vars_indep].values
y = df["Default"].values

# Split (optional, dar recomandat pt validare)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# 1. Calcul scoruri (pe tot setul sau test - aici facem pe test pt predictie, 
# dar subiectul cere adesea scoruri pt instante).
# transform returneaza proiectia pe axele discriminante
scoruri = model.transform(X) 
# LDA cu k clase are k-1 axe. Pt 2 clase (Default 0/1) avem 1 axa.
df_scoruri = pd.DataFrame(scoruri, columns=[f"LD{i+1}" for i in range(scoruri.shape[1])])
df_scoruri.to_csv("ScoruriDiscriminante.csv", index=False)

# 2. Matrice Confuzie & Acuratete (pe Test)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
acc = accuracy_score(y_test, pred)

print("Matrice de Confuzie:")
print(cm)
print(f"Acuratete: {acc:.4f}")

# 3. Plot Scoruri (Histograma pt singura axa discriminanta daca sunt 2 clase)
# Separam scorurile pe clase
plt.figure(figsize=(8, 5))
# Scorurile sunt array (n_samples, n_components)
scores_0 = model.transform(X[y == 0])
scores_1 = model.transform(X[y == 1])

plt.hist(scores_0, alpha=0.5, label="Non-Default (0)", bins=20)
plt.hist(scores_1, alpha=0.5, label="Default (1)", bins=20)
plt.title("Distributia Scorurilor Discriminante")
plt.xlabel("LD1")
plt.legend()
plt.show()
