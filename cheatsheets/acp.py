import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURARE ---
FILE_PATH = "Teritorial_2022.csv" # Modifica aici
DECIMAL_PLACES = 4

# --- FUNCTII AUXILIARE (Din Seminare) ---
def nan_replace(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

def tabelare_varianta(alpha):
    m = len(alpha)
    t = pd.DataFrame(index=["C"+str(i+1) for i in range(m)])
    t["Varianta"] = alpha
    t["Varianta cumulata"] = np.cumsum(alpha)
    varianta_totala = sum(alpha)
    t["Procent varianta"] = alpha*100/varianta_totala
    t["Procent cumulat"] = np.cumsum(t["Procent varianta"])
    return t

def plot_varianta(alpha, procent_minimal=80):
    m = len(alpha)
    x = np.arange(1, m+1)
    plt.figure("Plot Varianta", figsize=(8,5))
    plt.title("Plot varianta componente", color="b", fontsize=16)
    plt.plot(x, alpha, 'o-', label="Varianta")
    plt.xlabel("Componente")
    plt.ylabel("Varianta")
    plt.xticks(x)
    
    # Criteriul Kaiser
    plt.axhline(1, c="g", label="Criteriul Kaiser")
    
    # Criteriul Acoperirii Minimale
    procent_cumulat = np.cumsum(alpha*100/np.sum(alpha))
    k2_indices = np.where(procent_cumulat > procent_minimal)[0]
    if len(k2_indices) > 0:
        k2 = k2_indices[0] + 1
        plt.axhline(alpha[k2-1], c="m", label=f"Acoperire minimala ({procent_minimal}%)")
    
    # Criteriul Cattell (Scree Plot / Elbow)
    if m > 2:
        eps = alpha[:m-1] - alpha[1:]
        sigma = eps[:m-2] - eps[1:]
        negative_sigma = np.where(sigma < 0)[0]
        if len(negative_sigma) > 0:
            k3 = negative_sigma[0] + 2
            plt.axhline(alpha[k3-1], c="c", label="Criteriul Cattell")

    plt.legend()

def show():
    plt.show()

# --- MAIN ---
# 1. Citire si prelucrare
t = pd.read_csv(FILE_PATH, index_col=0)
nan_replace(t)

# Identificare automata variabile numerice
vars = list(t.select_dtypes(include=[np.number]).columns)
x = t[vars].values
obs = t.index
n, m = x.shape

# Standardizare
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# 2. Modelare ACP
pca = PCA()
pca.fit(x_std)
alpha = pca.explained_variance_

# 3. Rezultate
# Tabel varianta (MANDATORY)
t_var = tabelare_varianta(alpha)
print(t_var)
t_var.to_csv("Varianta.csv")

# Scoruri (Componente)
c = pca.transform(x_std)
t_scoruri = pd.DataFrame(c, index=obs, columns=["C"+str(i+1) for i in range(m)])
t_scoruri.to_csv("Scoruri.csv")

# Corelatii factoriale (Incarcaturi)
r_xc = pca.components_.T * np.sqrt(alpha)
t_rxc = pd.DataFrame(r_xc, index=vars, columns=["C"+str(i+1) for i in range(m)])
t_rxc.to_csv("Corelatii_Factoriale.csv")

# Comunalitati
comm = np.cumsum(r_xc**2, axis=1)
t_comm = pd.DataFrame(comm, index=vars, columns=["C"+str(i+1) for i in range(m)])
t_comm.to_csv("Comunalitati.csv")

# Contributii
contributii = (c**2) / (n * alpha)
t_contrib = pd.DataFrame(contributii, index=obs, columns=["C"+str(i+1) for i in range(m)])
t_contrib.to_csv("Contributii.csv")

# 4. Grafice
plot_varianta(alpha)

# Cercul corelatiilor (C1-C2)
plt.figure("Cercul Corelatiilor", figsize=(7,7))
plt.title("Cercul Corelatiilor (C1-C2)")
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--')
plt.axhline(0, color='k'); plt.axvline(0, color='k')
plt.scatter(r_xc[:, 0], r_xc[:, 1])
for i, txt in enumerate(vars):
    plt.text(r_xc[i, 0], r_xc[i, 1], txt)
plt.xlabel("C1"); plt.ylabel("C2")

# Plot Scoruri
plt.figure("Plot Scoruri")
plt.scatter(c[:, 0], c[:, 1])
for i, txt in enumerate(obs):
    plt.text(c[i, 0], c[i, 1], txt)
plt.xlabel("C1"); plt.ylabel("C2")

# Corelograma
plt.figure("Corelograma")
sb.heatmap(t_rxc, vmin=-1, vmax=1, cmap="RdYlBu", annot=True)

show()
