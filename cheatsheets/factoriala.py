import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# --- CONFIGURARE ---
FILE_PATH = "Teritorial_2022.csv"

# --- FUNCTII AUXILIARE ---
def nan_replace(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

def tabelare_varianta_factori(ev):
    t = pd.DataFrame(index=["F"+str(i+1) for i in range(len(ev))])
    t["Varianta"] = ev
    t["Varianta cumulata"] = np.cumsum(ev)
    t["Procent varianta"] = ev * 100 / sum(ev)
    t["Procent cumulat"] = np.cumsum(t["Procent varianta"])
    return t

def show():
    plt.show()

# --- MAIN ---
t = pd.read_csv(FILE_PATH, index_col=0)
nan_replace(t)

vars = list(t.select_dtypes(include=[np.number]).columns)
x = t[vars].values
obs = t.index
n, m = x.shape

# 1. Teste Factorabilitate (MANDATORY)
chi_square_value, p_value = calculate_bartlett_sphericity(t[vars])
print(f"Bartlett: Chi2={chi_square_value:.2f}, p={p_value:.4f}")

kmo_all, kmo_model = calculate_kmo(t[vars])
print(f"KMO Model: {kmo_model:.3f}")

# 2. Analiza Factoriala (Initiala - pentru varianta)
fa = FactorAnalyzer(n_factors=m, rotation=None)
fa.fit(x)
ev, v = fa.get_eigenvalues()

# Tabel Varianta
t_var = tabelare_varianta_factori(ev)
print(t_var)
t_var.to_csv("Varianta_FA.csv")

# 3. Analiza Factoriala (Cu rotatie)
# Determinare numar factori (Kaiser)
k = sum(ev > 1)
print(f"Numar factori (Kaiser): {k}")

fa_opt = FactorAnalyzer(n_factors=k, rotation="varimax")
fa_opt.fit(x)

# Loadings
l = fa_opt.loadings_
t_loadings = pd.DataFrame(l, index=vars, columns=["F"+str(i+1) for i in range(k)])
t_loadings.to_csv("Loadings.csv")

# Comunalitati
comm = fa_opt.get_communalities()
t_comm = pd.DataFrame(comm, index=vars, columns=["Comunalitati"])
t_comm.to_csv("Comunalitati.csv")

# Scoruri
scoruri = fa_opt.transform(x)
t_scoruri = pd.DataFrame(scoruri, index=obs, columns=["F"+str(i+1) for i in range(k)])
t_scoruri.to_csv("Scoruri_FA.csv")

# 4. Grafice
# Scree Plot
plt.figure("Scree Plot")
plt.plot(range(1, m+1), ev, 'bo-')
plt.axhline(1, c='r', ls='--')
plt.title("Scree Plot")

# Corelograma Loadings
plt.figure("Corelograma Factori")
sb.heatmap(t_loadings, cmap="RdYlBu", vmin=-1, vmax=1, annot=True)

# Plot Scoruri
if k >= 2:
    plt.figure("Plot Scoruri F1-F2")
    plt.scatter(scoruri[:,0], scoruri[:,1])
    for i, txt in enumerate(obs):
        plt.text(scoruri[i,0], scoruri[i,1], txt, fontsize=8)
    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.axhline(0, c='k'); plt.axvline(0, c='k')

show()
