import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURARE ---
FILE_PATH = "Date_CCA.csv"
NAR_VARS_X = 4 # Numar variabile in setul X (primele coloane)

# --- FUNCTII AUXILIARE ---
def nan_replace(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

def show():
    plt.show()

# --- MAIN ---
t = pd.read_csv(FILE_PATH, index_col=0)
nan_replace(t)

# Separare Seturi (Generic)
# Presupunem ca primele N sunt X, restul Y
vars_all = list(t.select_dtypes(include=[np.number]).columns)
vars_x = vars_all[:NAR_VARS_X]
vars_y = vars_all[NAR_VARS_X:]

x = t[vars_x].values
y = t[vars_y].values
obs = t.index
n, p = x.shape
_, q = y.shape

# Standardizare (MANDATORY)
scaler_x = StandardScaler()
x_std = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y_std = scaler_y.fit_transform(y)

# Modelare CCA
m = min(p, q)
cca = CCA(n_components=m)
cca.fit(x_std, y_std)

# 1. Scoruri Canonice (MANDATORY)
z, u = cca.transform(x_std, y_std)

t_z = pd.DataFrame(z, index=obs, columns=["Z"+str(i+1) for i in range(m)])
t_u = pd.DataFrame(u, index=obs, columns=["U"+str(i+1) for i in range(m)])

t_z.to_csv("X_Scores.csv")
t_u.to_csv("Y_Scores.csv")

# 2. Corelatii (Loadings)
rxz = cca.x_loadings_
ryu = cca.y_loadings_

t_rxz = pd.DataFrame(rxz, index=vars_x, columns=["Z"+str(i+1) for i in range(m)])
t_ryu = pd.DataFrame(ryu, index=vars_y, columns=["U"+str(i+1) for i in range(m)])

t_rxz.to_csv("Loadings_X.csv")
t_ryu.to_csv("Loadings_Y.csv")

# Corelatii Canonice (intre Z si U)
correlatii = [np.corrcoef(z[:,i], u[:,i])[0,1] for i in range(m)]
print("Corelatii Canonice:", correlatii)

# 3. Grafice
# Biplot (Z1 vs U1) - Relatia canonica
plt.figure("Biplot Z1-U1")
plt.scatter(z[:,0], u[:,0], c='b')
plt.xlabel("Scor Z1")
plt.ylabel("Scor U1")
plt.title(f"Corelatie Canonica 1: {correlatii[0]:.3f}")

# Cercul Corelatiilor
plt.figure("Cercul Corelatiilor")
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--')
plt.axhline(0, c='k'); plt.axvline(0, c='k')

# Variabile X
plt.scatter(rxz[:,0], rxz[:,1], c='r', marker='o', label='Set X')
for i, txt in enumerate(vars_x):
    plt.text(rxz[i,0], rxz[i,1], txt, color='r')

# Variabile Y
plt.scatter(ryu[:,0], ryu[:,1], c='b', marker='^', label='Set Y')
for i, txt in enumerate(vars_y):
    plt.text(ryu[i,0], ryu[i,1], txt, color='b')
plt.legend()

show()
