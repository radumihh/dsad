import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from pandas.api.types import is_numeric_dtype

# Analiza canonică
df = pd.read_csv("data_in/Teritorial_2022.csv", index_col=0)

df_num = df.copy()
for c in df_num.columns:
    df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
df_num = df_num.fillna(df_num.mean(numeric_only=True))

X_cols = df_num.columns[3:12]
Y_cols = df_num.columns[12:]

X = df_num[X_cols].to_numpy(dtype=float)
Y = df_num[Y_cols].to_numpy(dtype=float)

n, p, q = X.shape[0], X.shape[1], Y.shape[1]
m = min(p, q)

sx = StandardScaler()
sy = StandardScaler()
Xz = sx.fit_transform(X)
Yz = sy.fit_transform(Y)

cca = CCA(n_components=m, max_iter=5000)
cca.fit(Xz, Yz)

# Calcul scoruri canonice (variabile canonice)
U, V = cca.transform(Xz, Yz)

# Calcul corelații canonice
r = np.array([np.corrcoef(U[:, k], V[:, k])[0, 1] for k in range(m)])
r2 = r ** 2
print("Corelații canonice (R):", np.round(r, 6))

# Determinare relevanță rădăcini canonice (Test Bartlett)
def bartlett_wilks_test(r, n, p, q):
    r = np.asarray(r)
    m = min(len(r), p, q)
    r2 = r[:m] ** 2
    lambdas = np.array([np.prod(1 - r2[k:]) for k in range(m)])
    c = (n - 1 - (p + q + 1) / 2.0)
    chi2_stat = -c * np.log(lambdas)
    dfs = np.array([(p - k) * (q - k) for k in range(m)], dtype=int)
    pvals = 1 - chi2.cdf(chi2_stat, dfs)
    return lambdas, chi2_stat, dfs, pvals

lambdas, chi2_stat, dfs, pvals = bartlett_wilks_test(r, n, p, q)
tabel_semn = pd.DataFrame({
    "R": np.round(r, 6),
    "R2": np.round(r2, 6),
    "Wilks_Lambda": np.round(lambdas, 6),
    "Chi2": np.round(chi2_stat, 6),
    "df": dfs,
    "p_value": np.round(pvals, 6)
}, index=[f"root{i+1}" for i in range(m)])
tabel_semn.index.name = "Radacina"
print(tabel_semn)

# Calcul corelații variabile observate - variabile canonice
corr_XU = np.corrcoef(Xz.T, U.T)[:p, p:]
corr_YV = np.corrcoef(Yz.T, V.T)[:q, q:]

df_corr_XU = pd.DataFrame(corr_XU, index=X_cols, columns=[f"U{i+1}" for i in range(m)])
df_corr_YV = pd.DataFrame(corr_YV, index=Y_cols, columns=[f"V{i+1}" for i in range(m)])

print("\nCorelații X - U:")
print(df_corr_XU.round(6))
print("\nCorelații Y - V:")
print(df_corr_YV.round(6))

# Trasare plot corelații variabile observate - variabile canonice (cercul corelațiilor)
if m >= 2:
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(0, color="grey", lw=1)
    circle = plt.Circle((0, 0), 1, color="black", fill=False, linestyle="dashed")
    ax.add_patch(circle)

    ax.scatter(df_corr_XU["U1"], df_corr_XU["U2"], label="X (corr cu U1,U2)")
    for name in df_corr_XU.index:
        ax.text(df_corr_XU.loc[name, "U1"], df_corr_XU.loc[name, "U2"], name)

    ax.scatter(df_corr_YV["V1"], df_corr_YV["V2"], label="Y (corr cu V1,V2)")
    for name in df_corr_YV.index:
        ax.text(df_corr_YV.loc[name, "V1"], df_corr_YV.loc[name, "V2"], name)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("axa 1")
    ax.set_ylabel("axa 2")
    ax.set_title("Cercul corelațiilor")
    ax.grid(True)
    ax.legend()
    plt.show()

# Trasre corelograma corelații variabile observate - variabile canonice
plt.figure(figsize=(10, 6))
sb.heatmap(df_corr_XU, annot=True, cmap="coolwarm", center=0)
plt.title("Corelogramă corelații: X - U")
plt.show()

plt.figure(figsize=(10, 6))
sb.heatmap(df_corr_YV, annot=True, cmap="coolwarm", center=0)
plt.title("Corelogramă corelații: Y - V")
plt.show()

# Trasare plot instanțe în spațiile celor două variabile (Biplot)
def biplot(scores_2d, loadings_2d, var_labels, title):
    plt.figure(figsize=(9, 7))
    plt.scatter(scores_2d[:, 0], scores_2d[:, 1], alpha=0.7)
    for i, lab in enumerate(var_labels):
        plt.arrow(0, 0, loadings_2d[i, 0], loadings_2d[i, 1],
                  head_width=0.03, length_includes_head=True)
        plt.text(loadings_2d[i, 0] * 1.08, loadings_2d[i, 1] * 1.08, lab)
    plt.axhline(0, color="grey", lw=1)
    plt.axvline(0, color="grey", lw=1)
    plt.xlabel("axa 1")
    plt.ylabel("axa 2")
    plt.title(title)
    plt.grid(True)
    plt.show()

if m >= 2:
    biplot(U[:, :2], corr_XU[:, :2], X_cols, "Biplot instanțe în spațiul X (U1,U2)")
    biplot(V[:, :2], corr_YV[:, :2], Y_cols, "Biplot instanțe în spațiul Y (V1,V2)")
else:
    plt.figure(figsize=(9, 4))
    plt.scatter(U[:, 0], np.zeros(n), alpha=0.7, label="U1 (spațiul X)")
    plt.scatter(V[:, 0], np.ones(n), alpha=0.7, label="V1 (spațiul Y)")
    plt.yticks([0, 1], ["X", "Y"])
    plt.title("Instanțe în spațiile celor două variabile (1 rădăcină)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Calcul varianță explicată și redundanță informațională
r2_UX = corr_XU ** 2
r2_VY = corr_YV ** 2

VKx = r2_UX.sum(axis=0)
VKy = r2_VY.sum(axis=0)

VKx_prop = VKx / p
VKy_prop = VKy / q

SX = VKx_prop * r2
SY = VKy_prop * r2

tabel_var_red = pd.DataFrame({
    "VKx": np.round(VKx, 6),
    "VKx/p": np.round(VKx_prop, 6),
    "VKy": np.round(VKy, 6),
    "VKy/q": np.round(VKy_prop, 6),
    "R2_canonical": np.round(r2, 6),
    "SX": np.round(SX, 6),
    "SY": np.round(SY, 6)
}, index=[f"root{i+1}" for i in range(m)])
tabel_var_red.index.name = "Radacina"
print("\nVarianță explicată și redundanță informațională:")
print(tabel_var_red)
