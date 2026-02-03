# => Analiza de clusteri

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype

from pandas.api.types import is_numeric_dtype

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane,
                    nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x, nume_linii, nume_coloane)
    if nume_fisier_output is not None:
        temp.to_csv(nume_fisier_output)
    return temp

def diferente_agregare(h: np.ndarray):
    m = h.shape[0]
    dif = h[1:, 2] - h[:m - 1, 2]
    j = int(np.argmax(dif) + 1)
    k = int((m + 1) - j)
    return dif, j, k

def calcul_partitie(h: np.ndarray, k=None):
    m = h.shape[0]
    n = m + 1
    if k is None:
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        j = n - k
    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2
    c = np.arange(n)
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i
    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]
    return k, color_threshold, np.array(partitie)

DATA_OUT = "data_out"
GRAFICE_OUT = "grafice_out"
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(GRAFICE_OUT, exist_ok=True)

t = pd.read_csv("data_in/mortalitate_ro.csv", index_col=1)
nan_replace_df(t)

variabile_observate = list(t)[1:]
x = t[variabile_observate].values
instante = t.index

metoda_grupare = "complete"


# Calcul ierarhie (matricea ierarhie)

h = linkage(x, method=metoda_grupare)
pd.DataFrame(h, columns=["i", "j", "dist", "size"]).to_csv(f"{DATA_OUT}/Ierarhie.csv")


# Calcul partiție optimală (Elbow pe diferențe distanțe agregare) (NU KMeans)
dif, j_elbow, k_elbow = diferente_agregare(h)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(dif) + 1), dif, marker="o")
plt.title("Diferentele dintre distantele de agregare (Elbow ierarhic)")
plt.xlabel("Pas de agregare")
plt.ylabel("Diferenta distantei")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/01_elbow_ierarhic_diferente.png", dpi=200, bbox_inches="tight")
# plt.show()

k_opt, color_th_opt, p_opt = calcul_partitie(h)


# Calcul partiție oarecare (k prestabilit / citit)
k_fix = 3
k, color_th_k, p_k = calcul_partitie(h, k_fix)


# Calcul indecși Silhouette la nivel de partiție și de instanțe
sil_inst_opt = silhouette_samples(x, p_opt)
sil_avg_opt = silhouette_score(x, p_opt)

sil_inst_k = silhouette_samples(x, p_k)
sil_avg_k = silhouette_score(x, p_k)

t_part = pd.DataFrame(index=instante)
t_part["Partitie O"] = p_opt
t_part["Scor_Silh_Opt"] = sil_inst_opt
t_part[f"Partitie {k}"] = p_k
t_part.to_csv(f"{DATA_OUT}/Partitii.csv")

pd.DataFrame(
    {"Silhouette_avg": [sil_avg_opt, sil_avg_k]},
    index=["Partitie O", f"Partitie {k}"]
).to_csv(f"{DATA_OUT}/Silhouette_rezumat.csv")


# Trasare plot dendrogramă cu evidențierea partiției (optimală și partiție-k)
plt.figure(figsize=(10, 7))
dendrogram(h, labels=instante.astype(str))
plt.title("Dendrograma - Calcul Ierarhie")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/02_dendrograma.png", dpi=200, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(10, 7))
dendrogram(h, labels=instante.astype(str), color_threshold=color_th_opt)
plt.title(f"Dendrograma - Evidentiere partitie optima (k={k_opt})")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/03_dendrograma_partitie_optima.png", dpi=200, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(10, 7))
dendrogram(h, labels=instante.astype(str), color_threshold=color_th_k)
plt.title(f"Dendrograma - Evidentiere partitie-{k}")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/04_dendrograma_partitie_k.png", dpi=200, bbox_inches="tight")
# plt.show()


# Trasare plot Silhouette partiție (optimală și partiție-k)
sil_df_opt = pd.DataFrame({"Instanta": instante.astype(str), "Silhouette": sil_inst_opt}).sort_values("Silhouette")
plt.figure(figsize=(10, 7))
sb.barplot(data=sil_df_opt, x="Silhouette", y="Instanta", orient="h")
plt.title(f"Silhouette pe instante - Partitia optima (avg={sil_avg_opt:.3f})")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/05_silhouette_optima.png", dpi=200, bbox_inches="tight")
# plt.show()

sil_df_k = pd.DataFrame({"Instanta": instante.astype(str), "Silhouette": sil_inst_k}).sort_values("Silhouette")
plt.figure(figsize=(10, 7))
sb.barplot(data=sil_df_k, x="Silhouette", y="Instanta", orient="h")
plt.title(f"Silhouette pe instante - Partitia-{k} (avg={sil_avg_k:.3f})")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/06_silhouette_k.png", dpi=200, bbox_inches="tight")
# plt.show()


# Trasare histograme clusteri pentru fiecare variabilă observată
# (partiție optimală și partiție-k)
v = variabile_observate[0]   # o singura variabila, exemplu

# histograma - partitia optima
plt.figure(figsize=(10, 6))
tmp = pd.DataFrame({"X": t[v].values, "Cluster": p_opt}, index=instante)
sb.histplot(data=tmp, x="X", hue="Cluster", kde=False, element="bars", stat="count")
plt.title(f"Histograme clusteri - {v} (Partitia optima)")
plt.xlabel(v)
# plt.show()

# histograma - partitia k
plt.figure(figsize=(10, 6))
tmp = pd.DataFrame({"X": t[v].values, "Cluster": p_k}, index=instante)
sb.histplot(data=tmp, x="X", hue="Cluster", kde=False, element="bars", stat="count")
plt.title(f"Histograme clusteri - {v} (Partitia {k})")
plt.xlabel(v)
# plt.show()


# Trasare plot partiție în axe principale (PCA)
# (partiție optimală și partiție-k)
pca = PCA(2)
z = pca.fit_transform(x)
t_z = salvare_ndarray(z, instante, ["Z1", "Z2"], None)

plt.figure(figsize=(10, 7))
plt.scatter(t_z["Z1"], t_z["Z2"], c=p_opt)
for i, inst in enumerate(instante.astype(str)):
    plt.text(t_z["Z1"].iloc[i], t_z["Z2"].iloc[i], inst, fontsize=8)
plt.title(f"Partitie in axe principale (PCA) - Optima (k={k_opt})")
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/07_pca_optima.png", dpi=200, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(t_z["Z1"], t_z["Z2"], c=p_k)
for i, inst in enumerate(instante.astype(str)):
    plt.text(t_z["Z1"].iloc[i], t_z["Z2"].iloc[i], inst, fontsize=8)
plt.title(f"Partitie in axe principale (PCA) - Partitia {k}")
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.tight_layout()
plt.savefig(f"{GRAFICE_OUT}/08_pca_k.png", dpi=200, bbox_inches="tight")
# plt.show()