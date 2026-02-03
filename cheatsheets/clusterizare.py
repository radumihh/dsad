import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# --- CONFIGURARE ---
FILE_PATH = "Date_Cluster.csv"
METHOD = "ward" # ward, complete, average, single
METRIC = "euclidean"

# --- FUNCTII AUXILIARE ---
def nan_replace(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

def calcul_partitie(h, k=None):
    # Functie adaptata din seminar (Elbow method)
    m = h.shape[0]
    n = m + 1
    if k is None:
        # Metoda Elbow
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        j = n - k
        
    # Threshold de taiere (mijlocul saltului)
    dist_k = h[j-1, 2] 
    # Atentie la indexare: h[j] este unirea care reduce nr clusteri de la n-j la n-j-1
    # Daca vrem k clusteri, ne uitam la pasul n-k (adica index n-k-1)
    # Seminar logic: j is index in diferente array.
    # Daca j = argmax(diff) + 1 ==> diferenta maxima e intre pasul j si j-1
    
    threshold = (h[j, 2] + h[j-1, 2]) / 2
    
    # Determinare label-uri
    # fcluster cu 'maxclust' este echivalent cu taierea la k
    c = fcluster(h, t=k, criterion='maxclust')
    
    # Formatare ca string "C1", "C2"...
    partitie = np.array(["C"+str(x) for x in c])
    return k, threshold, partitie

def show():
    plt.show()

# --- MAIN ---
t = pd.read_csv(FILE_PATH, index_col=0)
nan_replace(t)

vars = list(t.select_dtypes(include=[np.number]).columns)
x = t[vars].values
obs = t.index

# Standardizare (MANDATORY)
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# 1. Matrice Ierarhie (MANDATORY)
h = linkage(x_std, method=METHOD, metric=METRIC)
t_h = pd.DataFrame(h, columns=["Cluster 1", "Cluster 2", "Distanta", "Nr.Obs"])
print("Matrice Ierarhie (primele 5 linii):\n", t_h.head())

# 2. Partitie Optimala
k_opt, thresh, labels_opt = calcul_partitie(h, k=None)
print(f"Numar optimal clusteri: {k_opt}")

t_part = pd.DataFrame({"Cluster": labels_opt}, index=obs)
t_part.to_csv("Partitie_Optimala.csv")

# 3. Evaluare (Silhouette)
if len(np.unique(labels_opt)) > 1:
    sil_global = silhouette_score(x_std, labels_opt)
    sil_samples = silhouette_samples(x_std, labels_opt)
    print(f"Silhouette Score Global: {sil_global:.3f}")
    t_part["Silhouette"] = sil_samples
    t_part.to_csv("Partitie_si_Silhouette.csv")
else:
    print("Prea putini clusteri pentru Silhouette.")

# 4. Grafice
# Dendrograma
plt.figure("Dendrograma", figsize=(12, 6))
dendrogram(h, labels=obs, leaf_rotation=45, color_threshold=thresh)
plt.axhline(thresh, c='r', linestyle='--', label=f"Threshold (k={k_opt})")
plt.legend()
plt.title("Dendrograma")

# Silhouette Plot
if len(np.unique(labels_opt)) > 1:
    plt.figure("Silhouette Plot")
    y_lower = 10
    # Convert labels back to ints for plotting logic
    c_int = fcluster(h, t=k_opt, criterion='maxclust')
    for i in range(1, k_opt+1):
        ith_vals = sil_samples[c_int == i]
        ith_vals.sort()
        size_i = ith_vals.shape[0]
        y_upper = y_lower + size_i
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10
    plt.axvline(sil_global, c="red", linestyle="--")
    plt.title(f"Silhouette Plot (Avg: {sil_global:.2f})")

show()
