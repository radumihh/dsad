# => Analiza Factoriala (EFA) pe Teritorial_2022

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import seaborn as sb
import matplotlib.pyplot as plt
import os

OUT_DIR = "grafice_out"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(name, dpi=300):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=dpi, bbox_inches="tight")
    plt.close()

pd.set_option("display.max_columns", None)
np.set_printoptions(3, suppress=True)

t = pd.read_csv("data_in/Teritorial_2022.csv", index_col=0)
variabile_observate = list(t.columns)[3:]
x_df = t[variabile_observate]

from pandas.api.types import is_numeric_dtype

def nan_replace_df(t):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(x_df)  # pe x_df sau pe t (cum are profesorul)
x = x_df.values

n, m = x.shape
model_af = FactorAnalyzer(m, rotation="varimax")
model_af.fit(x)

print("Dimensiuni x (n, m):", (n, m))
print("Primele coloane (variabile_observate):", variabile_observate[:10])

# Analiza factorabilitatii - Bartlett
chi_square, p_value = calculate_bartlett_sphericity(x)
print("\nFactorabilitatea Bartlett")
print(f"Valoarea chi-square: {chi_square}")
print(f"P-Value: {p_value}")

if p_value > 0.001:
    print("Nu exista factori comuni (p > 0.001). Oprire.")
    raise SystemExit(0)

# Analiza factorabilitatii - KMO (pe fiecare variabila + total)
kmo_all, kmo_total = calculate_kmo(x)
print(f"\nScorul KMO total: {kmo_total}")

t_kmo = pd.DataFrame({"KMO": np.append(kmo_all, kmo_total)},
                     index=variabile_observate + ["Total"])
print("\nKMO pe variabile + Total:\n", t_kmo)

# optional: salvare ca la profesor
# t_kmo.to_csv("data_out_fa/kmo.csv")

plt.figure(figsize=(6, max(6, 0.25 * len(t_kmo))))
sb.heatmap(t_kmo, annot=True, cmap="Reds", cbar=True)
plt.title("Index KMO")
plt.tight_layout()
# plt.show()
save_fig("kmo_heatmap.png")

# Calcul varianta factori (cu/fara rotatie)
n_factors = m

# fara rotatie
fa_fara_rotatie = FactorAnalyzer(n_factors=n_factors, rotation=None)
fa_fara_rotatie.fit(x)

# cu rotatie
fa_cu_rotatie = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
fa_cu_rotatie.fit(x)

varianta_fara = fa_fara_rotatie.get_factor_variance()
varianta_cu = fa_cu_rotatie.get_factor_variance()

print("\nVarianta factori fara rotatie (SS Loadings / Proportion Var / Cumulative Var):\n", varianta_fara)
print("\nVarianta factori cu rotatie (SS Loadings / Proportion Var / Cumulative Var):\n", varianta_cu)

# plot "scree" / varianta cumulata (util pt comparatie cu profesor)
cum_var = varianta_cu[2]  # cumulative variance
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.axhline(0.70, linestyle="--")
plt.title("Varianta cumulata (cu rotatie varimax)")
plt.xlabel("Factor")
plt.ylabel("Cumulative variance")
plt.grid(True)
plt.tight_layout()
# plt.show()
save_fig("varianta_cumulata_varimax.png")

# nr minim factori pt 70% (similar cu profesor)
nr_min_factori = int(np.argmax(cum_var >= 0.70) + 1) if np.any(cum_var >= 0.70) else len(cum_var)
print("\nNr minim factori pt >=70% varianta cumulata:", nr_min_factori)

# Calcul corelatii factoriale (cu/fara rotatie)
L_fara = fa_fara_rotatie.loadings_
L_cu = fa_cu_rotatie.loadings_

# tabele cu etichete
et_factori = [f"F{i}" for i in range(1, n_factors + 1)]
t_L_fara = pd.DataFrame(L_fara, index=variabile_observate, columns=et_factori)
t_L_cu = pd.DataFrame(L_cu, index=variabile_observate, columns=et_factori)

print("\nCorelatii factoriale (loadings) fara rotatie:\n", t_L_fara.iloc[:, :min(6, n_factors)])
print("\nCorelatii factoriale (loadings) cu rotatie:\n", t_L_cu.iloc[:, :min(6, n_factors)])

# optional: salvare ca la profesor
# t_L_cu.to_csv("data_out_fa/l.csv")

# Trasare corelograma corelatii factoriale (cu/factoriale)
plt.figure(figsize=(10, max(6, 0.30 * len(variabile_observate))))
sb.heatmap(t_L_fara.iloc[:, :nr_min_factori], center=0, cmap="coolwarm", annot=(m <= 10))
plt.title("Corelograma factoriala fara rotatie")
plt.xlabel("Factori")
plt.ylabel("Variabile")
plt.tight_layout()
# plt.show()
save_fig("corelograma_loadings_fara_rotatie.png")

plt.figure(figsize=(10, max(6, 0.30 * len(variabile_observate))))
sb.heatmap(t_L_cu.iloc[:, :nr_min_factori], center=0, cmap="coolwarm", annot=(m <= 10))
plt.title("Corelograma factoriala cu rotatie (varimax)")
plt.xlabel("Factori")
plt.ylabel("Variabile")
plt.tight_layout()
# plt.show()
save_fig("corelograma_loadings_cu_rotatie.png")

# Trasare cercul corelatiilor (cu/fara rotatie)
def cerc_corelatii(loadings_df: pd.DataFrame, titlu: str):
    if loadings_df.shape[1] < 2:
        print("Nu ai cel putin 2 factori pentru cercul corelatiilor.")
        return

    x1 = loadings_df.iloc[:, 0].values
    y1 = loadings_df.iloc[:, 1].values

    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="dashed")
    ax.add_patch(circle)

    plt.scatter(x1, y1)

    for i, label in enumerate(loadings_df.index):
        plt.text(x1[i], y1[i], label, fontsize=8, ha="right", va="bottom")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title(titlu)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    safe_name = titlu.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_fig(f"{safe_name}.png")


cerc_corelatii(t_L_fara.iloc[:, :nr_min_factori], "Cercul corelatiilor fara rotatie (F1 vs F2)")
cerc_corelatii(t_L_cu.iloc[:, :nr_min_factori], "Cercul corelatiilor cu rotatie varimax (F1 vs F2)")

# Calcul comunalitati si varianta specifica
comunalitati = fa_cu_rotatie.get_communalities()
psi = fa_cu_rotatie.get_uniquenesses()

t_comm = pd.DataFrame(
    comunalitati,
    index=variabile_observate,
    columns=["Comunalitati"]
)

print("\nComunalitati:\n")
print(t_comm)

psi = fa_cu_rotatie.get_uniquenesses()

t_psi = pd.DataFrame(
    psi,
    index=variabile_observate,
    columns=["Varianta_specifica"]
)

print("\nVarianta specifica:\n")
print(t_psi)

# optional: salvare ca la profesor
# t_comm.to_csv("data_out_fa/Comm.csv")
# t_psi.to_csv("data_out_fa/Varianta_specifica.csv")

# Trasare corelograma comunalitati si varianta specifica
plt.figure(figsize=(10, 4))
sb.barplot(x=t_comm.index, y=t_comm["Comunalitati"].to_numpy())
plt.title("Comunalitati (rotatie varimax)")
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()
save_fig("comunalitati_barplot.png")

# Calcul scoruri (cu/fara rotatie)
F_fara = fa_fara_rotatie.transform(x)
F_cu = fa_cu_rotatie.transform(x)

t_F_fara = pd.DataFrame(F_fara, index=t.index, columns=et_factori)
t_F_cu = pd.DataFrame(F_cu, index=t.index, columns=et_factori)

print("\nScoruri fara rotatie (primele randuri):\n", t_F_fara.iloc[:5, :nr_min_factori])
print("\nScoruri cu rotatie (primele randuri):\n", t_F_cu.iloc[:5, :nr_min_factori])

# optional: salvare ca la profesor
# t_F_cu.to_csv("data_out_fa/f.csv")

# Trasare plot scoruri
def plot_scoruri(scores_df: pd.DataFrame, titlu: str):
    if scores_df.shape[1] < 2:
        print("Nu ai cel putin 2 factori pentru plot scoruri.")
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(scores_df.iloc[:, 0], scores_df.iloc[:, 1], alpha=0.8)
    plt.title(titlu)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    safe_name = titlu.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_fig(f"{safe_name}.png")

plot_scoruri(t_F_fara.iloc[:, :nr_min_factori], "Plot scoruri factoriale fara rotatie (F1 vs F2)")
plot_scoruri(t_F_cu.iloc[:, :nr_min_factori], "Plot scoruri factoriale cu rotatie varimax (F1 vs F2)")
