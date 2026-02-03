import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# ==============================================================================
# CHEATSHEET MANIPULARE PANDAS - EXAMEN (METODE SEMINAR)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. FUNCTII UTILITARE (STANDARD SEMINAR)
# ------------------------------------------------------------------------------
def nan_replace(t):
    """Inlocuieste NaN cu media (numeric) sau modul (categorial)."""
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

# Functie pentru calcul procente pe linie (folosita in apply)
def calcul_procent(linie):
    return linie * 100 / linie.sum()

# Functie pentru determinarea categoriei dominante (argmax)
def determina_dominant(linie):
    # linie.values returneaza array numpy
    idx_max = np.argmax(linie.values) 
    val_max = linie.values[idx_max]
    nume_col = linie.index[idx_max]
    return pd.Series([nume_col, val_max], index=["Dominant", "Valoare"])

# ------------------------------------------------------------------------------
# 2. CITIRE SI PREGATIRE
# ------------------------------------------------------------------------------
# Citire cu index pe prima coloana (uzual ID)
df = pd.read_csv("data.csv", index_col=0) 

# Inlocuire valorilor lipsa
nan_replace(df)

# Selectie variabile (Liste de coloane)
# a) Toate coloanele numerice
cols_num = list(df.select_dtypes(include=[np.number]).columns)
# b) Lista explicita (ex: toate industriile/anii)
target_cols = df.columns[3:] 
# c) Prin excludere
cols_fara_pop = [c for c in df.columns if c != "Populatie"]

# ------------------------------------------------------------------------------
# 3. MERGE (COMBINARE TABELE)
# ------------------------------------------------------------------------------
df_coduri = pd.read_csv("coduri.csv", index_col=0)

# Merge pe index (daca ambele au acelasi index, ex: Siruta/CountryID)
df_complet = df.merge(df_coduri, left_index=True, right_index=True)

# Merge index-coloana
# df_complet = df.merge(df_coduri, left_on="CodJudet", right_index=True)

# ------------------------------------------------------------------------------
# 4. CALCULE PE LINII (VECTORIZAT vs APPLY)
# ------------------------------------------------------------------------------
# Varianta A: Vectorizat (Rapid - Recomandat pt operatii simple)
df["Densitate"] = df["Populatie"] / df["Suprafata"]
df["Total"] = df[target_cols].sum(axis=1)

# Varianta B: Apply (Seminar Style - Pt procente/operatii complexe)
# Calcul procent din total pe fiecare linie
df_procente = df[target_cols].apply(func=calcul_procent, axis=1)

# Calcul per capita (daca impartim la o coloana din afara listei target)
# Putem folosi vectorizare:
# df_per_capita = df[target_cols].div(df["Populatie"], axis=0) 
# Sau apply cu lambda:
df_per_capita = df[target_cols].apply(lambda x: x / df["Populatie"], axis=0) 

# ------------------------------------------------------------------------------
# 5. AGREGARE (GROUP BY) - Cerinte de tip "Nivel Judet/Continent"
# ------------------------------------------------------------------------------
# Suma pe grupuri
# Selectam coloanele de interes + coloana de grupare
grupare = df[target_cols + ["Judet"]].groupby(by="Judet").sum()

# Medie pe grupuri
# grupare_medie = df.groupby("Judet")[target_cols].mean()

# Agregare si salvare
grupare.to_csv("Agregare_Judet.csv")

# ------------------------------------------------------------------------------
# 6. FILTRARE SI SELECTIE AVANSATA
# ------------------------------------------------------------------------------
# Filtrare folosind 'isin' (pastram doar anumite judete)
judete_dorite = ["Alba", "Cluj", "Timis"]
df_filtrat = df[df["Judet"].isin(judete_dorite)]

# Filtrare conditionata
df_mari = df[df["Populatie"] > 100000]

# ------------------------------------------------------------------------------
# 7. EXEMPLE "CERINTA 1 & 2" (SPECIFICE EXAMEN)
# ------------------------------------------------------------------------------

# SCENARIU 1: Categoria dominanta pe fiecare linie (ex: Industria cu CA maxima)
# Folosim functia 'determina_dominant' definita sus
rez_dominant = df[target_cols].apply(determina_dominant, axis=1)
# Concatenam cu datele de identificare
final_dominant = pd.concat([df[["NumeLocalitate"]], rez_dominant], axis=1)
final_dominant.to_csv("Cerinta_Ind_Dominanta.csv")

# SCENARIU 2: Localitatea cu valoarea MAXIMA din fiecare Judet
# Folosim idxmax() care returneaza indexul unde e maximul
idx_max = df.groupby("Judet")["ValoareTarget"].idxmax()
localitati_top = df.loc[idx_max] # Selectam liniile complete
localitati_top[["Judet", "NumeLocalitate", "ValoareTarget"]].to_csv("Top_Localitati.csv")

# SCENARIU 3: Coeficientul de variatie (cv = std / mean) la nivel de regiune
# Agregam medii si abateri standard
cv_grup = df.groupby("Continent")[target_cols].apply(lambda x: x.std() / x.mean())
cv_grup.to_csv("Coeficienti_Variatie.csv")

# SCENARIU 4: Rata Spor Natural (Nasteri - Decese) per 1000 locuitori
# (Variabilele sunt deja coloane in df)
df["RataSpor"] = (df["Nasteri"] - df["Decese"]) * 1000 / df["Populatie"]

# ------------------------------------------------------------------------------
# 8. SORTARE COMPLEXA
# ------------------------------------------------------------------------------
# A. Sortare simpla descendenta
df_sortat = df.sort_values(by="RataSpor", ascending=False)

# B. Sortare dupa o valoare calculata "on the fly" (fara a crea coloana noua)
# Ex: Sortare dupa diferenta dintre Export si Import
df_sort_calc = df.iloc[ (df["Export"] - df["Import"]).argsort() ]
# Sau mai simplu, cream coloana temporara, sortam, apoi o stergem (mai safe)

# C. Sortare dupa multiple coloane cu directii diferite
# Ex: Judet (Ascendent A-Z), apoi Populatie (Descendent mereu cele mai mari primele)
df_multi_sort = df.sort_values(by=["Judet", "Populatie"], ascending=[True, False])

# D. Top N valori
top_10_judete = df.sort_values(by="PIB", ascending=False).head(10)

# SCENARIU: Salvarea primelor 3 localitati ca venit din fiecare judet
def get_top_3(grup):
    return grup.sort_values(by="Venit", ascending=False).head(3)

top_3_per_judet = df.groupby("Judet").apply(get_top_3) # Returneaza MultiIndex

# ------------------------------------------------------------------------------
# 9. GRUPARE AVANSATA (GROUPBY)
# ------------------------------------------------------------------------------
# A. Agregare Multipla (Diferite functii pe diferite coloane)
# Ex: Suma la Populatie, Media la Varsta
agg_dict = {
    "Populatie": "sum",
    "Varsta": "mean",
    "Venit": ["min", "max", "mean"] # mai multe functii pe aceeasi coloana
}
df_agg_complex = df.groupby("Judet").agg(agg_dict)
# Atentie: Rezultatul va avea MultiIndex pe coloane daca folosim liste de functii

# B. Grupare dupa mai multe coloane
# Ex: Nivel Continent SI Regiune
df_grup_multi = df.groupby(["Continent", "Regiune"])[target_cols].mean()

# C. Filtrarea grupurilor (ex: Pastram doar judetele cu populatie totala > 1 mil)
def filter_big_groups(x):
    return x["Populatie"].sum() > 1000000

df_only_big_judete = df.groupby("Judet").filter(filter_big_groups)

# D. Transformare (Broadcast rezultatul agregarii inapoi la dimensiunea originala)
# Ex: Adaugam o coloana cu "Media Venitului pe Judetul curent" langa fiecare localitate
df["Media_Judet"] = df.groupby("Judet")["Venit"].transform("mean")
# Acum putem calcula abaterea fata de media judetului
df["Abatere_fata_de_Judet"] = df["Venit"] - df["Media_Judet"]
