import pandas as pd
import numpy as np

# ==============================================================================
# CHEATSHEET PANDAS & NUMPY - EXAMEN (CERINȚE 1 și 2)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CITIREA ȘI PREGĂTIREA DATELOR
# ------------------------------------------------------------------------------

# Citire simplă (cu index pe prima coloană dacă e ID unic)
df = pd.read_csv("dataIN/Fisier.csv", index_col=0) 

# Dacă fișierul are alt separator (ex: ';') sau decimal ','
# df = pd.read_csv("dataIN/Fisier.csv", sep=';', decimal=',')

# Înlocuirea valorilor lipsă (NaN) cu media coloanei sau 0
df = df.fillna(0)
# SAU
# df = df.fillna(df.mean())

# ------------------------------------------------------------------------------
# 2. MERGE (COMBINAREA A DOUĂ TABELE) - Vital pentru Coduri Județe/Regiuni
# ------------------------------------------------------------------------------
# Presupunem df1 (Date) și df2 (Coduri/Județe)
# 'left_on' și 'right_on' sunt coloanele comune (cheia)

# Varianta 1: Cheia e coloană în ambele
# df_merge = pd.merge(df1, df2, on="CodSiruta")

# Varianta 2: Cheia e index în unul, coloană în celălalt
# df_merge = pd.merge(df1, df2, left_index=True, right_on="CodSiruta")

# ------------------------------------------------------------------------------
# 3. LISTE DE COLOANE (Foarte util pentru operații pe ani/sectoare)
# ------------------------------------------------------------------------------
# Cum selectezi toți anii sau toate industriile fără să le scrii manual?

# A. Dacă sunt toate coloanele de la a 3-a până la final:
cols_ani = df.columns[3:] 

# B. Dacă sunt coloane numerice specifice:
# cols_ind = ["Ind1", "Ind2", "Ind3"]

# ------------------------------------------------------------------------------
# 3.1. OPERAȚII DE BAZĂ (ADD, RENAME, DROP, FILTER LIST)
# ------------------------------------------------------------------------------

# A. Adăugarea unei coloane noi
# Cu o valoare constantă
df["ColoanaNoua"] = 0 
# Prin calcul (A + B)
df["VenitTotal"] = df["Salariu"] + df["Bonus"]

# B. Ștergerea unei coloane
# axis=1 pentru coloane
df = df.drop(columns=["ColoanaInutila"]) # sau df.drop("ColoanaInutila", axis=1)

# C. Redenumirea coloanelor
# Se dă un dicționar {Vechi: Nou}
df = df.rename(columns={"OldName": "NewName", "T1": "Trimestrul1"})

# D. Filtrare pe baza unei LISTE (isin)
# Foarte util: "Păstrați doar județele din lista cerută"
lista_judete_target = ["Alba", "Cluj", "Sibiu"]
df_filtrat = df[df["NumeJudet"].isin(lista_judete_target)]

# Negarea filtrului (cei care NU sunt in listă) folosind ~
df_restul = df[~df["NumeJudet"].isin(lista_judete_target)]

# E. Selectarea coloanelor prin excludere (ex: totul în afară de 'Data')
cols_to_keep = [c for c in df.columns if c != "Data"]
df_subset = df[cols_to_keep]

# F. Sortarea datelor
# Ascendent (implicit)
df_sortat = df.sort_values(by="Venit")
# Descendent
df_sortat_desc = df.sort_values(by="Venit", ascending=False)
# După mai multe coloane (întâi după Județ, apoi după Venit)
df_multi_sort = df.sort_values(by=["Judet", "Venit"], ascending=[True, False])

# ------------------------------------------------------------------------------
# 4. CERINȚA 1 - OPERAȚII PE LINII (Calculare indicatori noi, Filtrare)
# ------------------------------------------------------------------------------

# A. Calcul matematic simplu între coloane (Vectorizat - Foarte Rapid)
df["SporNatural"] = df["Nascuti"] - df["Decedati"]
df["Densitate"] = df["Populatie"] / df["Suprafata"]

# B. Sumă pe linie (ex: Total Cifra Afaceri din toate industriile)
# axis=1 înseamnă "pe orizontală" (adună coloanele)
df["Total_Business"] = df[cols_ani].sum(axis=1)

# C. Medie pe linie
df["Medie_Ani"] = df[cols_ani].mean(axis=1)

# D. FILTRARE COMPLEXĂ (Ex: "Localități unde diversitatea a fost 0 în cel puțin un an")
# (df[cols_ani] == 0) returnează True/False
# .any(axis=1) returnează True dacă măcar o valoare pe linie e True
# .all(axis=1) returnează True doar dacă TOATE valorile pe linie sunt True

mask_zero = (df[cols_ani] == 0).any(axis=1)
rezultat_1 = df[mask_zero]

# Salvare coloane specifice
rezultat_1[["NumeLocalitate", "SporNatural"]].to_csv("dataOUT/Cerinta1.csv")

# ------------------------------------------------------------------------------
# 5. CERINȚA 2 - GRUPĂRI ȘI AGREGARE (Nivel de Județ/Continent/Regiune)
# ------------------------------------------------------------------------------
# Model clasic: "Pentru fiecare județ, determinați..."

# A. Sumă/Medie pe grup (Ex: Populația totală pe Județ)
grup_judet = df.groupby("Judet")["Populatie"].sum()
# grup_judet este acum o Serie cu Index=Judet, Value=Suma

# B. Ceva mai complex: Procente (Cât % reprezintă cheltuiala X din total pe județ)
# Pas 1: Calcul sume pe județ
df_agg = df.groupby("Judet")[["Cheltuiala1", "TotalCheltuieli"]].sum()
# Pas 2: Calcul procent direct pe grupul agregat
df_agg["Procent"] = (df_agg["Cheltuiala1"] / df_agg["TotalCheltuieli"]) * 100

# ------------------------------------------------------------------------------
# 6. CERINȚA 2 "HARDCORE" - "Localitatea cu valoarea MAXIMĂ din fiecare Județ"
# ------------------------------------------------------------------------------
# Aceasta e cea mai frecventă capcană. Nu folosiți `max()`, ci `idxmax()`.

# Pas 1: Grupăm după Județ și luăm coloana de interes (ex: 'Valoare')
# idxmax() returnează INDEXUL liniei unde se află maximul
idx_maxime = df.groupby("Judet")["Valoare"].idxmax()

# Pas 2: Selectăm liniile din dataframe-ul original folosind indicii găsiți
localitati_top = df.loc[idx_maxime]

# Pas 3: Selectăm coloanele cerute (Judet, NumeLocalitate, Valoare)
rezultat_2 = localitati_top[["Judet", "NumeLocalitate", "Valoare"]]
rezultat_2.to_csv("dataOUT/Cerinta2.csv", index=False)

# ------------------------------------------------------------------------------
# 7. PREGĂTIRE MATRICE: CURĂȚARE (NaN) ȘI STANDARDIZARE
# ------------------------------------------------------------------------------

# --- GHID RAPID: CÂND TREBUIE SĂ STANDARDIZEZ? ---
# 1. ACP (PCA):            DA (Obligatoriu dacă unitățile de măsură diferă - ex: $ vs %)
# 2. Factorială (FA):      DA (De obicei se lucrează pe Corelații = date standardizate)
# 3. Clusteri (HCA):       DA (CRITIC! Distanțele sunt distorsionate de valori mari)
# 4. Discriminantă (LDA):  DA (Recomandat pentru scoruri și interpretare axe)
# 5. Canonică (CCA):       DA (Obligatoriu pentru interpretarea corectă a greutăților)

# --- A. FUNCȚIE PENTRU ÎNLOCUIREA VALORILOR LIPSĂ (NaN) ---
# Adesea examenul cere înlocuirea valorilor lipsă cu media coloanei.
def curatare_nan(df):
    """
    Înlocuiește valorile NaN cu media pe coloanele numerice.
    """
    for col in df.columns:
        # Verificăm dacă coloana e numerică
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                media = df[col].mean()
                df[col] = df[col].fillna(media)
    return df

# Utilizare:
# df = curatare_nan(df)

# --- B. FUNCȚIE PENTRU STANDARDIZARE (Z-SCORE) ---
# Formula: (Valoare - Medie) / Abatere Standard
def standardizare(X):
    """
    Primește un DataFrame sau np.array.
    Returnează matricea standardizată.
    """
    # Se recomandă ddof=0 (doar dacă proful nu cere explicit ddof=1)
    # ddof=0 este abaterea standard a POPULAȚIEI (numitor N)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=0) 
    
    # Evitare împărțire la 0 (în caz că o coloană e constantă)
    stds = np.where(stds == 0, 1, stds)
    
    X_std = (X - means) / stds
    return X_std

# Utilizare practică înainte de analiză:
# 1. Selectăm doar variabilele numerice cauzale (observate)
# vars = df.columns[3:]  # sau o listă manuală
# X = df[vars].values
# 2. Standardizăm
# X_std = standardizare(X)
# 3. Salvăm X_std (cerință frecventă)
# pd.DataFrame(X_std, index=df.index, columns=vars).to_csv("dataOUT/X_std.csv")

# ------------------------------------------------------------------------------
# 8. EXTRAS - MATRICEA DE VARIANȚĂ / COVARIANȚĂ
# ------------------------------------------------------------------------------
# Dacă datele sunt standardizate, Covarianța == Corelația.
# cov_matrix = np.cov(X_std, rowvar=False) 
# pd.DataFrame(cov_matrix, index=vars, columns=vars).to_csv("dataOUT/Cov_Corr.csv")

# ------------------------------------------------------------------------------
# SEMNE ȘI SINTAXĂ RAPIDĂ
# ------------------------------------------------------------------------------
# df.shape         -> (linii, coloane)
# df.index         -> acces la etichetele liniilor
# df.columns       -> acces la etichetele coloanelor
# df.T             -> transpusa matricii
# axis=0           -> Vertical (de-a lungul liniilor, calculează media unei coloane)
# axis=1           -> Orizontal (de-a lungul coloanelor, calculează media unei linii)
