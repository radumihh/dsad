# CHEATSHEET ANALIZĂ DE DATE - EXAMEN (STIL SEMINAR)
# Sintaxă "Minimalistă" bazată pe flow-ul din seminare (11, 12, 13)
# Include cod concret pentru calculele numerice obligatorii, fără grafice externe.

import numpy as np
import pandas as pd
import sys

# Biblioteci Specifice Seminarelor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cross_decomposition import CCA
from scipy.stats import f

# Setări afișare seminar
pd.set_option("display.max_columns", None)
np.set_printoptions(3, sys.maxsize, suppress=True)

# ==============================================================================
# 0. FUNCȚII AUXILIARE (Din `functii.py` seminare)
# ==============================================================================
# Înlocuiește valorile lipsă cu media (numerice) sau modul (categoriale/string)
def nan_replace_df(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

# ==============================================================================
# 1. ANALIZA FACTORIALĂ (FA) - Seminar 10/11
# ==============================================================================
def analiza_factoriala(t, variabile_observate):
    """
    t: DataFrame-ul inițial
    variabile_observate: lista numelor coloanelor numerice
    """
    x = t[variabile_observate].values
    
    # A. Teste de Factorabilitate
    # 1. Bartlett (p-value < 0.05 => există factori comuni)
    test_bartlett = calculate_bartlett_sphericity(x)
    print("Bartlett:", test_bartlett) # (chi2, p-value)
    
    # 2. KMO (Index Kaiser-Meyer-Olkin > 0.6 => bun)
    kmo_all, kmo_model = calculate_kmo(x)
    # Putem salva KMO pe variabile
    t_kmo = pd.DataFrame(
        data={"KMO": np.append(kmo_all, kmo_model)},
        index=variabile_observate + ["Total"]
    )
    t_kmo.to_csv("dataOUT/KMO.csv")
    
    # B. Construire Model (FactorAnalyzer)
    # Nr factori = nr variabile (inițial, pentru a vedea varianța) sau mai mic
    m = len(variabile_observate)
    model_af = FactorAnalyzer(n_factors=m, rotation=None)
    model_af.fit(x)
    
    # C. Analiza Varianței
    # get_factor_variance() returnează 3 array-uri:
    # [0] SS Loadings (Varianța), [1] Proportion Var, [2] Cumulative Var
    varianta = model_af.get_factor_variance()
    
    t_varianta = pd.DataFrame(
        data={
            "Varianta": varianta[0],
            "Procent varianta": varianta[1],
            "Procent cumulat": varianta[2]
        },
        index=["F"+str(i+1) for i in range(len(varianta[0]))]
    )
    t_varianta.to_csv("dataOUT/Varianta_Factori.csv")
    
    # D. Varianta Specifică & Comunalități
    psi = model_af.get_uniquenesses() # Varianta specifică
    comm = model_af.get_communalities() # Comunalități
    
    t_comm = pd.DataFrame({
        "Comunalitati": comm,
        "Varianta Specifica": psi
    }, index=variabile_observate)
    t_comm.to_csv("dataOUT/Comunalitati.csv")
    
    # E. Scoruri Factoriale (Dacă se cer factori rotiți, schimbi rotation='varimax' sus)
    scoruri = model_af.transform(x)
    # pd.DataFrame(scoruri...).to_csv(...)

# ==============================================================================
# 2. ANALIZA DISCRIMINANTĂ (LDA) - Seminar 12
# ==============================================================================
def analiza_discriminanta(t, predictori, tinta):
    """
    predictori: lista coloanelor X
    tinta: numele coloanei Y (etichete clase)
    """
    # 1. Splitare Train/Test (uzual 70/30 sau 60/40)
    # t[tinta] sunt etichetele
    t_train, t_test, y_train, y_test = train_test_split(
        t[predictori], t[tinta], test_size=0.3
    )
    
    # 2. Model LDA
    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(t_train, y_train)
    
    # 3. Evaluare Predictori (Discriminare) - Calcul manual F-Ratio (Complex)
    # (Vezi cod seminar pentru calcul detaliat cu sst, ssb, ssw dacă se cere explicit puterea discriminare)
    # model_lda.priors_ (probabilitati a priori), model_lda.means_ (media claselor)
    
    # 4. Predicție și Evaluare
    predictie_test = model_lda.predict(t_test)
    
    # Salvare predicții
    t_pred = pd.DataFrame({
        "Real": y_test,
        "Predictie": predictie_test
    }, index=t_test.index)
    t_pred.to_csv("dataOUT/Predictii_Test.csv")
    
    # Matrice Confuzie & Acuratețe
    conf_matrix = confusion_matrix(y_test, predictie_test)
    acc = accuracy_score(y_test, predictie_test)
    
    # Salvare Matrice Confuzie
    # Etichete clase
    clase = model_lda.classes_
    t_conf = pd.DataFrame(conf_matrix, index=clase, columns=clase)
    t_conf.to_csv("dataOUT/Matrice_Confuzie.csv")
    print(f"Acuratete: {acc}")
    
    # 5. Scoruri Discriminante (Z)
    # transform returnează scorurile pe axele discriminante (k-1 axe)
    z = model_lda.transform(t[predictori]) 
    # Atentie: transform se aplică pe tot setul sau pe train, depinde de cerință

# ==============================================================================
# 3. ANALIZA DE CLUSTERI (Ierarhică) - Seminar 13
# ==============================================================================
def analiza_clusteri(t, variabile_observate):
    x = t[variabile_observate].values
    
    # A. Matricea Ierarhiei (Linkage)
    # Metode uzuale: 'ward', 'complete', 'average', 'single'
    # Metrica: 'euclidean' (implicit la ward)
    metoda = "ward"
    h = linkage(x, method=metoda)
    
    # Format 'h': [idx1, idx2, distanța, nr_elem_cluster]
    
    # B. Determinare Partiție Optimală (Metoda Elbow/Maximum Diff)
    # Diferența dintre distanțele de agregare (ultima coloană din h)
    distante = h[:, 2]
    # Calculăm diferențele dintre distanțele pașilor i și i+1
    diff = np.diff(distante, 2) # diferența de ordin 2 (acceleratia graficului)
    
    # Indexul unde diferența e maximă (Elbow point)
    k_opt = len(diff) - np.argmax(diff) + 1 # +1 offset
    # SAU varianta simplă din seminar (manuală):
    # Ne uităm la salturile mari din `distante`.
    
    print("Număr optim clusteri propus:", k_opt)
    
    # C. Obținere Partiție (fcluster)
    # t=k_opt (numărul de clusteri) cu criterion='maxclust'
    p_opt = fcluster(h, t=k_opt, criterion='maxclust')
    
    # Salvare partiție
    t["Cluster_Opt"] = p_opt
    t[["Cluster_Opt"]].to_csv("dataOUT/Partitie_Optimala.csv")
    
    # D. Indici Silhouette
    # Scor global
    scor_silh = silhouette_score(x, p_opt)
    print("Scor Silhouette Global:", scor_silh)
    
    # Scor per instanță
    scoruri_instante = silhouette_samples(x, p_opt)
    t["Silhouette_Values"] = scoruri_instante
    # t.to_csv("Partitie_cu_Silhouette.csv")

# ==============================================================================
# 4. ANALIZA CANONICĂ (CCA) - Seminar (Model)
# ==============================================================================
def analiza_canonica(t, vars_X, vars_Y):
    # Standardizare obligatorie înainte de CCA!
    scaler = StandardScaler()
    X_std = scaler.fit_transform(t[vars_X])
    Y_std = scaler.fit_transform(t[vars_Y])
    
    p = len(vars_X)
    q = len(vars_Y)
    m = min(p, q) # Nr perechi canonice
    
    cca = CCA(n_components=m)
    cca.fit(X_std, Y_std)
    
    # Scoruri (Variabile Canonice U, V)
    X_c, Y_c = cca.transform(X_std, Y_std)
    
    # Corelații Canonice
    # Se calculează ca corelația dintre perechile de scoruri (U1, V1), (U2, V2)...
    red_list = []
    for i in range(m):
        corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        red_list.append(corr)
    
    print("Corelații Canonice:", red_list)
    
    # Loadings (Corelații Variabile Obs - Variabile Canonice)
    # x_loadings_ și y_loadings_
    Rxz = cca.x_loadings_
    Ryu = cca.y_loadings_
    
    pd.DataFrame(Rxz, index=vars_X, columns=["U"+str(i+1) for i in range(m)]).to_csv("dataOUT/Loadings_X.csv")

