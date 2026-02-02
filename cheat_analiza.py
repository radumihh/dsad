# CHEATSHEET ANALIZĂ DE DATE - EXAMEN
# Sintaxă "Minimalistă" bazată pe bibliotecile standard folosite la seminar
# Nu include grafice. Doar calcule numerice.

import pandas as pd
import numpy as np

# Importurile esentiale pentru fiecare tip de analiza
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cross_decomposition import CCA

# ==============================================================================
# 0. PREGĂTIRE DE BAZĂ (Valabil pt toate)
# ==============================================================================
def pregatire_date(df, coloane_numerice):
    # 1. Extragere date
    X = df[coloane_numerice].values
    
    # 2. Curatare NaN (cu media)
    imputer = SimpleImputer(strategy='mean') # Sau manual: df.fillna(df.mean())
    # X = imputer.fit_transform(X) # Daca folositi sklearn, sau manual in Pandas
    
    # 3. Standardizare (OBLIGATORIU pentru majoritatea)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    return X_std, df.index, coloane_numerice

# ==============================================================================
# 1. ANALIZA ÎN COMPONENTE PRINCIPALE (ACP / PCA)
# ==============================================================================
def cod_acp(X_std):
    # Model
    model_pca = PCA()
    model_pca.fit(X_std)
    
    # A. Varianta Componentelor (Valori Proprii / Alpha)
    varianta = model_pca.explained_variance_ 
    varianta_procentuala = model_pca.explained_variance_ratio_
    varianta_cumulata = np.cumsum(varianta_procentuala)
    
    # B. Scorurile (Componentele Principale - C) -> Proiectia instantelor
    scoruri = model_pca.transform(X_std)
    # Salvare scoruri
    # pd.DataFrame(scoruri, index=indecsi, columns=[f"C{i+1}" for i in range(scoruri.shape[1])])
    
    # C. Corelatiile Factoriale (Rxc) -> Intre variabile originale si componente
    # Formula: loadings * sqrt(eigenvalues)
    # In sklearn: components_ sunt vectorii proprii transpusi (Loadings)
    # Corelatie = components_ * sqrt(varianta)
    loadings = model_pca.components_.T * np.sqrt(model_pca.explained_variance_)
    # pd.DataFrame(loadings, index=coloane, columns=[f"C{i+1}"...])
    
    return varianta, scoruri, loadings

# ==============================================================================
# 2. ANALIZA FACTORIALĂ (FA)
# ==============================================================================
def cod_factorial(X_std):
    # A. Teste de Factorabilitate
    # Bartlett (p-value < 0.05 => bun)
    chi_square_value, p_value = calculate_bartlett_sphericity(X_std)
    
    # KMO (> 0.6 => bun)
    kmo_all, kmo_model = calculate_kmo(X_std)
    
    # B. Modelare (Rotatie 'varimax' sau 'none')
    nr_factori = 3 # Se alege pe baza criteriului Kaiser (eigenvalues > 1) de la PCA
    model_fa = FactorAnalyzer(n_factors=nr_factori, rotation="varimax")
    model_fa.fit(X_std)
    
    # C. Varianta Factorilor
    # Returneaza 3 linii: SS Loadings, Proportion Var, Cumulative Var
    matrice_varianta = model_fa.get_factor_variance()
    
    # D. Incarcaturi Factoriale (Loadings) - Matricea Corelatiilor Variabile-Factori
    loadings = model_fa.loadings_
    
    # E. Scorurile Factoriale
    scoruri_factoriale = model_fa.transform(X_std)
    
    # F. Comunalitati (Cat la % din varianta variabilei e explicata de factori)
    comunalitati = model_fa.get_communalities()
    
    return matrice_varianta, loadings, scoruri_factoriale

# ==============================================================================
# 3. ANALIZA DE CLUSTERI (Ierarhică - HCA)
# ==============================================================================
def cod_clusteri(X_std):
    # A. Matricea Ierarhiei (Z) - Linkage
    # Metoda 'ward' si metrica 'euclidean' sunt standard
    matrice_ierarhie = linkage(X_std, method='ward', metric='euclidean')
    # Structura Z: [idx1, idx2, distanta, nr_elemente]
    
    # B. Determinarea nr optim de clusteri (Metoda Elbow pe distante)
    distante = matrice_ierarhie[:, 2] # coloana a 3-a
    diferente = np.diff(distante, 2) # diferenta de ordin 2 (acceleratia)
    point_elbow = np.argmax(diferente) + 1 # +1 pt ajustare index
    k_optim = len(X_std) - point_elbow
    
    # C. Partiționarea (Cui apartine fiecare instanta?)
    # k = numarul de clusteri dorit
    partitie = fcluster(matrice_ierarhie, t=k_optim, criterion='maxclust')
    
    # Salvare
    # df['Cluster'] = partitie
    
    return matrice_ierarhie, partitie

# ==============================================================================
# 4. ANALIZA DISCRIMINANTĂ (LDA)
# ==============================================================================
def cod_discriminant(df, vars_independent, var_tinta):
    X = df[vars_independent].values
    y = df[var_tinta].values
    
    # Model
    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(X, y)
    
    # A. Scorurile Discriminante
    scoruri = model_lda.transform(X)
    
    # B. Predictie
    predictii = model_lda.predict(X)
    
    # C. Evaluare (Matrice Confuzie & Acuratete)
    matrice_confuzie = confusion_matrix(y, predictii)
    acuratete = accuracy_score(y, predictii)
    
    return scoruri, matrice_confuzie, acuratete

# ==============================================================================
# 5. ANALIZA CANONICĂ (CCA)
# ==============================================================================
def cod_canonic(X_std, Y_std):
    # X_std (Set 1), Y_std (Set 2)
    n = X_std.shape[0]
    p = X_std.shape[1]
    q = Y_std.shape[1]
    m = min(p, q) # Numarul de perechi canonice
    
    # Model
    model_cca = CCA(n_components=m)
    model_cca.fit(X_std, Y_std)
    
    # A. Scorurile Canonice (U si V)
    X_c, Y_c = model_cca.transform(X_std, Y_std)
    # X_c sunt scorurile variabilelor canonice din spatiul X
    # Y_c sunt scorurile variabilelor canonice din spatiul Y
    
    # B. Corelatiile Canonice (Vectorul de corelatii dintre perechi)
    # Se calculeaza manual ca corelatia dintre colonale pereche din X_c si Y_c
    corelatii_canonice = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(m)]
    
    # C. Incarcaturi (Corelatii Variabile Observate - Variabile Canonice) (Rxz, Ryu)
    # x_loadings_ si y_loadings_ din sklearn
    loadings_X = model_cca.x_loadings_
    loadings_Y = model_cca.y_loadings_
    
    return X_c, Y_c, corelatii_canonice, loadings_X, loadings_Y
