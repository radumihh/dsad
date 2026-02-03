import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# --- CONFIGURARE ---
FILE_PATH = "Date_LDA.csv"

# --- FUNCTII AUXILIARE ---
def nan_replace(t):
    for c in t.columns:
        if t[c].isna().any():
            if pd.api.types.is_numeric_dtype(t[c]):
                t[c] = t[c].fillna(t[c].mean())
            else:
                t[c] = t[c].fillna(t[c].mode()[0])

def calcul_metrici(y_true, y_pred, clase):
    cm = confusion_matrix(y_true, y_pred, labels=clase)
    t_cm = pd.DataFrame(cm, index=clase, columns=clase)
    
    # Acuratete pe clase (Diagonal / Suma linie)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_clase = np.diag(cm) * 100 / np.sum(cm, axis=1)
    t_cm["Acuratete"] = np.nan_to_num(acc_clase)
    
    # Statistici globale
    acc_global = accuracy_score(y_true, y_pred) * 100
    acc_medie = np.mean(t_cm["Acuratete"])
    kappa = cohen_kappa_score(y_true, y_pred, labels=clase)
    
    stats = pd.Series({
        "Acuratete globala": acc_global,
        "Acuratete medie": acc_medie,
        "Kappa": kappa
    })
    return t_cm, stats

def show():
    plt.show()

# --- MAIN ---
t = pd.read_csv(FILE_PATH, index_col=0)
nan_replace(t)

# Separare X si Y (Presupunem ultima coloana este target)
vars = list(t.columns[:-1])
target = t.columns[-1]

x = t[vars].values
y = t[target].values
obs = t.index

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Modelare LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# 1. Evaluare Model (NON-MANDATORY but useful)
pred_test = lda.predict(x_test)
clase = lda.classes_

t_cm, stats = calcul_metrici(y_test, pred_test, clase)
print("Matrice de confuzie si acuratete:\n", t_cm)
print("Statistici globale:\n", stats)
t_cm.to_csv("Matrice_Confuzie.csv")

# Predicție pe set complet
pred_total = lda.predict(x)
t_rez = t.copy()
t_rez["Predictie"] = pred_total
t_rez.to_csv("Rezultate_Predictie.csv")

# 2. Scoruri Discriminante (MANDATORY)
z = lda.transform(x)
k = z.shape[1] # min(nr_clase-1, nr_vars)
t_z = pd.DataFrame(z, index=obs, columns=["Z"+str(i+1) for i in range(k)])
t_z.to_csv("Scoruri_LDA.csv")

# 3. Grafice
# Plot axe discriminante
if k >= 2:
    plt.figure("Plot Scoruri Discriminante", figsize=(10,6))
    for cls in clase:
        mask = (y == cls)
        plt.scatter(z[mask, 0], z[mask, 1], label=str(cls), alpha=0.7)
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.legend()

# Distributii pe prima axa
plt.figure("Distributii Scoruri Z1")
for cls in clase:
    sb.kdeplot(z[y==cls, 0], label=str(cls), fill=True) 
plt.legend()

show()
