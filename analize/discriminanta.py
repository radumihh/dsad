import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from scipy.stats import f
from seaborn import kdeplot,scatterplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,cohen_kappa_score

t = pd.read_csv("data_in/Hernia/hernia.csv",index_col=0)
variabile = list(t)
tinta = variabile[-1]
predictori = variabile[:-1]

# Splitare in invatare si testare
t_train,t_test,y_train,y_test = train_test_split(t[predictori],t[tinta],test_size=0.3)

# Evaluare predictori
x = t_train[predictori].values
x_ = np.mean(x,axis=0)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(t_train,y_train)

g = model_lda.means_
n = len(t_train)
ponderi = model_lda.priors_
q = len(ponderi)
dg = np.diag(ponderi)*n
# print(dg)

# Imprastiere totala
sst = (x-x_).T@(x-x_)

# Imprastiere interclasa
ssb = (g-x_).T@dg@(g-x_)

# Imprastiere intraclasa
ssw = sst-ssb

# Putere discriminare
def f_distributii(t:pd.DataFrame,variabila,y,clase,titlu="Distributii"):
    f = plt.figure(titlu+" - "+variabila,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu+" - "+variabila,fontdict={"fontsize":16})
    kdeplot(t,x=variabila,hue=y,hue_order=clase,ax=ax,fill=True)
    plt.savefig(f"grafice_out/{titlu}-{variabila}.png")
    plt.close()


f_predictori = (np.diag(ssb)/(q-1))/(np.diag(ssw)/(n-q))
pvalues = 1 - f.cdf(f_predictori,q-1,n-q)

t_predictori = pd.DataFrame(data={"Putere discriminare":f_predictori,"PValues":pvalues}, index=predictori)
t_predictori.to_csv("data_out/Predictori.csv")

clase = model_lda.classes_
for predictor in predictori:
    f_distributii(t_train,predictor,y_train,clase)

# show()
# Testare model pe setul de test

predictie_test_lda = model_lda.predict(t_test)
t_predictii_test = pd.DataFrame(data={tinta:y_test,"Predictie LDA":predictie_test_lda}, index=t_test.index)
t_predictii_test.to_csv("data_out/Predictii_test.csv")

# Evaluare model liniar pe setul de testare (matricea de confuzie + indicatori)
def calcul_metrici(y,predictie,clase):
    cm = confusion_matrix(y,predictie,labels=clase)
    # print(cm)
    t_cm = pd.DataFrame(cm,clase,clase)
    t_cm["Acuratete"] = np.diag(cm)*100/np.sum(cm,axis=1)
    acuratete_g = sum(np.diag(cm))*100/len(y)
    acuratete_m = t_cm["Acuratete"].mean()
    i_ck = cohen_kappa_score(y,predictie,labels=clase)
    acuratete = pd.Series(
        [acuratete_g,acuratete_m,i_ck],["Acuratete globala","Acuratete medie","Index CK"],
        name="Indicatori acuratete"
    )
    return t_cm,acuratete

metrici_lda = calcul_metrici(y_test,predictie_test_lda,clase)
metrici_lda[0].to_csv("data_out/CM_lda.csv")
metrici_lda[1].to_csv("data_out/Acuratete_lda.csv")

# Analiza scorurilor discriminante
def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.to_csv(nume_fisier_output)
    return temp

# Calcul scoruri discriminante model liniar
z = model_lda.transform(t_train)
etichete_z = ["Z"+str(i+1) for i in range(q-1)]
t_z = salvare_ndarray(z,t_train.index,etichete_z,"data_out/z.csv")
assert isinstance(t_z,pd.DataFrame)
t_gz = t_z.groupby(by=y_train).mean()
gz = t_gz.values

# Trasare plot scoruri/instante în axe discriminante
def f_scatter(t:pd.DataFrame,tg:pd.DataFrame,y,clase,
              varx="Z1",vary="Z2",titlu="Plot instante in axe discriminante"):
    f = plt.figure(titlu+" - "+varx+" "+vary,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    scatterplot(t,x=varx,y=vary,hue=y,hue_order=clase,ax=ax)
    scatterplot(tg,x=varx,y=vary,hue=clase,hue_order=clase,legend=False,
                ax=ax,marker = "s", s = 100)
    plt.axvline(0, c="k")
    plt.axhline(0, c="k")
    plt.savefig(f"grafice_out/{titlu}-{varx}-{vary}.png")
    plt.close()
f_scatter(t_z,t_gz,y_train,clase)
z_ = np.mean(z,axis=0)
sst_z = (z-z_).T@(z-z_)
ssb_z = (gz-z_).T@dg@(gz-z_)
ssw_z = sst_z-ssb_z
f_z = (np.diag(ssb_z)/(q-1))/(np.diag(ssw_z)/(n-q))
pvalues_z = 1 - f.cdf(f_z,q-1,n-q)
t_discriminatori = pd.DataFrame(data={"Putere discriminare":f_z,"PValues":pvalues_z}, index=etichete_z)
t_discriminatori.to_csv("data_out/Discriminatori.csv")

# Trasare plot distribuții în axele discriminante
for discriminator in etichete_z:
    f_distributii(t_z,discriminator,y_train,clase,"Distributii discriminatori")

# Predicția în setul de aplicare model liniar
t_ = pd.read_csv("data_in/Hernia/hernia_apply.csv",index_col=0)
predictie_lda = model_lda.predict(t_[predictori])
t_predictii = pd.DataFrame(data={"Predictie LDA":predictie_lda}, index=t_.index)

# Predicția în setul de testare model bayesian
model_bayes = GaussianNB()
model_bayes.fit(t_train,y_train)

# Testare model pe setul de test
predictie_test_bayes = model_bayes.predict(t_test)
t_predictii_test["Predictie Bayes"]:predictie_test_bayes

t_predictii_test.to_csv("data_out/Predictii_test.csv")

# Evaluare model bayesian (matricea de confuzie + indicatori)
metrici_bayes = calcul_metrici(y_test,predictie_test_bayes,clase)
metrici_bayes[0].to_csv("data_out/CM_bayes.csv")
metrici_bayes[1].to_csv("data_out/Acuratete_bayes.csv")

# Predicția în setul de aplicare model bayesian
predictie_bayes = model_bayes.predict(t_)
t_predictii["Predictie Bayes"] = predictie_bayes
t_predictii.to_csv("data_out/Predictii.csv")

# Calcul erori
# Erori lda
err_lda = t_predictii_test[y_test!=predictie_test_lda]
# Erori bayes
err_bayes = t_predictii_test[y_test!=predictie_test_bayes]
# Diferente
diferente = t_predictii_test[predictie_test_lda!=predictie_test_bayes]

err_lda.to_csv("data_out/err_lda.csv")
err_bayes.to_csv("data_out/err_bayes.csv")
diferente.to_csv("data_out/diferente.csv")

def show():
    plt.show()
show()