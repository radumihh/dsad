import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

df_div = pd.read_csv("Diversitate.csv", index_col=0)

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

nan_replace_df(df_div)

ani = list(df_div.columns[1:]) # numele de coloane

mask = (df_div[ani] == 0).any(axis=1)
c1 = df_div[mask]
c1 = c1[["Localitate"] + ani]
c1.to_csv("data_out/C1.csv")

df_cod = pd.read_csv("Coduri_Localitati.csv")
nan_replace_df(df_cod)

df_div["Diversitate_medie"] = df_div[ani].mean(axis=1)
jud = df_div.join(df_cod[["Judet"]])
idx = jud.groupby("Judet")["Diversitate_medie"].idxmax()
c2 = jud.loc[idx, ["Judet", "Localitate", "Diversitate_medie"]]
c2.rename(columns={"Diversitate_medie":"Diversitate_maxima"}, inplace=True)
c2.to_csv("data_out/C2.csv", index=False)

