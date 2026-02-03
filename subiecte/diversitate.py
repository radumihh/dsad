import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                pd.DataFrame({c: t[c].mean()}, inplace=True)
            else:
                pd.DataFrame({c: t[c].mode()[0]}, inplace=True)

diversitate = pd.read_csv("Diversitate.csv", index_col=0)
cod_loc = pd.read_csv("Coduri_Localitati.csv", index_col=0)

nan_replace_df(diversitate)
nan_replace_df(cod_loc)

ani = list(diversitate.columns[1:])

medie_ani = diversitate[ani].mean(axis=1)
medie_ani.name = "Diversitate"
c1 = pd.concat([diversitate[["Localitate"]], medie_ani], axis=1)
c1 = c1.sort_values("Diversitate", ascending=False)
c1.to_csv("C1.csv")

jud_df = diversitate[ani].merge(cod_loc[["Judet"]], right_index=True, left_index=True)
c2 = ((jud_df[ani]==0).groupby(jud_df["Judet"]).sum())
c2.to_csv("C2.csv")