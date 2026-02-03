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

e_nsal = pd.read_csv("E_NSAL_2008-2021.csv", index_col=0)
pop_loc = pd.read_csv("PopulatieLocalitati.csv", index_col=0)

nan_replace_df(e_nsal)
nan_replace_df(pop_loc)

indicatori = list(e_nsal.columns[1:])

index_max = e_nsal[indicatori].idxmax(axis=1)
index_max.name = "Index_max"
c1 = pd.concat([e_nsal[["Localitate"]], index_max], axis=1)
c1.to_csv("C1.csv")

jud_df = e_nsal[indicatori].merge(pop_loc[["Judet", "Populatie"]], right_index=True, left_index=True)
ani_jud = jud_df.groupby("Judet")[indicatori].sum()
pop_jud = jud_df.groupby("Judet")["Populatie"].sum()
jud = ani_jud.div(pop_jud, axis=0)
jud["Rata_medie"] = jud.mean(axis=1)
jud = jud.sort_values("Rata_medie", ascending=False)
c2 = jud.reset_index()
c2.to_csv("C2.csv", index=False)