import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if t[c].isna().any():
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

calitate_aer = pd.read_csv("CalitateaAeruluiTari.csv", index_col=0)
cod_tari = pd.read_csv("CoduriTari.csv", index_col=0)

nan_replace_df(calitate_aer)
nan_replace_df(cod_tari)

indicatori = list(calitate_aer.columns[1:])

coef_var = calitate_aer[indicatori].std()/calitate_aer[indicatori].mean()
c1 = coef_var.reset_index()
c1.columns = ["Indicator", "CV"]
c1.to_csv("C1.csv", index=False)

continent_df = calitate_aer[indicatori].merge(cod_tari["Continent"], right_index=True, left_index=True)
continent = continent_df.groupby("Continent")[indicatori].std(ddof=0)/continent_df.groupby("Continent")[indicatori].mean()
idx_max = continent.idxmax(axis=1, skipna=True)
val_max = continent.max(axis=1, skipna=True)
c2 = pd.DataFrame({
    "Indicator": idx_max.values,
    "CV": val_max.values
})
c2.to_csv("C2.csv", index=False)