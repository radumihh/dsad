import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

miscarea_nat = pd.read_csv("MiscareaNaturala.csv")
cod_tari = pd.read_csv("CoduriTariExtins.csv")

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c:t[c].mean()}, inplace=True)
            else:
                t.fillna({c:t[c].mode()[0]}, inplace=True)

nan_replace_df(miscarea_nat)
nan_replace_df(cod_tari)

indicatori = list(miscarea_nat.columns[2:])

medie = miscarea_nat["RS"].mean()
c1 = miscarea_nat[miscarea_nat["RS"]<medie]
c1 = c1.sort_values("RS", ascending=False)
c1 = c1[["Three_Letter_Country_Code", "Country_Name", "RS"]]
c1.to_csv("data_out/C1.csv", index=False)

continent = pd.merge(miscarea_nat,cod_tari[["Three_Letter_Country_Code", "Continent_Name"]],
                               on="Three_Letter_Country_Code")
c2 = pd.DataFrame()

for ind in indicatori:
    idx_max = continent.groupby('Continent_Name')[ind].idxmax()
    tarile_maxime = continent.loc[idx_max].set_index('Continent_Name')['Three_Letter_Country_Code']
    c2[ind] = tarile_maxime

c2.sort_index(inplace=True)
c2.index.name = 'Continent Name'
c2.to_csv('data_out/C2.csv')

