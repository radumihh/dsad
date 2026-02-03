import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

emissions = pd.read_csv("Emissions.csv", index_col=0)
pop_europa = pd.read_csv("PopulatieEuropa.csv", index_col=0)

nan_replace_df(emissions)
nan_replace_df(pop_europa)

emisii_total = list(emissions.columns[1:])
emissions["GreenGE"] = emissions["GreenGE"]*1000
emissions["GreenGIE"] = emissions["GreenGIE"]*1000

tone = pd.DataFrame()
den = emissions[emisii_total].sum(axis=1)
tone["Emisii_total_tone"] = den

c1 = pd.concat([emissions[["Country"]], tone], axis=1)
c1.to_csv("C1.csv")

reg = emissions[emisii_total].merge(pop_europa[["Region", "Population"]], right_on="Code", left_index=True)
num = (reg[emisii_total].mul(reg["Population"], axis=0)).groupby(reg["Region"]).sum()
den = reg.groupby("Region")["Population"].sum()
c2 = num.div(den,axis=0)
c2.to_csv("C2.csv")