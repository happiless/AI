import pandas as pd
import numpy as np


def make_df(col, ind):
    return pd.DataFrame({c: [str(c)+str(i) for i in ind] for c in col})


print(make_df('ABC', '123'))
