from __future__ import division
import os
from collections import OrderedDict

import pandas as pd
import numpy as np
import json
import zlib
import pandas.io.data as web


def json_to_obj(json):
    return pd.json.loads(str(zlib.decompress(json)))


def one_dec_places(x, pos):
    return '%.1f' % x


def percentage(x, pos):
    return '%.0f%%' % x


def round_two_dec_places(x):
    return np.round(x, 2)


def get_symbol_rets(symbol):
    px = web.get_data_yahoo(symbol, start='1/1/1970')
    px = pd.DataFrame.rename(px, columns={'Adj Close': 'AdjClose'})
    px.columns.name = symbol
    rets = px.AdjClose.pct_change().dropna()
    return rets


def vectorize(func):
    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper

def load_portfolio_risk_factors(filepath_prefix=None):
    if filepath_prefix is None:
        import quantrisk
        filepath = os.path.join(os.path.dirname(quantrisk.__file__), 'historical_data')
    else:
        filepath = filepath_prefix

    factors = pd.read_csv(os.path.join(
        filepath, 'F-F_Research_Data_Factors_daily.csv'), index_col=0)
    mom = pd.read_csv(os.path.join(
        filepath, 'daily_mom_factor_returns_fixed_dates2.csv'), index_col=0, parse_dates=True)

    factors.index = [datetime.fromtimestamp(
        time.mktime(time.strptime(str(t), "%Y%m%d"))) for t in factors.index]

    five_factors = factors.join(mom)
    # transform the returns from percent space to raw values (to be consistent
    # with our portoflio returns values)
    five_factors = five_factors / 100

    return five_factors
