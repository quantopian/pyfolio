#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division
import os
from collections import OrderedDict

from datetime import datetime
import time

import pandas as pd
import numpy as np
import json
import zlib
import pandas.io.data as web


def json_to_obj(json):
    """
    Convert a JSON string to a DataFrame.
    
    Parameters
    ----------
    json : str
        Daily returns of the strategy, non-cumulative.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the JSON data.
    """

    return pd.json.loads(str(zlib.decompress(json)))


def one_dec_places(x, pos):
    """
    Used to add 1/10th decimal to plot ticks.
    """

    return '%.1f' % x


def percentage(x, pos):
    """
    Used to add percentage sign to plot ticks.
    """

    return '%.0f%%' % x


def round_two_dec_places(x):
    """
    Rounds a number to 1/100th decimal.
    """

    return np.round(x, 2)


def get_symbol_rets(symbol):
    """
    Convert a JSON string to a DataFrame.

    Parameters
    ----------
    json : str
        Daily returns of the strategy, non-cumulative.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the JSON data.
    """

    px = web.get_data_yahoo(symbol, start='1/1/1970')
    px = pd.DataFrame.rename(px, columns={'Adj Close': 'AdjClose'})
    px.columns.name = symbol
    rets = px.AdjClose.pct_change().dropna()
    return rets


def vectorize(func):
    """
    Decorator so that functions can be written to work on Series but may still be called with DataFrames.
    """

    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper

def load_portfolio_risk_factors(filepath_prefix=None):
    """
    Loads historical risk factors -- Mkt-Rf, SMB, HML, Rf, and UMD -- from the 'historical_data' directory.
    
    Loads from F-F_Research_Data_Factors_daily.csv and daily_mom_factor_returns_fixed_dates2.csv.
    
    Parameters
    ----------
    filepath_prefix : str, optional
        Used to specify an exact location of the data.

    Returns
    -------
    five_factors : pd.DataFrame
        A DataFrame containing the JSON data.
    """

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
