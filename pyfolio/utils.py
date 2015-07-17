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

from datetime import datetime
import time

import pandas as pd
import numpy as np
import zlib
import pandas.io.data as web

from . import pos
from . import txn


def json_to_obj(json):
    """
    Converts a JSON string to a DataFrame.

    Parameters
    ----------
    json : str
        Data to convert.

    Returns
    -------
    pd.DataFrame
        The converted data.
    """

    return pd.json.loads(str(zlib.decompress(json)))


def one_dec_places(x, pos):
    """
    Adds 1/10th decimal to plot ticks.
    """

    return '%.1f' % x


def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x


def round_two_dec_places(x):
    """
    Rounds a number to 1/100th decimal.
    """

    return np.round(x, 2)


def get_symbol_rets(symbol):
    """
    Gets returns for a symbol.

    Queries Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. APPL.

    Returns
    -------
    pd.DataFrame
        Daily returns for the symbol.
    """

    px = web.get_data_yahoo(symbol, start='1/1/1970')
    px = pd.DataFrame.rename(px, columns={'Adj Close': 'AdjClose'})
    px.columns.name = symbol
    rets = px.AdjClose.pct_change().dropna()
    return rets


def vectorize(func):
    """Decorator so that functions can be written to work on Series but
    may still be called with DataFrames.
    """

    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper


def load_portfolio_risk_factors(filepath_prefix=None):
    """
    Loads historical risk factors -- Mkt-Rf, SMB, HML, Rf,
    and UMD -- from the 'data' directory.

    Loads from F-F_Research_Data_Factors_daily.csv and
    daily_mom_factor_returns_fixed_dates2.csv.

    Parameters
    ----------
    filepath_prefix : str, optional
        Used to specify an exact location of the data.

    Returns
    -------
    five_factors : pd.DataFrame
        Risk factors timeseries.
    """

    if filepath_prefix is None:
        import pyfolio
        filepath = os.path.join(os.path.dirname(pyfolio.__file__), 'data')
    else:
        filepath = filepath_prefix

    factors = pd.read_csv(os.path.join(
        filepath, 'F-F_Research_Data_Factors_daily.csv'), index_col=0)
    mom = pd.read_csv(os.path.join(
        filepath, 'daily_mom_factor_returns_fixed_dates2.csv'),
                      index_col=0, parse_dates=True)

    factors.index = [datetime.fromtimestamp(
        time.mktime(time.strptime(str(t), "%Y%m%d"))) for t in factors.index]

    five_factors = factors.join(mom)
    # transform the returns from percent space to raw values (to be consistent
    # with our portoflio returns values)
    five_factors = five_factors / 100

    return five_factors


def extract_rets_pos_txn_from_zipline(backtest):
    """Extract returns, positions, transactions and leverage from the
    backtest data structure returned by zipline.TradingAlgorithm.run().

    The returned data structures are in a format compatible with the
    rest of pyfolio and can be directly passed to
    e.g. tears.create_full_tear_sheet().

    Parameters
    ----------
    backtest : pd.DataFrame
        DataFrame returned by zipline.TradingAlgorithm.run()

    Returns
    -------
    returns : pd.Series
        Daily returns of backtest
    positions : pd.DataFrame
        Daily net position values
    transactions : pd.DataFrame
        Daily transaction volume and dollar ammount.
    gross_lev : pd.Series
        Daily gross leverage.


    Example (on the Quantopian research platform)
    ---------------------------------------------
    >>> backtest = my_algo.run()
    >>> returns, positions, transactions, gross_lev =
    >>>     pyfolio.utils.extract_rets_pos_txn_from_zipline(backtest)
    >>> pyfolio.tears.create_full_tear_sheet(returns,
    >>>     positions, transactions, gross_lev=gross_lev)

    """

    backtest.index = backtest.index.normalize()
    returns = backtest.returns
    gross_lev = backtest.gross_leverage
    raw_positions = []
    for dt, pos_row in backtest.positions.iteritems():
        df = pd.DataFrame(pos_row)
        df.index = [dt] * len(df)
        raw_positions.append(df)
    positions = pd.concat(raw_positions)
    positions = pos.extract_pos(positions, backtest.ending_cash)
    transactions_frame = txn.make_transaction_frame(backtest.transactions)
    transactions = txn.get_txn_vol(transactions_frame)
    transactions.index = transactions.index.normalize()
    transactions.index.tz = None

    return returns, positions, transactions, gross_lev


def extract_rets_pos_txn_from_backtest_obj(backtest):
    """Extract returns, positions, and transactions from the backtest
    object returned by get_backtest() on the Quantopian research
    platform.

    The returned data structures are in a format compatible with the
    rest of pyfolio and can be directly passed to
    e.g. tears.create_full_tear_sheet().

    Parameters
    ----------
    backtest : qexec.research.backtest.BacktestResult
        Object returned by get_backtest() on the Quantopian research
        platform containing all results of a backtest

    Returns
    -------
    returns : pd.Series
        Daily returns of backtest
    positions : pd.DataFrame
        Daily net position values
    transactions : pd.DataFrame
        Daily transaction volume and dollar ammount.
    gross_lev : pd.Series
        Daily gross leverage.


    Example (on the Quantopian research platform)
    ---------------------------------------------
    >>> backtest = get_backtest('548f4f3d885aef09019e9c36')
    >>> returns, positions, transactions, gross_lev =
    >>>     pyfolio.utils.extract_rets_pos_txn_from_backtest_obj(backtest)
    >>> pyfolio.tears.create_full_tear_sheet(returns,
    >>>     positions, transactions, gross_lev=gross_lev)
    """
    returns = backtest.daily_performance.returns
    returns.index = returns.index.normalize()

    positions = pos.extract_pos(backtest.positions,
                                backtest.daily_performance.ending_cash)
    transactions = txn.get_txn_vol(backtest.transactions)
    gross_lev = backtest.daily_performance.gross_leverage

    return returns, positions, transactions, gross_lev
