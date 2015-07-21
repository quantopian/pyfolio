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

import time

import pandas as pd
import numpy as np
import zlib
import pandas.io.data as web

import urllib2
import zipfile
from StringIO import StringIO
import os.path

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


def load_portfolio_risk_factors():
    """
    Loads risk factors Mkt-Rf, SMB, HML, Rf, and UMD.

    Data is stored in HDF5 file. If the data is more than 2
    days old, redownload from Dartmouth.

    Returns
    -------
    five_factors : pd.DataFrame
        Risk factors timeseries.
    """

    five_factors = None

    # If it's been more than two days since we updated, redownload CSVs
    if time.time() - os.path.getmtime('data/factors.h5') > 60*60*24*2:
        try:
            umd_req = urllib2.urlopen('http://mba.tuck.dartmouth.edu/page'
                                      's/faculty/ken.french/ftp/F-F_Momentum'
                                      '_Factor_daily_CSV.zip')
            factors_req = urllib2.urlopen('http://mba.tuck.dartmouth.edu/pag'
                                          'es/faculty/ken.french/ftp/F-F_Re'
                                          'search_Data_Factors_daily_CSV.zip')

            umd_zip = zipfile.ZipFile(StringIO(umd_req.read()), 'r')
            factors_zip = zipfile.ZipFile(StringIO(factors_req.read()),
                                          'r')
            umd_csv = umd_zip.read('F-F_Momentum_Factor_daily.CSV')
            umd_csv = umd_csv.split('\r\n\r\n')[2]
            factors_csv = factors_zip.read('F-F_Research_Data_'
                                           'Factors_daily.CSV')
            factors_csv = factors_csv.split('\r\n\r\n')[1]

            factors = pd.DataFrame.from_csv(StringIO(factors_csv), sep=',')
            umd = pd.DataFrame.from_csv(StringIO(umd_csv), sep=',')

            five_factors = factors.join(umd)
            five_factors = five_factors / 100
            five_factors.to_hdf('data/factors.h5', 'df')
        except Exception as e:
            print('Unable to download factors: %s' % e)

    if not isinstance(five_factors, pd.DataFrame):
        five_factors = pd.read_hdf('data/factors.h5', 'df')

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
