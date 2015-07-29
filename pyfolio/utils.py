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
from os.path import (
    abspath,
    dirname,
    getmtime,
    join,
)

from datetime import datetime
import pytz

import pandas as pd
import numpy as np
import zlib
import pandas.io.data as web

import zipfile


try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from io import StringIO
except:
    from urllib2 import urlopen
    from StringIO import StringIO

from . import pos
from . import txn


def pyfolio_root():
    return dirname(abspath(__file__))


def data_path(name):
    return join(pyfolio_root(), 'data', name)


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


def get_utc_date(dt):
    """
    returns the UTC representation
    of a timestamp or DatetimeIndex

    Parameters
    ----------

    dt : pandas.Timestamp or pandas.DatetimeIndex
        The date(s) to be converted

    Returns
    -------
    same dtype as dt
        UTC representation of the input date(s)
    """
    try:
        dt = dt.tz_localize(pytz.UTC)
    except TypeError:
        dt = dt.tz_convert(pytz.UTC)
    return dt.normalize()


def default_returns_func(symbol):
    """
    Gets returns for a symbol.
    Queries Yahoo Finance. Attempts to cache SPY in HDF5.
    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. APPL.

    Returns
    -------
    pd.DataFrame
        Daily returns for the symbol.
    """

    rets = None
    if symbol == 'SPY':
        filepath = data_path('spy.h5')
        try:
            # If it's been less than a day since we got benchmark
            if datetime.now() - pd.to_datetime(
                    getmtime(filepath), unit='s') < pd.Timedelta(days=1):
                rets = pd.read_hdf(filepath, 'df')
                rets.index = get_utc_date(rets.index)
        except:
            pass

    if rets is None:
        px = web.get_data_yahoo(symbol, start='1/1/1970')
        px = pd.DataFrame.rename(px, columns={'Adj Close': 'AdjClose'})
        px.columns.name = symbol
        px.index = get_utc_date(px.index)
        rets = px.AdjClose.pct_change().dropna()

        if symbol == 'SPY':
            try:
                rets.to_hdf(filepath, 'df')
            except:
                pass

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
    Loads risk factors Mkt-Rf, SMB, HML, Rf, and UMD.

    Data is stored in HDF5 file. If the data is more than 2
    days old, redownload from Dartmouth.

    Returns
    -------
    five_factors : pd.DataFrame
        Risk factors timeseries.
    """

    if filepath_prefix is None:
        filepath = data_path('factors.h5')
    else:
        filepath = filepath_prefix

    five_factors = None

    # If it's been less than two days since we updated
    if datetime.now() - pd.to_datetime(getmtime(filepath),
                                       unit='s') < pd.Timedelta(days=2):
        try:
            five_factors = pd.read_hdf(filepath, 'df')
        except:
            pass

    if not isinstance(five_factors, pd.DataFrame):
        try:
            umd_req = urlopen('http://mba.tuck.dartmouth.edu/page'
                              's/faculty/ken.french/ftp/F-F_Momentum'
                              '_Factor_daily_CSV.zip')
            factors_req = urlopen('http://mba.tuck.dartmouth.edu/pag'
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

            five_factors = factors.join(umd).dropna(axis=0)
            five_factors = five_factors / 100

            five_factors.to_hdf(filepath, 'df')
        except:
            pass

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
    if backtest.index.tzinfo is None:
        backtest.index = backtest.index.tz_localize('UTC')
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

    return returns, positions, transactions, gross_lev


# Settings dict to store functions/values that may
# need to be overridden depending on the users environment
SETTINGS = {
    'returns_func': default_returns_func
}


def register_return_func(func):
    """
    Registers the 'returns_func' that will be called for
    retrieving returns data.

    Parameters
    ----------
    func : function
        A function that returns a pandas Series of asset returns.
        The signature of the function must be as follows

        >>> func(symbol)

        Where symbol is an asset identifier

    Returns
    -------
    None
    """
    SETTINGS['returns_func'] = func


def get_symbol_rets(symbol):
    """
    Calls the currently registered 'returns_func'

    Parameters
    ----------
    symbol : object
        An identifier for the asset whose return
        series is desired.
        e.g. ticker symbol or database ID

    Returns
    -------
    pandas.Series
        the series returned by the current 'returns_func'
    """
    return SETTINGS['returns_func'](symbol)
