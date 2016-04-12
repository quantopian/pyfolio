#
# Copyright 2016 Quantopian, Inc.
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
from datetime import datetime
import errno
from os import makedirs, environ
from os.path import expanduser, join, getmtime, isdir
import warnings

from IPython.display import display
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas_datareader import data as web
import numpy as np

from . import pos
from . import txn

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

MM_DISPLAY_UNIT = 1000000.

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR
}


def cache_dir(environ=environ):
    try:
        return environ['PYFOLIO_CACHE_DIR']
    except KeyError:
        return join(
            environ.get(
                'XDG_CACHE_HOME',
                expanduser('~/.cache/'),
            ),
            'pyfolio',
        )


def data_path(name):
    return join(cache_dir(), name)


def ensure_directory(path):
    """
    Ensure that a directory named "path" exists.
    """
    try:
        makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not isdir(path):
            raise


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


def get_utc_timestamp(dt):
    """
    returns the Timestamp/DatetimeIndex
    with either localized or converted to UTC.

    Parameters
    ----------
    dt : Timestamp/DatetimeIndex
        the date(s) to be converted

    Returns
    -------
    same type as input
        date(s) converted to UTC
    """
    dt = pd.to_datetime(dt)
    try:
        dt = dt.tz_localize('UTC')
    except TypeError:
        dt = dt.tz_convert('UTC')
    return dt


_1_bday = BDay()


def _1_bday_ago():
    return pd.Timestamp.now().normalize() - _1_bday


def get_returns_cached(filepath, update_func, latest_dt, **kwargs):
    """Get returns from a cached file if the cache is recent enough,
    otherwise, try to retrieve via a provided update function and
    update the cache file.

    Parameters
    ----------
    filepath : str
        Path to cached csv file
    update_func : function
        Function to call in case cache is not up-to-date.
    latest_dt : pd.Timestamp (tz=UTC)
        Latest datetime required in csv file.
    **kwargs : Keyword arguments
        Optional keyword arguments will be passed to update_func()

    Returns
    -------
    pandas.DataFrame
        DataFrame containing returns
    """
    update_cache = False

    try:
        mtime = getmtime(filepath)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        update_cache = True
    else:
        if pd.Timestamp(mtime, unit='s') < _1_bday_ago():
            update_cache = True
        else:
            returns = pd.read_csv(filepath, index_col=0, parse_dates=True)
            returns.index = returns.index.tz_localize("UTC")

    if update_cache:
        returns = update_func(**kwargs)
        try:
            ensure_directory(cache_dir())
        except OSError as e:
            warnings.warn(
                'could not update cache: {}. {}: {}'.format(
                    filepath, type(e).__name__, e,
                ),
                UserWarning,
            )

        try:
            returns.to_csv(filepath)
        except OSError as e:
            warnings.warn(
                'could not update cache {}. {}: {}'.format(
                    filepath, type(e).__name__, e,
                ),
                UserWarning,
            )

    return returns


def get_symbol_from_yahoo(symbol, start=None, end=None):
    """Wrapper for pandas.io.data.get_data_yahoo().
    Retrieves prices for symbol from yahoo and computes returns
    based on adjusted closing prices.

    Parameters
    ----------
    symbol : str
        Symbol name to load, e.g. 'SPY'
    start : pandas.Timestamp compatible, optional
        Start date of time period to retrieve
    end : pandas.Timestamp compatible, optional
        End date of time period to retrieve

    Returns
    -------
    pandas.DataFrame
        Returns of symbol in requested period.
    """
    px = web.get_data_yahoo(symbol, start=start, end=end)
    rets = px[['Adj Close']].pct_change().dropna()
    rets.index = rets.index.tz_localize("UTC")
    rets.columns = [symbol]
    return rets


def default_returns_func(symbol, start=None, end=None):
    """
    Gets returns for a symbol.
    Queries Yahoo Finance. Attempts to cache SPY.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. APPL.
    start : date, optional
        Earliest date to fetch data for.
        Defaults to earliest date available.
    end : date, optional
        Latest date to fetch data for.
        Defaults to latest date available.

    Returns
    -------
    pd.Series
        Daily returns for the symbol.
         - See full explanation in tears.create_full_tear_sheet (returns).
    """
    if start is None:
        start = '1/1/1970'
    if end is None:
        end = _1_bday_ago()

    start = get_utc_timestamp(start)
    end = get_utc_timestamp(end)

    if symbol == 'SPY':
        filepath = data_path('spy.csv')
        rets = get_returns_cached(filepath,
                                  get_symbol_from_yahoo,
                                  end,
                                  symbol='SPY',
                                  start='1/1/1970',
                                  end=datetime.now())
        rets = rets[start:end]
    else:
        rets = get_symbol_from_yahoo(symbol, start=start, end=end)

    return rets[symbol]


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


def get_fama_french():
    """Retrieve Fama-French factors via pandas-datareader

    Returns
    -------
    pandas.DataFrame
        Percent change of Fama-French factors
    """
    start = '1/1/1970'
    research_factors = web.DataReader('F-F_Research_Data_Factors_daily',
                                      'famafrench', start=start)[0]
    momentum_factor = web.DataReader('F-F_Momentum_Factor_daily',
                                     'famafrench', start=start)[0]
    five_factors = research_factors.join(momentum_factor).dropna()
    five_factors /= 100.
    five_factors.index = five_factors.index.tz_localize('utc')

    return five_factors


def load_portfolio_risk_factors(filepath_prefix=None, start=None, end=None):
    """
    Loads risk factors Mkt-Rf, SMB, HML, Rf, and UMD.

    Data is stored in HDF5 file. If the data is more than 2
    days old, redownload from Dartmouth.

    Returns
    -------
    five_factors : pd.DataFrame
        Risk factors timeseries.
    """
    if start is None:
        start = '1/1/1970'
    if end is None:
        end = _1_bday_ago()

    start = get_utc_timestamp(start)
    end = get_utc_timestamp(end)

    if filepath_prefix is None:
        filepath = data_path('factors.csv')
    else:
        filepath = filepath_prefix

    five_factors = get_returns_cached(filepath, get_fama_french, end)

    return five_factors.loc[start:end]


def get_treasury_yield(start=None, end=None, period='3MO'):
    """Load treasury yields from FRED.

    Parameters
    ----------
    start : date, optional
        Earliest date to fetch data for.
        Defaults to earliest date available.
    end : date, optional
        Latest date to fetch data for.
        Defaults to latest date available.
    period : {'1MO', '3MO', '6MO', 1', '5', '10'}, optional
        Which maturity to use.

    Returns
    -------
    pd.Series
        Annual treasury yield for every day.
    """
    if start is None:
        start = '1/1/1970'
    if end is None:
        end = _1_bday_ago()

    treasury = web.DataReader("DGS3{}".format(period), "fred",
                              start, end)

    treasury = treasury.ffill()

    return treasury


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
        Daily returns of strategy.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    gross_lev : pd.Series, optional
        The leverage of a strategy.
         - See full explanation in tears.create_full_tear_sheet.


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
    if not raw_positions:
        raise ValueError("The backtest does not have any positions.")
    positions = pd.concat(raw_positions)
    positions = pos.extract_pos(positions, backtest.ending_cash)
    transactions = txn.make_transaction_frame(backtest.transactions)
    if transactions.index.tzinfo is None:
        transactions.index = transactions.index.tz_localize('utc')

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


def get_symbol_rets(symbol, start=None, end=None):
    """
    Calls the currently registered 'returns_func'

    Parameters
    ----------
    symbol : object
        An identifier for the asset whose return
        series is desired.
        e.g. ticker symbol or database ID
    start : date, optional
        Earliest date to fetch data for.
        Defaults to earliest date available.
    end : date, optional
        Latest date to fetch data for.
        Defaults to latest date available.

    Returns
    -------
    pandas.Series
        Returned by the current 'returns_func'
    """
    return SETTINGS['returns_func'](symbol,
                                    start=start,
                                    end=end)


def print_table(table, name=None, fmt=None):
    """Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.

    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if fmt is not None:
        prev_option = pd.get_option('display.float_format')
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    if name is not None:
        table.columns.name = name

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)
