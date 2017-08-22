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

import warnings

import numpy as np
import pandas as pd
from IPython.display import display
from empyrical.utils import default_returns_func

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


def one_dec_places(x, pos):
    """
    Adds 1/10th decimal to plot ticks.
    """

    return '%.1f' % x


def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """

    return '%.2f' % x


def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x


def format_asset(asset):
    """
    If zipline asset objects are used, we want to print them out prettily
    within the tear sheet. This function should only be applied directly
    before displaying.
    """

    try:
        import zipline.assets
    except:
        return asset

    if isinstance(asset, zipline.assets.Asset):
        return asset.symbol
    else:
        return asset


def vectorize(func):
    """
    Decorator so that functions can be written to work on Series but
    may still be called with DataFrames.
    """

    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper


def extract_rets_pos_txn_from_zipline(backtest):
    """
    Extract returns, positions, transactions and leverage from the
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


    Example (on the Quantopian research platform)
    ---------------------------------------------
    >>> backtest = my_algo.run()
    >>> returns, positions, transactions =
    >>>     pyfolio.utils.extract_rets_pos_txn_from_zipline(backtest)
    >>> pyfolio.tears.create_full_tear_sheet(returns,
    >>>     positions, transactions)
    """

    backtest.index = backtest.index.normalize()
    if backtest.index.tzinfo is None:
        backtest.index = backtest.index.tz_localize('UTC')
    returns = backtest.returns
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

    return returns, positions, transactions


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
    """
    Pretty print a pandas DataFrame.

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


def standardize_data(x):
    """
    Standardize an array with mean and standard deviation.

    Parameters
    ----------
    x : np.array
        Array to standardize.

    Returns
    -------
    np.array
        Standardized array.
    """

    return (x - np.mean(x)) / np.std(x)


def detect_intraday(positions, transactions, threshold=0.25):
    """
    Attempt to detect an intraday strategy. Get the number of
    positions held at the end of the day, and divide that by the
    number of unique stocks transacted every day. If the average quotient
    is below a threshold, then an intraday strategy is detected.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.

    Returns
    -------
    boolean
        True if an intraday strategy is detected.
    """

    daily_txn = transactions.copy()
    daily_txn.index = daily_txn.index.date
    txn_count = daily_txn.groupby(level=0).symbol.nunique().sum()
    daily_pos = positions.drop('cash', axis=1).replace(0, np.nan)
    return daily_pos.count(axis=1).sum() / txn_count < threshold


def check_intraday(estimate, returns, positions, transactions):
    """
    Logic for checking if a strategy is intraday and processing it.

    Parameters
    ----------
    estimate: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in tears.create_full_tear_sheet.
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.

    Returns
    -------
    pd.DataFrame
        Daily net position values, adjusted for intraday movement.
    """

    if estimate == 'infer':
        if positions is not None and transactions is not None:
            if detect_intraday(positions, transactions):
                warnings.warn('Detected intraday strategy; inferring positi' +
                              'ons from transactions. Set estimate_intraday' +
                              '=False to disable.')
                return estimate_intraday(returns, positions, transactions)
            else:
                return positions
        else:
            return positions

    elif estimate:
        if positions is not None and transactions is not None:
            return estimate_intraday(returns, positions, transactions)
        else:
            raise ValueError('Positions and txns needed to estimate intraday')
    else:
        return positions


def estimate_intraday(returns, positions, transactions, EOD_hour=23):
    """
    Intraday strategies will often not hold positions at the day end.
    This attempts to find the point in the day that best represents
    the activity of the strategy on that day, and effectively resamples
    the end-of-day positions with the positions at this point of day.
    The point of day is found by detecting when our exposure in the
    market is at its maximum point. Note that this is an estimate.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.

    Returns
    -------
    pd.DataFrame
        Daily net position values, resampled for intraday behavior.
    """

    # Construct DataFrame of transaction amounts
    txn_val = transactions.copy()
    txn_val.index.names = ['date']
    txn_val['value'] = txn_val.amount * txn_val.price
    txn_val = txn_val.reset_index().pivot_table(
        index='date', values='value',
        columns='symbol').replace(np.nan, 0)

    # Cumulate transaction amounts each day
    txn_val['date'] = txn_val.index.date
    txn_val = txn_val.groupby('date').cumsum()

    # Calculate exposure, then take peak of exposure every day
    txn_val['exposure'] = txn_val.abs().sum(axis=1)
    condition = (txn_val['exposure'] == txn_val.groupby(
        pd.TimeGrouper('24H'))['exposure'].transform(max))
    txn_val = txn_val[condition].drop('exposure', axis=1)

    # Compute cash delta
    txn_val['cash'] = -txn_val.sum(axis=1)

    # Shift EOD positions to positions at start of next trading day
    positions_shifted = positions.copy().shift(1).fillna(0)
    starting_capital = positions.iloc[0].sum() / (1 + returns[0])
    positions_shifted.cash[0] = starting_capital

    # Format and add start positions to intraday position changes
    txn_val.index = txn_val.index.normalize()
    corrected_positions = positions_shifted.add(txn_val, fill_value=0)
    corrected_positions.index.name = 'period_close'
    corrected_positions.columns.name = 'sid'

    return corrected_positions


def to_utc(df):
    """
    For use in tests; applied UTC timestamp to DataFrame.
    """

    try:
        df.index = df.index.tz_localize('UTC')
    except TypeError:
        df.index = df.index.tz_convert('UTC')

    return df


def to_series(df):
    """
    For use in tests; converts DataFrame's first column to Series.
    """

    return df[df.columns[0]]
