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

import pandas as pd


def get_portfolio_alloc(positions):
    """
    Determines a portfolio's allocations.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains position values or amounts.

    Returns
    -------
    positions_alloc : pd.DataFrame
        Positions and their allocations.
    """
    return positions.divide(
        positions.abs().sum(axis='columns'),
        axis='rows'
    )


def get_long_short_pos(positions):
    """
    Determines the long amount, short amount, and cash of a portfolio.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.

    Returns
    -------
    df_long_short : pd.DataFrame
        Net long, short, and cash positions.
    """

    pos_wo_cash = positions.drop('cash', axis=1)
    longs = pos_wo_cash[pos_wo_cash > 0].sum(axis=1)
    shorts = pos_wo_cash[pos_wo_cash < 0].abs().sum(axis=1)
    cash = positions.cash
    df_long_short = pd.DataFrame({'long': longs,
                                  'short': shorts,
                                  'cash': cash})
    # Normalize data
    df_long_short /= df_long_short.abs().sum(axis=1)

    # Apply gross leverage
    gross_lev = (longs + shorts) / (longs - shorts + cash)
    df_long_short *= gross_lev

    return df_long_short


def get_top_long_short_abs(positions, top=10):
    """
    Finds the top long, short, and absolute positions.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    top : int, optional
        How many of each to find (default 10).

    Returns
    -------
    df_top_long : pd.DataFrame
        Top long positions.
    df_top_short : pd.DataFrame
        Top short positions.
    df_top_abs : pd.DataFrame
        Top absolute positions.
    """

    positions = positions.drop('cash', axis='columns')
    df_max = positions.max()
    df_min = positions.min()
    df_abs_max = positions.abs().max()
    df_top_long = df_max[df_max > 0].nlargest(top)
    df_top_short = df_min[df_min < 0].nsmallest(top)
    df_top_abs = df_abs_max.nlargest(top)
    return df_top_long, df_top_short, df_top_abs


def extract_pos(positions, cash):
    """Extract position values from backtest object as returned by
    get_backtest() on the Quantopian research platform.

    Parameters
    ----------
    positions : pd.DataFrame
        timeseries containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        last_sale_price.
    cash : pd.Series
        timeseries containing cash in the portfolio.

    Returns
    -------
    pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    """
    positions = positions.copy()
    positions['values'] = positions.amount * positions.last_sale_price
    cash.name = 'cash'

    values = positions.reset_index().pivot_table(index='index',
                                                 columns='sid',
                                                 values='values')

    values = values.join(cash)

    return values


def get_turnover(transactions, positions, period=None):
    """
    Portfolio Turnover Rate:

    Average value of purchases and sales divided
    by the average portfolio value for the period.

    If no period is provided the period is one time step.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Contains transactions data.
        - See full explanation in tears.create_full_tear_sheet
    positions : pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    period : str, optional
        Takes the same arguments as df.resample.

    Returns
    -------
    turnover_rate : pd.Series
        timeseries of portfolio turnover rates.
    """

    traded_value = transactions.txn_volume
    portfolio_value = positions.sum(axis=1)
    if period is not None:
        traded_value = traded_value.resample(period, how='sum')
        portfolio_value = portfolio_value.resample(period, how='mean')
    # traded_value contains the summed value from buys and sells;
    # this is divided by 2.0 to get the average of the two.
    turnover = traded_value / 2.0
    turnover_rate = turnover / portfolio_value
    return turnover_rate
