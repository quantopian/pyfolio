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
import json
import pandas as pd
import numpy as np


def get_portfolio_alloc(positions_vals):
    """
    Determines a portfolio's allocations.

    Parameters
    ----------
    positions_vals : pd.DataFrame
        Contains position values or amounts.

    Returns
    -------
    positions_alloc : pd.DataFrame
        Positions and their allocations.
    """

    positions_alloc = (
        positions_vals.T /
        positions_vals.abs().sum(
            axis='columns').T).T
    return positions_alloc


def get_long_short_pos(positions, gross_lev=1.):
    """
    Determines the long amount, short amount, and cash of a portfolio.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    gross_lev : float, optional
        The porfolio's gross leverage (default 1).

    Returns
    -------
    df_long_short : pd.DataFrame
        Net long, short, and cash positions.
    """

    positions_wo_cash = positions.drop('cash', axis='columns')
    df_long = positions_wo_cash.apply(lambda x: x[x > 0].sum(), axis='columns')
    df_short = - \
        positions_wo_cash.apply(lambda x: x[x < 0].sum(), axis='columns')
    # Shorting positions adds to cash
    df_cash = positions.cash.abs() - df_short
    df_long_short = pd.DataFrame({'long': df_long,
                                  'short': df_short,
                                  'cash': df_cash})
    # Renormalize
    df_long_short /= df_long_short.sum(axis='columns')

    # Renormalize to leverage
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
    df_max = positions.max().sort(inplace=False, ascending=False)
    df_min = positions.min().sort(inplace=False, ascending=True)
    df_abs_max = positions.abs().max().sort(inplace=False, ascending=False)
    df_top_long = df_max[df_max > 0][:top]
    df_top_short = df_min[df_min < 0][:top]
    df_top_abs = df_abs_max[:top]
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
        Net positional values per SID as well as cash.
    """

    pos = positions.reset_index().groupby(['index', 'sid']).apply(
        lambda ser: ser['amount'] * ser['last_sale_price'])
    pos.index = pos.index.droplevel(2)
    pos = pos.unstack()
    pos.index = pos.index.normalize()
    pos = pos.join(cash).rename_axis({'ending_cash': 'cash'}, axis='columns')

    return pos


def turnover(transactions_df, backtest_data_df, period='M'):
    """
    Calculates the percent absolute value portfolio turnover.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Contains transactional data.
    backtest_data_df : pd.DataFrame
        Contains backtest data, like positions.
    period : str, optional
        Takes the same arguments as df.resample.

    Returns
    -------
    turnoverpct : pd.DataFrame
        The number of shares traded for a period as a fraction of total shares.
    """

    turnover = transactions_df.apply(
        lambda z: z.apply(lambda r: abs(r))).resample(period, 'sum').sum(axis=1)
    portfolio_value = backtest_data_df.portfolio_value.resample(period, 'mean')
    turnoverpct = turnover / portfolio_value
    turnoverpct = turnoverpct.fillna(0)
    return turnoverpct
