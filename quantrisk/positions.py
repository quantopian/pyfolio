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


def map_transaction(txn):
    """
    Maps a single transaction row to a dictionary.

    Parameters
    ----------
    txn : pd.DataFrame
        A single transaction object to convert to a dictionary

    Returns
    -------
    dict
        Mapped transaction.
    """

    return {'sid': txn['sid']['sid'],
            'symbol': txn['sid']['symbol'],
            'price': txn['price'],
            'order_id': txn['order_id'],
            'amount': txn['amount'],
            'commission': txn['commission'],
            'dt': txn['dt']}


def pos_dict_to_df(df_pos):
    """
    Converts a dictionary of positions to a DataFrame of positions.

    Parameters
    ----------
    df_pos : dict
        Contains positional information where the indices are datetimes and the values are JSON data.

    Returns
    -------
    pd.DataFrame
        Contains positional information where the indices are datetimes.
    """

    return pd.concat([pd.DataFrame(json.loads(x), index=[dt])
                      for dt, x in df_pos.iteritems()]).fillna(0)


def make_transaction_frame(transactions):
    """
    Formats a transaction DataFrame.

    Parameters
    ----------
    transactions : pd.DataFrame
        Contains improperly formatted transactional data.

    Returns
    -------
    df : pd.DataFrame
        Contains transactions.
    """

    transaction_list = []
    for dt in transactions.index:
        txns = transactions.ix[dt]
        for algo_id in txns.index:
            algo_txns = txns.ix[algo_id]
            for algo_txn in algo_txns:
                txn = map_transaction(algo_txn)
                txn['algo_id'] = algo_id
                transaction_list.append(txn)
    df = pd.DataFrame(sorted(transaction_list, key=lambda x: x['dt']))
    df['txn_dollars'] = df['amount'] * df['price']
    df['date_time_utc'] = map(pd.Timestamp, df.dt.values)

    return df


def get_portfolio_values(positions):
    """
    Determines the net positions of a portfolio's state.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains positional data for a given point in time.

    Returns
    -------
    pd.DataFrame
        Net positional values.
    """

    def get_pos_values(pos):
        position_sizes = {
            i['sid']['symbol']: i['amount'] *
            i['last_sale_price'] for i in pos.pos}
        position_sizes['cash'] = pos.cash
        return json.dumps(position_sizes)

    position_values = positions.apply(get_pos_values, axis='columns')
    return pos_dict_to_df(position_values)


def get_portfolio_alloc(df_pos_vals):
    """
    Determines a portfolio's allocations.

    Parameters
    ----------
    df_pos_vals : pd.DataFrame
        Contains position values or amounts.

    Returns
    -------
    df_pos_alloc : pd.DataFrame
        Positions and their allocations.
    """

    df_pos_alloc = (df_pos_vals.T / df_pos_vals.abs().sum(axis='columns').T).T
    return df_pos_alloc


def get_long_short_pos(df_pos, gross_lev=1.):
    """
    Determines the long amount, short amount, and cash of a portfolio.

    Parameters
    ----------
    df_pos : pd.DataFrame
        The positions that the strategy takes over time.
    gross_lev : float, optional
        The porfolio's gross leverage (default 1).

    Returns
    -------
    df_long_short : pd.DataFrame
        Net long, short, and cash positions.
    """

    df_pos_wo_cash = df_pos.drop('cash', axis='columns')
    df_long = df_pos_wo_cash.apply(lambda x: x[x > 0].sum(), axis='columns')
    df_short = -df_pos_wo_cash.apply(lambda x: x[x < 0].sum(), axis='columns')
    # Shorting positions adds to cash
    df_cash = df_pos.cash.abs() - df_short
    df_long_short = pd.DataFrame({'long': df_long,
                                  'short': df_short,
                                  'cash': df_cash})
    # Renormalize
    df_long_short /= df_long_short.sum(axis='columns')

    # Renormalize to leverage
    df_long_short *= gross_lev

    return df_long_short


def get_top_long_short_abs(df_pos, top=10):
    """
    Finds the top long, short, and absolute positions.

    Parameters
    ----------
    df_pos : pd.DataFrame
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

    df_pos = df_pos.drop('cash', axis='columns')
    df_max = df_pos.max().sort(inplace=False, ascending=False)
    df_min = df_pos.min().sort(inplace=False, ascending=True)
    df_abs_max = df_pos.abs().max().sort(inplace=False, ascending=False)
    df_top_long = df_max[df_max > 0][:top]
    df_top_short = df_min[df_min < 0][:top]
    df_top_abs = df_abs_max[:top]
    return df_top_long, df_top_short, df_top_abs


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
        The number of shares traded for a period as a percentage of the total shares in a portfolio.
    """"
    turnover = transactions_df.apply(
        lambda z: z.apply(lambda r: abs(r))).resample(period, 'sum').sum(axis=1)
    portfolio_value = backtest_data_df.portfolio_value.resample(period, 'mean')
    turnoverpct = turnover / portfolio_value
    turnoverpct = turnoverpct.fillna(0)
    return turnoverpct
