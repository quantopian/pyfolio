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

import pandas as pd


def map_transaction(txn):
    """
    Maps a single transaction row to a dictionary.

    Parameters
    ----------
    txn : pd.DataFrame
        A single transaction object to convert to a dictionary.

    Returns
    -------
    dict
        Mapped transaction.
    """
    # sid can either be just a single value or a SID descriptor
    if isinstance(txn['sid'], dict):
        sid = txn['sid']['sid']
        symbol = txn['sid']['symbol']
    else:
        sid = txn['sid']
        symbol = txn['sid']

    return {'sid': sid,
            'symbol': symbol,
            'price': txn['price'],
            'order_id': txn['order_id'],
            'amount': txn['amount'],
            'commission': txn['commission'],
            'dt': txn['dt']}


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
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    """

    transaction_list = []
    for dt in transactions.index:
        txns = transactions.loc[dt]
        if len(txns) == 0:
            continue

        for txn in txns:
            txn = map_transaction(txn)
            transaction_list.append(txn)
    df = pd.DataFrame(sorted(transaction_list, key=lambda x: x['dt']))
    df['txn_dollars'] = -df['amount'] * df['price']

    df.index = list(map(pd.Timestamp, df.dt.values))
    return df


def get_txn_vol(transactions):
    """Extract daily transaction data from set of transaction objects.

    Parameters
    ----------
    transactions : pd.DataFrame
        Time series containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        price.

    Returns
    -------
    pd.DataFrame
        Daily transaction volume and number of shares.
         - See full explanation in tears.create_full_tear_sheet.
    """
    transactions.index = transactions.index.normalize()
    amounts = transactions.amount.abs()
    prices = transactions.price
    values = amounts * prices
    daily_amounts = amounts.groupby(amounts.index).sum()
    daily_values = values.groupby(values.index).sum()
    daily_amounts.name = "txn_shares"
    daily_values.name = "txn_volume"
    return pd.concat([daily_values, daily_amounts], axis=1)


def adjust_returns_for_slippage(returns, turnover, slippage_bps):
    """Apply a slippage penalty for every dollar traded.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    turnover: pd.Series
        Time series of daily total of buys and sells
        divided by portfolio value.
            - See txn.get_turnover.
    slippage_bps: int/float
        Basis points of slippage to apply.

    Returns
    -------
    pd.Series
        Time series of daily returns, adjusted for slippage.
    """
    slippage = 0.0001 * slippage_bps
    # Only include returns in the period where the algo traded.
    trim_returns = returns.loc[turnover.index]
    return trim_returns - turnover * slippage


def get_turnover(positions, transactions, period=None, average=True):
    """
    Portfolio Turnover Rate:

    Value of purchases and sales divided
    by the average portfolio value for the period.

    If no period is provided the period is one time step.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    period : str, optional
        Takes the same arguments as df.resample.
    average : bool
        if True, return the average of purchases and sales divided
        by portfolio value. If False, return the sum of
        purchases and sales divided by portfolio value.

    Returns
    -------
    turnover_rate : pd.Series
        timeseries of portfolio turnover rates.
    """
    txn_vol = get_txn_vol(transactions)
    traded_value = txn_vol.txn_volume
    portfolio_value = positions.sum(axis=1)
    if period is not None:
        traded_value = traded_value.resample(period).sum()
        portfolio_value = portfolio_value.resample(period).mean()
    # traded_value contains the summed value from buys and sells;
    # this is divided by 2.0 to get the average of the two.
    turnover = traded_value / 2.0 if average else traded_value
    turnover_rate = turnover.div(portfolio_value, axis='index')
    turnover_rate = turnover_rate.fillna(0)
    return turnover_rate
