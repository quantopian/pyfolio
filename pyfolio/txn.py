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
from collections import defaultdict

import pandas as pd


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
    # sid can either be just a single value or a SID descriptor
    if isinstance(txn['sid'], dict):
        sid = txn['sid']['sid']
        symbol = txn['sid']['symbol']
    else:
        sid = txn['sid']
        symbol = None

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
        Contains transactions.
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
    df['txn_dollars'] = df['amount'] * df['price']

    df.index = list(map(pd.Timestamp, df.dt.values))
    return df


def get_txn_vol(transactions):
    """Extract transaction data from

    Parameters
    ----------
    transactions : pd.DataFrame
        timeseries containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        price.

    Returns
    -------
    pd.DataFrame
        Daily transaction volume and number of shares.
    """

    txn_vol = transactions.reset_index().groupby('index').apply(
        lambda ser: (
            ser['amount'].abs() *
            ser['price']).sum())
    txn_amount = transactions.reset_index().groupby(
        'index')['amount'].apply(lambda ser: ser.abs().sum())
    transactions_out = pd.concat([txn_vol, txn_amount], axis=1)
    transactions_out.columns = ['txn_volume', 'txn_shares']
    transactions_out.index = transactions_out.index.normalize()

    return transactions_out


def create_txn_profits(transactions):
    """
    Compute per-trade profits.

    Generates a new transactions DataFrame with a profits column

    Parameters
    ----------
    transactions : pd.DataFrame
        A strategy's transactions.
        See pos.make_transaction_frame(transactions).

    Returns
    -------
    profits_dts : pd.DataFrame
        DataFrame containing transactions and their profits, datetimes,
        amounts, current prices, prior prices, and symbols.
    """

    txn_descr = defaultdict(list)

    for symbol, transactions_sym in transactions.groupby('symbol'):
        transactions_sym = transactions_sym.reset_index()

        for i, (amount, price, dt) in transactions_sym.iloc[1:][
                ['amount', 'price', 'date_time_utc']].iterrows():
            prev_amount, prev_price, prev_dt = transactions_sym.loc[
                i - 1, ['amount', 'price', 'date_time_utc']]
            profit = (price - prev_price) * -amount
            txn_descr['profits'].append(profit)
            txn_descr['dts'].append(dt - prev_dt)
            txn_descr['amounts'].append(amount)
            txn_descr['prices'].append(price)
            txn_descr['prev_prices'].append(prev_price)
            txn_descr['symbols'].append(symbol)

    profits_dts = pd.DataFrame(txn_descr)

    return profits_dts
