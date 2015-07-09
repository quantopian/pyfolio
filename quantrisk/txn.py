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


def create_txn_profits(df_txn):
    """
    Compute per-trade profits.

    Generates a new transactions DataFrame with a profits column

    Parameters
    ----------
    df_txn : pd.DataFrame
        A strategy's transactions. See positions.make_transaction_frame(df_txn).

    Returns
    -------
    profits_dts : pd.DataFrame
        DataFrame containing transactions and their profits, datetimes, amounts, current prices, prior prices, and symbols.
    """

    txn_descr = defaultdict(list)

    for symbol, df_txn_sym in df_txn.groupby('symbol'):
        df_txn_sym = df_txn_sym.reset_index()

        for i, (amount, price, dt) in df_txn_sym.iloc[1:][['amount', 'price', 'date_time_utc']].iterrows():
            prev_amount, prev_price, prev_dt = df_txn_sym.loc[
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
