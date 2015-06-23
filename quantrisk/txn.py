from collections import defaultdict


def create_txn_profits(df_txn):
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
