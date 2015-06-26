import json
import pandas as pd
import numpy as np


def map_transaction(txn):
    return {'sid': txn['sid']['sid'],
            'symbol': txn['sid']['symbol'],
            'price': txn['price'],
            'order_id': txn['order_id'],
            'amount': txn['amount'],
            'commission': txn['commission'],
            'dt': txn['dt']}


def pos_dict_to_df(df_pos):
    return pd.concat([pd.DataFrame(json.loads(x), index=[dt])
                      for dt, x in df_pos.iteritems()]).fillna(0)


def make_transaction_frame(transactions):
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
    def get_pos_values(pos):
        position_sizes = {
            i['sid']['symbol']: i['amount'] *
            i['last_sale_price'] for i in pos.pos}
        position_sizes['cash'] = pos.cash
        return json.dumps(position_sizes)

    position_values = positions.apply(get_pos_values, axis='columns')
    return pos_dict_to_df(position_values)


def get_portfolio_alloc(df_pos_vals):
    df_pos_alloc = (df_pos_vals.T / df_pos_vals.abs().sum(axis='columns').T).T
    return df_pos_alloc


def get_long_short_pos(df_pos):
    df_pos_wo_cash = df_pos.drop('cash', axis='columns')
    df_long = df_pos_wo_cash.apply(lambda x: x[x > 0].sum(), axis='columns')
    df_short = -df_pos_wo_cash.apply(lambda x: x[x < 0].sum(), axis='columns')
    # Shorting positions adds to cash
    df_cash = df_pos.cash.abs() - df_short
    df_long_short = pd.DataFrame({'long': df_long,
                                  'short': df_short,
                                  'cash': df_cash})
    return df_long_short


def get_top_long_short_abs(df_pos, top=10):
    df_pos = df_pos.drop('cash', axis='columns')
    df_max = df_pos.max().sort(inplace=False, ascending=False)
    df_min = df_pos.min().sort(inplace=False, ascending=True)
    df_abs_max = df_pos.abs().max().sort(inplace=False, ascending=False)
    df_top_long = df_max[df_max > 0][:top]
    df_top_short = df_min[df_min < 0][:top]
    df_top_abs = df_abs_max[:top]
    return df_top_long, df_top_short, df_top_abs


def turnover(transactions_df, backtest_data_df, period='M'):
    # Calculates the percent absolute value portfolio turnover
    # transactions_df and backtest_data_df can come straight from the test harness
    # period takes the same arguments as df.resample
    turnover = transactions_df.apply(
        lambda z: z.apply(
            lambda r: abs(r))).resample(
        period,
        'sum').sum(
                axis=1)
    portfolio_value = backtest_data_df.portfolio_value.resample(period, 'mean')
    turnoverpct = turnover / portfolio_value
    turnoverpct = turnoverpct.fillna(0)
    return turnoverpct

def get_all_tickers_traded(
                        algo_id, contest=None, 
                        backtest_min_years=None, 
                        backtest_max_years=None):
    
    engine_harness = sqlalchemy.create_engine(host_settings.SQLTESTHARNESS, echo=False)
    
    try:
        df_rets, df_pos, df_txn_daily, fetcher_urls = get_single_algo(algo_id, 
                                                                      minyears=backtest_min_years,
                                                                      maxyears=backtest_max_years,         
                                                                      engine=engine_harness, 
                                                                      contest=contest)
    except:
        return np.array([], 'object')

    df_pos_alloc = get_portfolio_alloc(df_pos)
    
    _, _, df_top_abs_all = get_top_long_short_abs(df_pos_alloc, top=1000)
 
    return df_top_abs_all.index.values