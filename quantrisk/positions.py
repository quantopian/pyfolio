import json

import numpy as np

import pandas as pd


def pos_dict_to_df(df_pos):
    return pd.concat([pd.DataFrame(json.loads(x), index=[dt])
                      for dt, x in df_pos.iteritems()]).fillna(0)


def get_portfolio_alloc(df_pos_vals):
    df_pos_alloc = (df_pos_vals.T / df_pos_vals.abs().sum(axis='columns').T).T
    return df_pos_alloc


def get_portfolio_values(positions):
    def get_pos_values(pos):
        position_sizes = {
            i['sid']['symbol']: i['amount'] *
            i['last_sale_price'] for i in pos.pos}
        position_sizes['cash'] = pos.cash
        return json.dumps(position_sizes)

    position_values = positions.apply(get_pos_values, axis='columns')
    return pos_dict_to_df(position_values)


def get_long_short_pos(df_pos):
    df_pos_wo_cash = df_pos.drop('cash', axis='columns')
    # renormalize
    df_long = df_pos_wo_cash.apply(lambda x: x[x > 0].sum(), axis='columns')
    df_short = -df_pos_wo_cash.apply(lambda x: x[x < 0].sum(), axis='columns')
    df_cash = df_pos.cash.abs()
    df_long_short = pd.DataFrame(
        {'long': df_long, 'short': df_short, 'cash': df_cash})
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
