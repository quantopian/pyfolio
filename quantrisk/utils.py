from __future__ import division

import pandas as pd
import json
import zlib
import pandas.io.data as web


def json_to_obj(json):
    return pd.json.loads(str(zlib.decompress(json)))


def one_dec_places(x, pos):
    return '%.1f' % x


def get_symbol_rets(symbol):
    px = web.get_data_yahoo(symbol, start='1/1/1970')
    px = pd.DataFrame.rename(px, columns={'Adj Close': 'AdjClose'})
    px.columns.name = symbol
    rets = px.AdjClose.pct_change().dropna()
    return rets
