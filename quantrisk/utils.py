from __future__ import division

import pandas as pd
import json
import zlib


def json_to_obj(json):
    return pd.json.loads(str(zlib.decompress(json)))


def one_dec_places(x, pos):
    return '%.1f' % x
