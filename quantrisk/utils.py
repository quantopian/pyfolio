from __future__ import division

from quant_utils.timeseries import *

import pandas as pd
import numpy as np
import seaborn as sns
from collections import *
from operator import *
import string

import zipline
from zipline.finance.risk import RiskMetricsCumulative, RiskMetricsPeriod
from zipline.utils.factory import create_simulation_parameters
from zipline.utils import tradingcalendar

import pandas.io.data as web
import statsmodels.tsa.stattools as ts

import datetime
from datetime import datetime
from datetime import timedelta
import pytz
import time

from bson import ObjectId

import os
import zlib

utc=pytz.UTC
indexTradingCal = pd.DatetimeIndex(tradingcalendar.trading_days)
indexTradingCal = indexTradingCal.normalize()

def flatten_list2d(list2d):
    return [ x for y in list2d for x in y ]

def flatten_list(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten_list(el))
        else:
            result.append(el)
    return result

def string_arr_to_single_str(strings_arr, sep_char=",", prefix_char="'", suffix_char="'"):
    # EXAMPLE USAGE:
    # string_arr_to_single_str(np.array(['54c724b0a1dbc627570003e2','54c4c2cf0b872f3229000225','54c0f94a10ee581094000186']))
    res_str = ""
    for i in range(len(strings_arr)):
        if i > 0:
            res_str = res_str + sep_char + prefix_char + strings_arr[i] + suffix_char
        else:
            res_str = prefix_char + strings_arr[i] + suffix_char

    return res_str

def determine_next_filename(primary_filename, filename_ext='', input_file_list=None):
    if input_file_list is None:
        file_list = os.listdir('.')
    else:
        file_list = input_file_list

    file_version = 1
    while file_version > 0:
        temp_filename = primary_filename + '_' + str(file_version) + filename_ext
        # print(temp_filename)
        if np.intersect1d( file_list, [temp_filename] ).size > 0:
            file_version = file_version + 1
        else:
            return temp_filename

def unionTupleElements(arrayOfTuples):
    tempUnion = []
    for i in range(0,arrayOfTuples.size):
        tempUnion = np.union1d(tempUnion, arrayOfTuples[i])

    return tempUnion

def keep_keys( in_dict, keys_to_keep ):
    temp_dict = {}
    for i in keys_to_keep:
        temp_dict[i] = in_dict.get(i)

    return temp_dict
