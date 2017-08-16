#
# Copyright 2017 Quantopian, Inc.
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
import numpy as np
import matplotlib.pyplot as plt

from pyfolio.utils import print_table


def perf_attrib(factor_loadings,
                factor_returns,
                strategy_daily_returns,
                strategy_daily_holdings):
    """
    Does performance attribution given risk info.

    Parameters
    ----------
    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns
        - Example:
                        momentum  reversal
            2017-01-01  0.002779 -0.005453
            2017-01-02  0.001096  0.010290

    strategy_daily_returns : pd.DataFrame
        Returns for each day in the date range.
        - Example:
                            AAPL       TLT       XOM
            2017-01-01 -0.014092 -0.003938 -0.001508
            2017-01-02  0.011921  0.003039 -0.001546

    strategy_daily_holdings: pd.Series
        Daily holdings for all days in the date range, indexed by date
        and ticker
        - Example:
            dt          ticker
            2017-01-01  AAPL      71
                        TLT       93
                        XOM       10
            2017-01-02  AAPL      71
                        TLT       16
                        XOM       71

    Returns
    -------
    exposures : pd.DataFrame
        df with factors as columns, and datetimes as index
        - Example:
                                 momentum    reversal
            dt         ticker
            2017-01-01 AAPL   -113.096901   60.550926
                       TLT      17.192383   83.284658
                       XOM       9.931600   11.493529
            2017-01-02 AAPL     -9.940674  -37.271562
                       TLT     -17.071643    2.966952
                       XOM    -127.686503   54.070000

    perf_attribution : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980
    """

    risk_exposures = factor_loadings.multiply(strategy_daily_holdings,
                                              axis='rows')
    risk_exposures_portfolio = risk_exposures.groupby(level='dt').sum()
    perf_attrib_style = risk_exposures_portfolio.multiply(factor_returns)

    common_pa = perf_attrib_style.sum(axis=1)
    total_returns_daily = strategy_daily_returns.sum(axis=1)
    specific_pa = total_returns_daily - common_pa

    perf_attrib_common = perf_attrib_style.assign(common_returns=common_pa)
    perf_attrib = perf_attrib_common.assign(specific_returns=specific_pa)

    return strategy_daily_holdings, risk_exposures, perf_attrib


def create_perf_attrib_stats(strategy_daily_holdings, risk_exposures,
                             perf_attrib):
    """
    Takes perf attribution data over a period of time and computes annualized
    alpha, multifactor sharpe, risk exposures.
    """

    summary = {}
    specific_returns = perf_attrib['specific_returns']
    common_returns = perf_attrib['common_returns']

    summary['daily_mf_alpha'] = (
        specific_returns / strategy_daily_holdings.groupby(['dt']).sum()
    ).mean()

    summary['annualized_mf_alpha'] =\
        np.power(summary['daily_mf_alpha'] + 1., 252) - 1.

    summary['mf_sharpe'] = summary['daily_mf_alpha'] * np.sqrt(252) / (
        specific_returns / strategy_daily_holdings.groupby(['dt']).sum()
    ).std()

    summary['total_specific_returns'] = specific_returns.sum()
    summary['total_common_returns'] = common_returns.sum()

    summary['total_returns'] =\
        summary['total_common_returns'] + summary['total_specific_returns']

    summary = pd.Series(summary)
    return risk_exposures, summary


def show_perf_attrib_stats(perf_attrib_data):
    """
    Takes perf attribution data over a period of time, computes stats on it,
    and displays them using `utils.print_table`.
    """
    perf_attrib_stats = create_perf_attrib_stats(perf_attrib_data)
    for df in perf_attrib_stats:
        print_table(df)
