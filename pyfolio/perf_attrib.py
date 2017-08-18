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

import empyrical
from pyfolio.pos import get_percent_alloc
from pyfolio.utils import print_table


def perf_attrib(returns, positions, factor_returns, factor_loadings):
    """
    Does performance attribution given risk info.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.
        - Example:
            2017-01-01   -0.017098
            2017-01-02    0.002683
            2017-01-03   -0.008669

    positions: pd.DataFrame
        Daily holdings (in dollars or percentages), indexed by date.
        Will be converted to percentages if positions are in dollars.
        - Example:
                        AAPL  TLT  XOM
            2017-01-01    71   93   10
            2017-01-02    71   16   71
            2017-01-03    46   43   63

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns
        - Example:
                        momentum  reversal
            2017-01-01  0.002779 -0.005453
            2017-01-02  0.001096  0.010290

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

    Returns
    -------
    perf_attribution : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980
    """
    # convert holdings to percentages, and convert positions to long format
    positions = get_percent_alloc(positions)
    positions = positions.stack()
    positions.index = positions.index.set_names(['dt', 'ticker'])

    risk_exposures = factor_loadings.multiply(positions,
                                              axis='rows')
    risk_exposures_portfolio = risk_exposures.groupby(level='dt').sum()
    perf_attrib_by_factor = risk_exposures_portfolio.multiply(factor_returns)

    common_returns = perf_attrib_by_factor.sum(axis='columns')
    specific_returns = returns - common_returns

    returns_df = pd.DataFrame({'total_returns': returns,
                               'common_returns': common_returns,
                               'specific_returns': specific_returns})

    return  pd.concat([perf_attrib_by_factor, returns_df], axis='columns')


def create_perf_attrib_stats(perf_attrib):
    """
    Takes perf attribution data over a period of time and computes annualized
    multifactor alpha, multifactor sharpe, risk exposures.
    """
    summary = {}
    specific_returns = perf_attrib['specific_returns']
    common_returns = perf_attrib['common_returns']

    summary['Annual multi-factor alpha'] =\
        empyrical.annual_return(specific_returns)

    summary['Multi-factor sharpe'] =\
        empyrical.sharpe_ratio(specific_returns)

    summary['Cumulative specific returns'] =\
        empyrical.cum_returns(specific_returns)
    summary['Cumulative common returns'] =\
        empyrical.cum_returns(common_returns)
    summary['Total returns'] =\
        empyrical.cum_returns(perf_attrib['total_returns'])

    summary = pd.Series(summary)
    return summary


def show_perf_attrib_stats(perf_attrib_data, risk_exposures):
    """
    Takes perf attribution data over a period of time, computes stats on it,
    and displays them using `utils.print_table`.
    """
    perf_attrib_stats = create_perf_attrib_stats(perf_attrib_data)
    print_table(perf_attrib_stats)
    print_table(risk_exposures)
