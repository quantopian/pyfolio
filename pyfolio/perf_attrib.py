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

import empyrical as ep
import matplotlib.pyplot as plt
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
        Short positions show up as cash in the 'cash' column.
        - Examples:
                        AAPL  TLT  XOM  cash
            2017-01-01    34   58   10     0
            2017-01-02    22   77   18     0
            2017-01-03   -15   27   30    15

                            AAPL       TLT       XOM  cash
            2017-01-01  0.333333  0.568627  0.098039   0.0
            2017-01-02  0.188034  0.658120  0.153846   0.0
            2017-01-03  0.208333  0.375000  0.416667   0.0

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
    tuple of (risk_exposures_portfolio, perf_attribution)

    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

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
    # remove cash after normalizing positions
    del positions['cash']
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

    return (risk_exposures_portfolio,
            pd.concat([perf_attrib_by_factor, returns_df], axis='columns'))


def create_perf_attrib_stats(perf_attrib):
    """
    Takes perf attribution data over a period of time and computes annualized
    multifactor alpha, multifactor sharpe, risk exposures.
    """
    summary = {}
    specific_returns = perf_attrib['specific_returns']
    common_returns = perf_attrib['common_returns']

    summary['Annual multi-factor alpha'] =\
        ep.annual_return(specific_returns)

    summary['Multi-factor sharpe'] =\
        ep.sharpe_ratio(specific_returns)

    summary['Cumulative specific returns'] =\
        ep.cum_returns_final(specific_returns)
    summary['Cumulative common returns'] =\
        ep.cum_returns_final(common_returns)
    summary['Total returns'] =\
        ep.cum_returns_final(perf_attrib['total_returns'])

    summary = pd.Series(summary)
    return summary


def show_perf_attrib_stats(returns, positions, factor_returns,
                           factor_loadings):
    """
    Calls `perf_attrib` using inputs, and displays outputs using
    `utils.print_table`.
    """
    risk_exposures, perf_attrib_data = perf_attrib(returns,
                                                   positions,
                                                   factor_returns,
                                                   factor_loadings)

    perf_attrib_stats = create_perf_attrib_stats(perf_attrib_data)
    print_table(perf_attrib_stats)
    print_table(risk_exposures)


def plot_returns(returns, specific_returns, common_returns, ax=None):
    """
    Plot total, specific, and common returns.
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(returns, color='g', label='Total returns')
    ax.plot(specific_returns, color='b', label='Cumulative specific returns')
    ax.plot(common_returns, color='r', label='Cumulative common returns')

    ax.set_title('Time Series of cumulative returns')
    ax.set_ylabel('Returns')
    ax.legend()

    return ax


def plot_alpha_returns(alpha_returns, ax=None):
    """
    Plot histogram of daily multi-factor alpha returns.
    """
    if ax is None:
        ax = plt.gca()

    ax.hist(alpha_returns, color='g', label='Multi-factor alpha')
    ax.set_title('Histogram of alphas')
    ax.axvline(0, color='k', linestyle='--', label='Zero')

    avg = alpha_returns.mean()
    ax.axvline(avg, color='b', label='Mean = {: 0.5f}'.format(avg))
    ax.legend()

    return ax


def plot_factor_contribution_to_perf(exposures, perf_attrib_data, ax=None):
    """
    Plot each factor's contribution to performance.

    Parameters
    ----------
    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
    """
    if ax is None:
        ax = plt.gca()

    ax.stackplot(perf_attrib_data.index,
                 [perf_attrib_data['factor1'],
                  perf_attrib_data['factor2'],
                  perf_attrib_data['specific_returns']],
                 labels=perf_attrib_data.iloc[:, :-3].columns)

    ax.axhline(0, color='k')
    ax.legend(frameon=True, framealpha=0.5, loc='upper left')

    ax.set_ylabel('Contribution to returns by factor')
    ax.set_title('Returns attribution')

    return ax
