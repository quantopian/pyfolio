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

import empyrical as ep
import pandas as pd

import matplotlib.pyplot as plt
from pyfolio.pos import get_percent_alloc
from pyfolio.utils import print_table, set_legend_location, COLORS


def perf_attrib(returns, positions, factor_returns, factor_loadings,
                pos_in_dollars=True):
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

    pos_in_dollars : bool
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

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
    if pos_in_dollars:
        # convert holdings to percentages
        positions = get_percent_alloc(positions)

    # remove cash after normalizing positions
    positions = positions.drop('cash', axis='columns')

    # convert positions to long format
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


def perf_attrib_periods(returns, positions, factor_returns, factor_loadings,
                        pos_in_dollars=True, periods=None):
    """
    Calls `perf_attrib` on subsections of the parameters.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.

    positions: pd.DataFrame
        Daily holdings (in dollars or percentages), indexed by date.
        Will be converted to percentages if positions are in dollars.
        Short positions show up as cash in the 'cash' column.

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.

    pos_in_dollars : bool
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

    periods : dict of (str, datetime)
        Dict containing datetimes that split up the performance and risk data
        into time periods, so that performance attribution is done separately
        on each of the time periods.

        Has time period names as keys, and time period start dates as values.

        One of the values must be the first date in the date range.
        - Example:
            Assuming the date range is 2017-01-01 to 2017-02-01:
            {'time_period1': '2017-01-01',
             'time_period2': '2017-01-05',
             'time_period3': '2017-01-20'}

    Returns
    -------
    perf_attrib_by_period : dict
        dict keyed by the same keys as `periods`, where the values are the
        output of `perf_attrib`, i.e., a tuple of
        (risk_exposures_portfolio, perf_attribution).
    """
    if periods is None:
        return perf_attrib(returns, positions, factor_returns, factor_loadings,
                           pos_in_dollars=pos_in_dollars)

    # have to sort dict to know how periods are organized
    periods_sorted = sorted(periods, key=periods.get, reverse=True)

    perf_attrib_by_period = {}
    last_period_start = None

    for period in periods_sorted:

        period_start = periods[period]
        period_returns = returns[period_start:last_period_start]
        period_positions = positions[period_start:last_period_start]

        period_factor_returns =\
            factor_returns[period_start:last_period_start]

        period_factor_loadings =\
            factor_loadings[period_start:last_period_start]

        perf_attrib_by_period[period] = perf_attrib(period_returns,
                                                    period_positions,
                                                    period_factor_returns,
                                                    period_factor_loadings,
                                                    pos_in_dollars)

        last_period_start = period_start

    return perf_attrib_by_period


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

    Parameters
    ----------
    returns : pd.Series
        total returns, indexed by datetime

    specific_returns : pd.Series
        specific returns, indexed by datetime

    commons_returns : pd.Series
        common returns, indexed by datetime

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    ax.plot(ep.cum_returns(returns), color='g', label='Total returns')
    ax.plot(ep.cum_returns(specific_returns), color='b',
            label='Cumulative specific returns')
    ax.plot(ep.cum_returns(common_returns), color='r',
            label='Cumulative common returns')

    ax.set_title('Time Series of cumulative returns')
    ax.set_ylabel('Returns')

    set_legend_location(ax)

    return ax


def plot_alpha_returns(alpha_returns, ax=None):
    """
    Plot histogram of daily multi-factor alpha returns (specific returns).

    Parameters
    ----------
    alpha_returns : pd.Series
        series of daily alpha returns indexed by datetime

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    ax.hist(alpha_returns, color='g', label='Multi-factor alpha')
    ax.set_title('Histogram of alphas')
    ax.axvline(0, color='k', linestyle='--', label='Zero')

    avg = alpha_returns.mean()
    ax.axvline(avg, color='b', label='Mean = {: 0.5f}'.format(avg))
    set_legend_location(ax)

    return ax


def plot_factor_contribution_to_perf(exposures, perf_attrib_data, ax=None):
    """
    Plot each factor's contribution to performance.

    Parameters
    ----------
    exposures : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    factors_and_specific = perf_attrib_data.drop(
        ['total_returns', 'common_returns'], axis='columns')

    ax.stackplot(
        factors_and_specific.index,
        [factors_and_specific[s] for s in factors_and_specific],
        labels=factors_and_specific.columns,
        colors=COLORS
    )

    ax.axhline(0, color='k')
    set_legend_location(ax)

    ax.set_ylabel('Contribution to returns by factor')
    ax.set_title('Returns attribution')

    return ax


def plot_risk_exposures(exposures, ax=None):
    """
    Parameters
    ----------
    exposures : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    ax.stackplot(exposures.index,
                 [exposures[s] for s in exposures],
                 labels=exposures.columns,
                 colors=COLORS)

    set_legend_location(ax)
    ax.set_ylabel('Factor exposures')
    ax.set_title('Risk factor exposures')

    return ax
