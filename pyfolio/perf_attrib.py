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
from collections import OrderedDict


# 31 visually distinct colors... http://phrogz.net/css/distinct-colors.html
# There are so many factors to plot (26 so far) that using a matplotlib
# colormap is unsatisfactory: either too few colors, or too similar colors
COLORS = [
    '#f23d3d', '#828c23', '#698c83', '#594080', '#994d4d',
    '#206380', '#dd39e6', '#cc9999', '#7c8060', '#66adcc',
    '#6c7dd9', '#8a698c', '#7f6340', '#66cc7a', '#a3abd9',
    '#d9c0a3', '#bfffcc', '#542699', '#b35986', '#d4e639',
    '#b380ff', '#e0e6ac', '#a253a6', '#418020', '#ff409f',
    '#ffa940', '#83ff40', '#3d58f2', '#e3ace6', '#d9a86c',
    '#2db391'
]


def perf_attrib(factor_loadings_long,
                resid_var_long,
                covariances_long,
                factor_returns_long,
                holdings,
                pnls,
                holdings_pnls,
                aums,
                date_range):
    '''
    Iteratively calls perf_attrib_1d to performs performance attribution over
    a date range. This function takes risk model data in long format (a.k.a
    tidy data), whereas it takes portfolio data in wide format. For more
    information, see Hadley Wickham's 2014 paper:
    http://vita.had.co.nz/papers/tidy-data.html

    Parameters
    ----------
    factor_loadings_long : pd.DataFrame
        Factor loadings for all days in the date range, in long (tidy) format
        - Example:
                 dt             sid   name        family         factor_loading
            0    2017-06-08	    24	  technology  sector	     0.9
            1    2017-06-08	    24	  materials	  sector	     0.0
            2    2017-06-08	    24	  momentum	  style	         0.5
            3    2017-06-08	    24	  stat_1	  statistical	 0.1

    resid_var_long : pd.DataFrame
        Stock specific variances for all days in the date range, for all
        stocks, in long (tidy) format
        - Example:
                dt          sid	    family	   residual	    variance
            0	2015-06-03	2	    sector	   0.002282	    0.000231
            1	2015-06-03	2	    style	   -0.006196	0.000216
            2	2015-06-03	2	    pca	       -0.005679	0.000205
            3	2015-06-03	24	    sector	   0.006446	    0.000231

    covariances_long : pd.DataFrame
        Factor covariances for all days in the date range, in long (tidy)
        format. Note that there is duplication: i.e. the covariance between
        common factors X and Y are included twice: once with X as primary and Y
        as secondary, and vice versa.
        - Example:
                dt           primary    secondary      covariance
            0   2017-06-08	 momentum	momentum	   0.1
            1   2017-06-08	 momentum	reversal	   0.2
            2   2017-06-08	 momentum	stat_1	       -.05
            3   2017-06-08	 reversal	momentum	   0.2
            4   2017-06-08	 reversal	reversal	   0.0

    factor_returns_long : pd.DataFrame
        Common factor returns for all days in the date range, in long (tidy)
        format.
        - Example:
                         dt      factor         returns
            0    2017-06-08      technology     0.01
            1    2017-06-08      momentum       -0.03
            2    2017-06-08      stat_1         -0.2

    holdings : pd.DataFrame
        Dollar value of positions in each stock, per day, in wide format.
        - Indexed by dates, columns are sids
        - Example:
                            2           24          35
            2017-06-08      103.19      18319.17    9913.01
            2017-06-09      221.01      26301.90    5510.13
            2017-06-10      -331.01     24105.36    -120.97

    pnls : pd.Series
        PnL per day
        - PnL as values, indexed by dates
        - Example:
            2017-06-08      10183.19
            2017-06-09      -9371.01
            2017-06-10      6312.38

    aums : pd.Series
        Assets under management per day
        - AUMs as values, indexed by dates
        - Example:
            2017-06-08      1.0183e6
            2017-06-09      1.0238e6
            2017-06-10      1.0192e6

    date_range : pd.DatetimeIndex
        Range of dates over which performance attribution is to be done

    Returns
    -------
    perf_attrib_dict : OrderedDict
        OrderedDict containing performance attribution metrics across all dates
        - pd.Timestamps as keys, OrderedDicts as values.
        - These OrderedDicts (that is, the values of perf_attrib_dict) have
            have strings as keys and performance attribution metrics as values
    '''

    perf_attrib_dict = OrderedDict([])

    factor_loadings_long.dt = pd.to_datetime(factor_loadings_long.dt)
    covariances_long.dt = pd.to_datetime(covariances_long.dt)
    resid_var_long.dt = pd.to_datetime(resid_var_long.dt)
    factor_returns_long.dt = pd.to_datetime(factor_returns_long.dt)

    for date in date_range:
        mask = (factor_loadings_long.dt == date)
        factor_loadings_wide = factor_loadings_long[mask] \
            .drop(['dt', 'family'], axis='columns') \
            .set_index(['sid', 'name']).unstack()
        factor_loadings_wide.index.name = None
        factor_loadings_wide.columns = factor_loadings_wide.columns.droplevel()
        factor_loadings_wide.columns.name = None

        mask = (covariances_long.dt == date)
        covariances_wide = covariances_long[mask].drop('dt', axis='columns') \
            .set_index(['primary', 'secondary']).unstack()
        covariances_wide.index.name = None
        covariances_wide.columns = covariances_wide.columns.droplevel()
        covariances_wide.columns.name = None

        # FORMAT resid_var_long... SORT OUT WHY THERE ARE DUPES.
        mask = ((resid_var_long.dt == date) & (resid_var_long.family == 'pca'))
        stock_specific_variances_wide = resid_var_long[mask] \
            .drop(['dt', 'family', 'residual'], axis='columns') \
            .sort_values('sid')

        mask = (factor_returns_long.dt == date)
        factor_returns_wide = factor_returns_long[mask] \
            .drop('dt', axis='columns').set_index('factor').squeeze()
        factor_returns_wide.name = None

        to_update = perf_attrib_1d(factor_loadings_wide,
                                   covariances_wide,
                                   stock_specific_variances_wide,
                                   factor_returns_wide,
                                   holdings[date],
                                   pnls[date],
                                   holdings_pnls[date],
                                   aums[date])

        perf_attrib_dict.update(to_update)

    return perf_attrib_dict


def perf_attrib_1d(factor_loadings_1d,
                   factor_covariances_1d,
                   stock_specific_variances_1d,
                   factor_returns_1d,
                   holdings_1d,
                   pnl_1d,
                   holdings_pnl_1d,
                   aum_1d):
    '''
    Performs performance attribution for a given day.

    Parameters
    ----------
    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common factors for the given day
        - Columns are common factors, indexed by sids
        - Example:
                momentum	size	     value
            2   0.185313    0.916532     0.174353
            24	1.791453    -1.97424     1.016321
            41  -0.14235    1.129351     -2.05923

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - Square matrix with both columns and index being common factors
        - Example:
                       momentum    size	        value
            momentum   0.000313    0.009123     0.000353
            size       0.000093    0.014261     0.000321
            value      0.000012    0.001012     0.000093

    stock_specific_variances_1d : pd.DataFrame
        Stock specific variances
        - Diagonal square matrix with both columns and index being sids
        - Example:
                  2        	  24	       41
            2     0.000383    0.000000     0.000000
            24    0.000000    0.007241     0.000000
            41    0.000000    0.000000     0.000498

    factor_returns_1d : pd.Series
        Returns associated with common factors for the given day
        - Returns, indexed by common factor
        - Example:
            momentum   0.002313
            size       -0.009314
            value      0.012018

    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - Dollar value of positions, indexed by sid
        - Example:
            2           -631.93
            24          9815.19
            41          0.00

    pnl_1d : float
        Total PnL of the algorithm for the given day

    holdings_pnl_1d : float
        Fraction of specific PnL due to holdings
        - Exact output of compute_trading_pnl_1d

    aum_1d : float
        Total assets under management by the algorithm for the given day

    Returns
    -------
    perf_attrib_entry : OrderedDict
        OrderedDict containing performance attribution metrics
        - Keys are strings (names of the performance attribution metric), and
            values are the performance attribution metrics
    '''
    # There may be stocks in our holdings that are not in the risk model.
    # Record them and drop them from our holdings.
    not_in_risk_model = []
    for stock in holdings_1d.index:
        if stock not in stock_specific_variances_1d.index:
            not_in_risk_model.append(stock)
    holdings_1d.drop(not_in_risk_model, inplace=True)

    # There may be stocks in the risk model that are not in our holdings.
    # Add them, holding 0
    not_in_portfolio = []
    for stock in stock_specific_variances_1d.index:
        if stock not in holdings_1d.index:
            not_in_portfolio.append(stock)
    to_add = pd.Series(np.zeros(len(not_in_portfolio)), index=not_in_portfolio)
    holdings_1d = holdings_1d.append(to_add)

    # Finally, if there are NaNs anywhere, this means we have no position in
    # that stock
    holdings_1d.replace(np.nan, 0, inplace=True)
    factor_loadings_1d.replace(np.nan, 0, inplace=True)

    # Now we proceed with performance attribution
    exposures_1d = compute_common_factor_exposures_1d(holdings_1d,
                                                      factor_loadings_1d)

    vol_weighted_exposures_1d = \
        compute_vol_weighted_common_factor_exposures_1d(exposures_1d,
                                                        factor_covariances_1d)

    common_factor_pnls_1d = compute_common_factor_pnls_1d(exposures_1d,
                                                          factor_returns_1d)

    specific_pnl_1d = compute_specific_pnl_1d(pnl_1d, common_factor_pnls_1d)

    holdings_pnl_1d = compute_holdings_pnl_1d()

    trading_pnl_1d = compute_trading_pnl_1d(specific_pnl_1d, holdings_pnl_1d)

    common_factor_var_1d, specific_var_1d, portfolio_var_1d = \
        compute_variances_1d(holdings_1d, factor_loadings_1d,
                             factor_covariances_1d,
                             stock_specific_variances_1d)

    proportion_specific_1d = specific_var_1d / portfolio_var_1d

    MCR_common_factor_1d, MCR_specific_1d, MCR_portfolio_1d = \
        compute_marginal_contributions_to_risk_1d(holdings_1d,
                                                  factor_loadings_1d,
                                                  factor_covariances_1d,
                                                  stock_specific_variances_1d)

    perf_attrib_entry = OrderedDict([
        ('total pnl', pnl_1d),
        ('factor exposure', exposures_1d),
        ('vol weighted factor exposure', vol_weighted_exposures_1d),
        ('AUM', aum_1d),
        ('common factor pnls', common_factor_pnls_1d),
        ('specific pnl', specific_pnl_1d),
        ('holdings pnl', holdings_pnl_1d),
        ('trading pnl', trading_pnl_1d),
        ('common factor risk', np.sqrt(common_factor_var_1d)),
        ('specific risk', np.sqrt(specific_var_1d)),
        ('portfolio risk', np.sqrt(portfolio_var_1d)),
        ('proportion specific', proportion_specific_1d),
        ('MCR common factor', MCR_common_factor_1d),
        ('MCR specific', MCR_specific_1d),
        ('MCR portfolio', MCR_portfolio_1d)
    ])

    return perf_attrib_entry


def compute_common_factor_exposures_1d(holdings_1d, factor_loadings_1d):
    '''
    Computes dollar common factor exposures for a given day

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common factors for the given day
        - See full explanation in perf_attrib_1d

    Returns
    -------
    exposures_1d : pd.Series
        Common factor exposures for the given day.
        - Common factor exposures, indexed by common factor
    '''
    exposures_1d = holdings_1d.dot(factor_loadings_1d)
    return exposures_1d


def plot_common_factor_exposures(exposures, ax=None):
    '''
    Plots time series of common factor exposures as a stack plot

    Parameters
    ----------
    exposures : pd.DataFrame
        Time series of dollar common factor exposures
        - Columns are common factors, index is datetime
        - The output of compute_common_factor_exposures_1d is only one row of
        this DataFrame
        - Example:
                            momentum	    size           	value
            2017-06-01	    69183.823143	3.919257e+05	1.412135e+06
            2017-06-02	    74165.961984	4.768590e+05	1.513102e+06

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    pos_exposures = exposures.copy()
    neg_exposures = exposures.copy()
    pos_exposures[pos_exposures < 0] = 0
    neg_exposures[neg_exposures > 0] = 0

    pos_plot = []
    neg_plot = []
    for i in range(len(exposures.columns)):
        pos_plot.append(pos_exposures.iloc[:, i].values)
        neg_plot.append(neg_exposures.iloc[:, i].values)

    ax.stackplot(exposures.index, pos_plot, colors=COLORS, alpha=0.8,
                 labels=pos_exposures.columns)
    ax.stackplot(exposures.index, neg_plot, colors=COLORS, alpha=0.8)
    ax.axhline(0, color='k')
    ax.legend(loc=2, frameon=True)
    ax.set_ylabel('Exposure ($)')
    ax.set_title('Risk Factor Exposures', fontsize='large')

    return ax


def compute_vol_weighted_common_factor_exposures_1d(exposures_1d,
                                                    factor_covariances_1d):
    '''
    Computes volatility-weighted dollar common factor exposures for a given day

    Parameters
    ----------
    exposures_1d : pd.Series
        Risk factor exposures of the portfolio for the given day
        - Exact output of compute_common_factor_exposures_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d

    Returns
    -------
    vol_weighted_exposures_1d : pd.Series
        Volatility-weighted common factor exposures for the given day.
        - Volatility-weighted common factor exposures, indexed by common factor
    '''
    vol = pd.Series(data=np.diag(factor_covariances_1d),
                    index=exposures_1d.index)
    vol_weighted_exposures_1d = exposures_1d.multiply(np.sqrt(vol))
    return vol_weighted_exposures_1d


def plot_vol_weighted_common_factor_exposures(vol_weighted_exposures, ax=None):
    '''
    Plots time series of volatility-weighted common factor exposures as a stack
    plot

    Parameters
    ----------
    vol_weighted_exposures : pd.DataFrame
        Time series of volatility-weighted dollar common factor exposures
        - Columns are common factors, index is datetime
        - The output of compute_vol_weighted_common_factor_exposures_1d is only
        one row of this DataFrame
        - Example:
                            momentum	    size           	value
            2017-06-01	    6083.823143	    9192.538167	    1421.304375
            2017-06-02	    7125.961984	    7685.951230	    1131.029048

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    pos_vol_weighted_exposures = vol_weighted_exposures.copy()
    neg_vol_weighted_exposures = vol_weighted_exposures.copy()
    pos_vol_weighted_exposures[pos_vol_weighted_exposures < 0] = 0
    neg_vol_weighted_exposures[neg_vol_weighted_exposures > 0] = 0

    pos_plot = []
    neg_plot = []
    for i in range(len(vol_weighted_exposures.columns)):
        pos_plot.append(pos_vol_weighted_exposures.iloc[:, i].values)
        neg_plot.append(neg_vol_weighted_exposures.iloc[:, i].values)

    ax.stackplot(vol_weighted_exposures.index, pos_plot, colors=COLORS,
                 alpha=0.8, labels=pos_vol_weighted_exposures.columns)
    ax.stackplot(vol_weighted_exposures.index, neg_plot, colors=COLORS,
                 alpha=0.8)
    ax.axhline(0, color='k')
    ax.legend(loc=2, frameon=True)
    ax.set_ylabel('Vol-Weighted Exposure ($)')
    ax.set_title('Volatility Weighted Risk Factor Exposures', fontsize='large')

    return ax


def compute_common_factor_pnls_1d(exposures_1d, factor_returns_1d):
    '''
    Computes PnL due to common factors for a given day

    Parameters
    ----------
    exposures_1d : pd.Series
        Risk factor exposures of the portfolio for the given day
        - Exact output of compute_common_factor_exposures_1d

    factor_returns_1d : pd.Series
        Returns associated with common factors for the given day
        - See full explanation in perf_attrib_1d

    Returns
    -------
    common_factor_pnls_1d : pd.Series
        PnL attributable to common factors for the given day
        - PnL indexed by common factor
    '''
    common_factor_pnls_1d = exposures_1d.multiply(factor_returns_1d)
    return common_factor_pnls_1d


def compute_specific_pnl_1d(pnl_1d, common_factor_pnls_1d):
    '''
    Computes PnL that is not due to common factors for a given day

    Parameters
    ----------
    pnl_1d : float
        Total PnL for the given day

    common_factor_pnls_1d : pd.Series
        PnL due to common factors for the given day
        - Exact output of compute_common_factor_pnls_1d

    Returns
    -------
    specific_pnl_1d : float
        PnL not attributable to any common factors for the given day
    '''
    specific_pnl_1d = pnl_1d - common_factor_pnls_1d.sum()
    return specific_pnl_1d


def plot_pnl_attribution(pnls, ax=None):
    '''
    Plots time series of common factor and specific PnL

    Parameters
    ----------
    pnls : pd.DataFrame
        Time series of PnL attributable to common factors, and specific (non-
        attributable) PnL
        - Columns are common factor and specific PnL, index is datetime
        - The output of compute_specific_pnl_1d is only one cell of the last
        column of this DataFrame
        - The output of compute_common_factor_pnls_1d is only one row of this
        DataFrame (ignoring the last cell)
        - Example:
                          technology	momentum      	specific
            2017-06-01	  6083.823143	-9192.538167	-1475.038534
            2017-06-02	  7125.961984	-7685.951230	-1715.340134

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    pos_pnls = pnls.copy()
    neg_pnls = pnls.copy()
    pos_pnls[pos_pnls < 0] = 0
    neg_pnls[neg_pnls > 0] = 0
    tot_pnl = pnls.sum(axis='columns')

    pos_plot = []
    neg_plot = []
    for i in range(len(pnls.columns)):
        pos_plot.append(pos_pnls.iloc[:, i].values)
        neg_plot.append(neg_pnls.iloc[:, i].values)

    ax.stackplot(pnls.index, pos_plot, colors=COLORS, alpha=0.8,
                 labels=pos_pnls.columns)
    ax.stackplot(pnls.index, neg_plot, colors=COLORS, alpha=0.8)
    ax.plot(pnls.index, tot_pnl, color='r', linestyle='--', label='total_pnl')
    ax.axhline(0, color='k')
    ax.legend(frameon=True)
    ax.set_ylabel('PnL ($)')
    ax.set_title('PnL Attribution', fontsize='large')

    return ax


def plot_gross_pnl_attribution(pnls, ax=None):
    '''
    Plots time series of common factor and specific PnL, normalized to its
    total contribution to daily PnL, as a stack plot

    Parameters
    ----------
    pnls : pd.DataFrame
        Time series of PnL attributable to common factors, and specific (non-
        attributable) PnL
        - See full description in plot_pnl_attribution

    ax : plt.Axes
        Axes on which to plot
    '''
    abs_pnls = pnls.abs()

    gross_plot = []
    tot_abs_pnls = abs_pnls.sum(axis=1)
    for i in range(len(abs_pnls.columns)):
        gross_plot.append(abs_pnls.iloc[:, i].divide(tot_abs_pnls).values)

    ax.stackplot(abs_pnls.index, gross_plot, colors=COLORS, alpha=0.8,
                 labels=abs_pnls.columns)
    ax.axhline(0, color='k')
    ax.legend(frameon=True, loc=2)
    ax.set_ylabel('Contribution to Gross PnL ($)')
    ax.set_title('Gross PnL Attribution', fontsize='large')

    return ax


def plot_pnl_time_series(common_factor_pnl, specific_pnl, ax=None):
    '''
    Plots time series of common factor pnl, specific pnl and portfolio pnl as
    a line graph

    Parameters
    ----------
    common_factor_pnl : pd.Series
        Time series of total PnL attributable to common risk factors
        - PnL indexed by datetime
        - Example:
            2017-06-05   -14715.038534
            2017-06-06    11774.696823
            2017-06-07    -6778.749595

    specific_pnl : pd.Series
        Time series of PnL not attributable to common risk factors
        - PnL indexed by datetime
        - common_factor_pnl and specific_pnl should sum to total PnL
        - Example:
            2017-06-05    -8213.041651
            2017-06-06       74.696823
            2017-06-07     -778.749595

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    tot_pnl = common_factor_pnl + specific_pnl

    ax.plot(specific_pnl, color='b', label='specific pnl')
    ax.plot(common_factor_pnl, color='r', label='common factor pnl')
    ax.plot(tot_pnl, color='g', label='total pnl')
    ax.set_title('Time Series of PnL')
    ax.set_ylabel('PnL ($)')
    ax.legend()

    return ax


def plot_cum_pnl_time_series(common_factor_pnl, specific_pnl, ax=None):
    '''
    Plots cumulative time series of common factor pnl, specific pnl and
    portfolio pnl as a line graph

    Parameters
    ----------
    common_factor_pnl : pd.Series
        Time series of total PnL attributable to common risk factors
        - PnL indexed by datetime
        - Example:
            2017-06-05   -14715.038534
            2017-06-06    11774.696823
            2017-06-07    -6778.749595

    specific_pnl : pd.Series
        Time series of PnL not attributable to common risk factors
        - PnL indexed by datetime
        - common_factor_pnl and specific_pnl should sum to total PnL
        - Example:
            2017-06-05    -8213.041651
            2017-06-06       74.696823
            2017-06-07     -778.749595

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    tot_pnl = common_factor_pnl + specific_pnl
    cum_common_factor_pnl = common_factor_pnl.cumsum()
    cum_specific_pnl = specific_pnl.cumsum()
    cum_tot_pnl = tot_pnl.cumsum()

    ax.plot(cum_specific_pnl, color='b', label='cum specific pnl')
    ax.plot(cum_common_factor_pnl, color='r', label='cum common factor pnl')
    ax.plot(cum_tot_pnl, color='g', label='cum total pnl')
    ax.set_title('Time Series of Cumulative PnL')
    ax.set_ylabel('PnL ($)')
    ax.legend()

    return ax


def compute_holdings_pnl_1d():
    '''
    Computes holdings PnL for a given day
    '''
    pass


def compute_trading_pnl_1d(specific_pnl_1d, holdings_pnl_1d):
    '''
    Computes fraction of specific PnL due to trading for a given day

    Parameters
    ----------
    specific_pnl_1d : float
        Specific PnL for the given day
        - Exact output of compute_specific_pnl_1d

    holdings_pnl_1d : float
        Fraction of specific PnL due to holdings
        - See full explanation in perf_attrib_1d

    Returns
    -------
    trading_pnl_1d : float
        Part of specific_pnl_1d that is due to trading
    '''
    trading_pnl_1d = specific_pnl_1d - holdings_pnl_1d
    return trading_pnl_1d


def compute_variances_1d(holdings_1d, factor_loadings_1d,
                         factor_covariances_1d,
                         stock_specific_variances_1d):
    '''
    Computes common factor variance, specific variance and portfolio variance
    of the algorithm for a given day

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common factors for the given day
        - See full explanation in perf_attrib_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d

    stock_specific_variances_1d : pd.DataFrame
        Stock specific variances
        - See full explanation in perf_attrib_1d

    Returns
    -------
    common_factor_var_1d : float
        Common factor variance for the portfolio for the given day

    specific_var_1d : float
        Specific variance for the portfolio for the given day

    portfolio_var_1d : float
        Portfolio variance for the portfolio for the given day
    '''
    common_factor_var_1d = holdings_1d.dot(factor_loadings_1d) \
        .dot(factor_covariances_1d).dot(factor_loadings_1d.transpose()) \
        .dot(holdings_1d.transpose())

    specific_var_1d = holdings_1d.dot(stock_specific_variances_1d) \
        .dot(holdings_1d.transpose())

    portfolio_var_1d = common_factor_var_1d + specific_var_1d

    return common_factor_var_1d, specific_var_1d, portfolio_var_1d


def plot_risks(common_factor_risk, specific_risk, portfolio_risk, ax=None):
    '''
    Plots time series of common factor risk, specific risk, and portfolio risk
    as line plots.

    Parameters
    ----------
    common_factor_risk : pd.Series
        Time series of common factor risk (the square root of common factor
        variance)
        - The first output of compute_variances_1d gives the square of one
        entry of this Series.
        - Example:
            2017-06-05    36102.833298
            2017-06-06    41031.349716
            2017-06-07    39010.162551

    specific_risk : pd.Series
        Time series of specific risk (the square root of specific variance)
        - The second output of compute_variances_1d gives the square of one
        entry of this Series.
        - Example:
            2017-06-05    58866.530000
            2017-06-06    49601.650490
            2017-06-07    50649.039005

    portfolio_risk : pd.Series
        Time series of portfolio risk (the square root of portfolio variance)
        - The third output of compute_variances_1d gives the square of one
        entry of this Series.
        - Example:
            2017-06-05    95321.833298
            2017-06-06    61031.646536
            2017-06-07    73268.162551

    ax : plt.Axes
        Axes on which to plot
    '''

    if ax is None:
        ax = plt.gca()

    ax.plot(common_factor_risk.index, common_factor_risk,
            label='Common Factor Risk', color='r')
    avg = common_factor_risk.mean()
    ax.axhline(avg, color='r', linestyle='--',
               label='Mean = {: .3}'.format(avg))

    ax.plot(specific_risk.index, specific_risk,
            label='Specific Risk', color='b')
    avg = specific_risk.mean()
    ax.axhline(avg, color='b', linestyle='--',
               label='Mean = {: .3}'.format(avg))

    ax.plot(portfolio_risk.index, portfolio_risk,
            label='Portfolio Risk', color='g')
    avg = portfolio_risk.mean()
    ax.axhline(avg, color='g', linestyle='--',
               label='Mean = {: .3}'.format(avg))

    ax.legend()
    ax.set_title('Common, Specific and Portfolio Risk', fontsize='medium')
    ax.set_ylabel('Risk ($)')

    return ax


def plot_proportion_specific(specific_variance, portfolio_variance, ax=None):
    '''
    Plots time series of common factor risk, specific risk, and portfolio risk
    as line plots.

    Parameters
    ----------
    specific_variance : pd.Series
        Time series of specific variance
        - Exact second output of compute_variances_1d

    portfolio_variance : pd.Series
        Time series of portfolio variance
        - Exact third output of compute_variances_1d

    ax : plt.Axes
        Axes on which to plot
    '''
    if ax is None:
        ax = plt.gca()

    proportion_specific = specific_variance.divide(portfolio_variance)

    ax.plot(proportion_specific.index, proportion_specific,
            label='Specific Variance / Portfolio Variance', color='k')
    avg = proportion_specific.mean()
    ax.axhline(avg, color='k', linestyle='--',
               label='Mean = {: .3}'.format(avg))

    ax.legend()
    ax.set_title('Specific Variance / Portfolio Variance', fontsize='medium')
    ax.set_ylabel('Proportion')

    return ax


def compute_marginal_contributions_to_risk_1d(holdings_1d, factor_loadings_1d,
                                              factor_covariances_1d,
                                              stock_specific_variances_1d):
    '''
    Compute marginal contributions to risk (MCR) to common factor risk,
    specific risk and portfolio risk for a given day

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common factors for the given day
        - See full explanation in perf_attrib_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d

    stock_specific_variances_1d : pd.DataFrame
        Stock specific variances
        - See full explanation in perf_attrib_1d

    Returns
    -------
    MCR_common_factor_1d : pd.DataFrame
        Marginal contribution to common factor risk for each stock

    MCR_specific_1d : pd.DataFrame
        Marginal contribution to specific risk for each stock

    MCR_portfolio_1d : pd.DataFrame
        Marginal contribution to portfolio risk for each stock
    '''
    var = factor_loadings_1d.dot(factor_covariances_1d) \
        .dot(factor_loadings_1d.transpose())
    MCR_common_factor_1d = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    var = stock_specific_variances_1d
    MCR_specific_1d = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    var = factor_loadings_1d.dot(factor_covariances_1d) \
        .dot(factor_loadings_1d.transpose()) + stock_specific_variances_1d
    MCR_portfolio_1d = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    return MCR_common_factor_1d, MCR_specific_1d, MCR_portfolio_1d


def autolabel(bars, heights, style='left', ax=None):
    '''
    Helper function to label bar charts with heights
    '''
    if ax is None:
        ax = plt.gca()

    for bar, height in zip(bars, heights):
        if style is 'left':
            ax.text(bar.get_x(), height,
                    '{: 3.1f}%'.format(100*height))
        elif style is 'right':
            ax.text(bar.get_x() + bar.get_width()/2.0, height,
                    '{: 3.1f}%'.format(100*height))


def compute_mcap_pnl(round_trips, mcaps):
    '''
    Computes PnL breakdown by market cap

    Parameters
    ----------
    round_trips : pd.DataFrame
        Round trips of algorithm, including open_dt, pnl and symbol
        - Output of pyfolio.round_trips.extract_round_trips()

    mcaps : pd.DataFrame
        Daily market cap per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03     1.327160e+10     6.402460e+10
        2017-04-04	   1.329620e+10     6.403694e+10
        2017-04-05	   1.297464e+10	    6.397187e+10
    '''
    mcap_attrib = pd.DataFrame()

    for row in round_trips.sort_values('symbol').iterrows():
        day = row[1].open_dt.normalize()
        asset = row[1].symbol
        pnl = row[1].pnl

        mkt_cap = mcaps.loc[day, asset.sid]
        mcap_attrib = mcap_attrib.append({'asset': asset,
                                          'market_cap_B': mkt_cap / 1e9,
                                          'pnl': pnl}, ignore_index=True)

    return mcap_attrib


def plot_mcap_pnl(market_cap_attrib, ax=None):
    '''
    Plots output of compute_mcap_pnl as a bar chart

    Parameters
    ----------
    mcap_attrib : pd.DataFrame
        PnL attribution by market cap
        - Exact output of compute_mcap_pnl()
    '''
    if ax is None:
        ax = plt.gca()

    tot_count = market_cap_attrib.count()['pnl']
    market_cap_buckets = OrderedDict([
        ('ETFs', market_cap_attrib[pd.isnull(
            market_cap_attrib.market_cap_B)].count()['pnl'] / tot_count),
        ('less than \$100m', market_cap_attrib[
            market_cap_attrib.market_cap_B < 0.1].count()['pnl'] / tot_count),
        ('\$100m - \$1b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 0.1) &
            (market_cap_attrib.market_cap_B < 1)].count()['pnl'] / tot_count),
        ('\$1b - \$10b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 1) &
            (market_cap_attrib.market_cap_B < 10)].count()['pnl'] / tot_count),
        ('\$10b - \$50b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 10) &
            (market_cap_attrib.market_cap_B < 50)].count()['pnl'] / tot_count),
        ('more than \$50b', market_cap_attrib[
            market_cap_attrib.market_cap_B > 50].count()['pnl'] / tot_count)
    ])
    market_cap_heights = market_cap_buckets.values()

    tot_pnl = market_cap_attrib.pnl.sum()
    pnl_market_cap_buckets = OrderedDict([
        ('ETFs', market_cap_attrib[pd.isnull(
            market_cap_attrib.market_cap_B)].pnl.sum() / tot_pnl),
        ('less than \$100m', market_cap_attrib[
            market_cap_attrib.market_cap_B < 0.1].pnl.sum() / tot_pnl),
        ('\$100m - \$1b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 0.1) &
            (market_cap_attrib.market_cap_B < 1)].pnl.sum() / tot_pnl),
        ('\$1b - \$10b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 1) &
            (market_cap_attrib.market_cap_B < 10)].pnl.sum() / tot_pnl),
        ('\$10b - \$50b', market_cap_attrib[
            (market_cap_attrib.market_cap_B > 10) &
            (market_cap_attrib.market_cap_B < 50)].pnl.sum() / tot_pnl),
        ('more than \$50b', market_cap_attrib[
            market_cap_attrib.market_cap_B > 50].pnl.sum() / tot_pnl)
    ])
    pnl_market_cap_heights = pnl_market_cap_buckets.values()

    pnl_colors = []
    for pnl in pnl_market_cap_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    bars1 = ax.bar(list(range(6)), market_cap_buckets.values(),
                   color='orange', alpha=0.8, width=0.8,
                   label='Contribution to Total Number of Round Trips')
    bars2 = ax.bar(list(range(6)), pnl_market_cap_buckets.values(),
                   color=pnl_colors, alpha=0.8, width=0.6,
                   label='Contribution to Total PnL')

    ax.set_title('PnL Attribution by Market Cap', fontsize='medium')
    ax.set_xlabel('Market Cap')
    ax.set_ylabel('Percentage Contribution')
    ax.legend()
    ax.set_xticklabels([''] + pnl_market_cap_buckets.keys() + [''])
    ax.set_yticklabels(['{:3.2f}%'.format(100*y) for y in ax.get_yticks()])
    autolabel(bars1, market_cap_heights, 'left', ax)
    autolabel(bars2, pnl_market_cap_heights, 'right', ax)

    return ax


def compute_sector_pnl(round_trips, sectors):
    '''
    Computes PnL breakdown by market cap

    Parameters
    ----------
    round_trips : pd.DataFrame
        Round trips of algorithm, including open_dt, pnl and symbol
        - Output of pyfolio.round_trips.extract_round_trips()

    sector : pd.DataFrame
        Daily Morningstar sector code per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	     311.0       206.0
        2017-04-04	     311.0       206.0
        2017-04-05	     311.0	     206.0
    '''
    sector_attrib = pd.DataFrame()

    for row in round_trips.sort_values('symbol').iterrows():
        day = row[1].open_dt.normalize()
        asset = row[1].symbol
        pnl = row[1].pnl

        sector = sectors.loc[day, asset.sid]
        sector_attrib = sector_attrib.append({'asset': asset,
                                              'sector': sector,
                                              'pnl': pnl}, ignore_index=True)

    return sector_attrib


def plot_sector_pnl(sector_attrib, ax=None):
    '''
    Plots output of compute_sector_pnl as a bar chart

    Parameters
    ----------
    sector_attrib : pd.DataFrame
        PnL attribution by sector
        - Exact output of compute_sector_pnl()
    '''
    if ax is None:
        ax = plt.gca()

    tot_count = sector_attrib.count()['pnl']
    sector_buckets = OrderedDict((
        ('Basic Materials', sector_attrib[sector_attrib.sector == 101]
            .count()['pnl'] / tot_count),
        ('Consumer Cyclical', sector_attrib[sector_attrib.sector == 102]
            .count()['pnl'] / tot_count),
        ('Financial Services', sector_attrib[sector_attrib.sector == 103]
            .count()['pnl'] / tot_count),
        ('Real Estate', sector_attrib[sector_attrib.sector == 104]
            .count()['pnl'] / tot_count),
        ('Consumer Defensive', sector_attrib[sector_attrib.sector == 205]
            .count()['pnl'] / tot_count),
        ('Healthcare', sector_attrib[sector_attrib.sector == 206]
            .count()['pnl'] / tot_count),
        ('Utilities', sector_attrib[sector_attrib.sector == 207]
            .count()['pnl'] / tot_count),
        ('Communication Services', sector_attrib[sector_attrib.sector == 308]
            .count()['pnl'] / tot_count),
        ('Energy', sector_attrib[sector_attrib.sector == 309]
            .count()['pnl'] / tot_count),
        ('Industrials', sector_attrib[sector_attrib.sector == 310]
            .count()['pnl'] / tot_count),
        ('Technology', sector_attrib[sector_attrib.sector == 311]
            .count()['pnl'] / tot_count)
    ))
    sector_heights = sector_buckets.values()

    tot_pnl = sector_attrib.pnl.sum()
    pnl_sector_buckets = OrderedDict((
        ('Basic Materials', sector_attrib[sector_attrib.sector == 101]
            .pnl.sum() / tot_pnl),
        ('Consumer Cyclical', sector_attrib[sector_attrib.sector == 102]
            .pnl.sum() / tot_pnl),
        ('Financial Services', sector_attrib[sector_attrib.sector == 103]
            .pnl.sum() / tot_pnl),
        ('Real Estate', sector_attrib[sector_attrib.sector == 104]
            .pnl.sum() / tot_pnl),
        ('Consumer Defensive', sector_attrib[sector_attrib.sector == 205]
            .pnl.sum() / tot_pnl),
        ('Healthcare', sector_attrib[sector_attrib.sector == 206]
            .pnl.sum() / tot_pnl),
        ('Utilities', sector_attrib[sector_attrib.sector == 207]
            .pnl.sum() / tot_pnl),
        ('Communication Services', sector_attrib[sector_attrib.sector == 308]
            .pnl.sum() / tot_pnl),
        ('Energy', sector_attrib[sector_attrib.sector == 309]
            .pnl.sum() / tot_pnl),
        ('Industrials', sector_attrib[sector_attrib.sector == 310]
            .pnl.sum() / tot_pnl),
        ('Technology', sector_attrib[sector_attrib.sector == 311]
            .pnl.sum() / tot_pnl)
    ))
    pnl_sector_heights = pnl_sector_buckets.values()

    pnl_colors = []
    for pnl in pnl_sector_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    bars1 = ax.bar(list(range(11)), sector_buckets.values(),
                   color='orange', alpha=0.8, width=0.8,
                   label='Contribution to Total Number of Round Trips')
    bars2 = ax.bar(list(range(11)), pnl_sector_buckets.values(),
                   color=pnl_colors, alpha=0.8, width=0.6,
                   label='Contribution to Total PnL')

    ax.set_title('PnL Attribution by Sector', fontsize='medium')
    ax.set_xlabel('Sector')
    ax.set_ylabel('Percentage Contribution')
    ax.legend()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xticklabels(pnl_sector_buckets.keys(), rotation=30)
    ax.set_yticklabels(['{:3.2f}%'.format(100*y) for y in ax.get_yticks()])

    autolabel(bars1, sector_heights, 'left', ax)
    autolabel(bars2, pnl_sector_heights, 'right', ax)

    return ax


def compute_q_univ_pnl(round_trips, q500, q1500):
    '''
    Computes PnL breakdown by market cap

    Parameters
    ----------
    round_trips : pd.DataFrame
        Round trips of algorithm, including open_dt, pnl and symbol
        - Output of pyfolio.round_trips.extract_round_trips()

    q500, q1500 : pd.DataFrames
        Boolean dataframes indicating stocks that are in the Q500 / Q1500 each
        day. True if the stock is in the universe, False if not.
        - Example:
        sid	        2	    21	    24
        2010-01-04	True	False	True
        2010-01-05	True	False	True
    '''
    q_univ_attrib = pd.DataFrame()

    for row in round_trips.sort_values('symbol').iterrows():
        day = row[1].open_dt.normalize()
        asset = row[1].symbol
        pnl = row[1].pnl

        if q500.loc[day, asset.sid] is True:
            q_univ = 500
        elif q1500.loc[day, asset.sid] is True:
            q_univ = 1500
        else:
            q_univ = -1

        q_univ_attrib = q_univ_attrib.append({'asset': asset,
                                              'q_univ': q_univ,
                                              'pnl': pnl}, ignore_index=True)

    return q_univ_attrib


def plot_q_univ_pnl(q_univ_attrib, ax=None):
    '''
    Plots output of compute_q_univ_pnl as a bar chart

    Parameters
    ----------
    q_univ_attrib : pd.DataFrame
        PnL attribution by Q universe
        - Exact output of compute_q_univ_pnl()
    '''
    if ax is None:
        ax = plt.gca()

    tot_count = q_univ_attrib.count()['pnl']
    q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500]
            .count().pnl / tot_count),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500) |
                                (q_univ_attrib.q_univ == 1500)]
            .count().pnl / tot_count),
        ('All Q Securities', 1)
    ))
    q_univ_heights = q_univ_buckets.values()

    tot_pnl = q_univ_attrib.pnl.sum()
    pnl_q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500]
            .pnl.sum() / tot_pnl),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500) |
                                (q_univ_attrib.q_univ == 1500)]
            .pnl.sum()/tot_pnl),
        ('All Q Securities', 1)
    ))
    pnl_q_univ_heights = pnl_q_univ_buckets.values()

    pnl_colors = []
    for pnl in pnl_q_univ_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    bars1 = ax.bar(list(range(3)), q_univ_buckets.values(),
                   color='orange', alpha=0.8, width=0.8,
                   label='Contribution to Total Number of Round Trips')
    bars2 = ax.bar(list(range(3)), pnl_q_univ_buckets.values(),
                   color=pnl_colors, alpha=0.8, width=0.6,
                   label='Contribution to Total PnL')

    ax.set_title('''PnL Attribution by Q Universe (Cumulative Plot) \n Note:
                 middle bar is round trips/PnL of Q1500 INCLUDING Q500''',
                 fontsize='medium')
    ax.set_xlabel('Q Universe')
    ax.set_ylabel('Percentage Contribution')
    ax.legend()
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels(pnl_q_univ_buckets.keys())
    ax.set_yticklabels(['{:3.2f}%'.format(100*y) for y in ax.get_yticks()])

    autolabel(bars1, q_univ_heights, 'left', ax)
    autolabel(bars2, pnl_q_univ_heights, 'right', ax)

    return ax


def compute_price_pnl(round_trips, pricing, minutely=False):
    '''
    Computes PnL breakdown by entering price (i.e. the price of the stock at
    the opening of the round trip)

    Parameters
    ----------
    round_trips : pd.DataFrame
        Round trips of algorithm, including open_dt, pnl and symbol
        - Output of pyfolio.round_trips.extract_round_trips()

    pricing : pd.DataFrame
        Pricing data per stock per day. If minutely is True, this must be
        minutely pricing data. Otherwise, this must be end of day pricing data.
        - Example (end of day pricing data):
                                      Equity(2 [ARNC])	    Equity(24 [AAPL])
        2015-01-05 00:00:00+00:00	  43.645	            134.350
        2015-01-06 00:00:00+00:00	  59.747	            101.001
        2015-01-07 00:00:00+00:00	  94.234	            197.932

    minutely : boolean
        If True, expects pricing to be minutely pricing data, and sorts round
        trips by the entering price at the exact minute of entering the round
        trip, instead of the end of day price.
        - Defaults to False
    '''
    price_attrib = pd.DataFrame()

    for row in round_trips.sort_values('symbol').iterrows():
        day = row[1].open_dt
        if not minutely:
            day = day.normalize()
        asset = row[1].symbol
        pnl = row[1].pnl

        # Assign price_bucket
        entering_price = pricing.loc[day, asset]
        if 0 <= entering_price and entering_price < 1:
            price_bucket = 0
        elif 1 <= entering_price and entering_price < 2:
            price_bucket = 1
        elif 2 <= entering_price and entering_price < 3:
            price_bucket = 2
        elif 3 <= entering_price and entering_price < 4:
            price_bucket = 3
        elif 4 <= entering_price and entering_price < 5:
            price_bucket = 4
        elif 5 <= entering_price and entering_price < 10:
            price_bucket = 5
        elif 10 <= entering_price and entering_price < 50:
            price_bucket = 6
        elif 50 <= entering_price and entering_price < 100:
            price_bucket = 7
        else:
            price_bucket = 8

        price_attrib = price_attrib.append({'asset': asset,
                                            'price_bucket': price_bucket,
                                            'pnl': pnl}, ignore_index=True)

    return price_attrib


def plot_price_pnl(price_attrib, ax=None):
    '''
    Plots output of compute_price_pnl as a bar chart

    Parameters
    ----------
    price_attrib : pd.DataFrame
        PnL attribution by entering price of round trip
        - Exact output of compute_price_pnl()
    '''
    if ax is None:
        ax = plt.gca()

    tot_count = price_attrib.count()['pnl']
    price_buckets = OrderedDict((
        ('$0-1', price_attrib[price_attrib.price_bucket == 0].count().pnl
            / tot_count),
        ('$1-2', price_attrib[price_attrib.price_bucket == 1].count().pnl
            / tot_count),
        ('$2-3', price_attrib[price_attrib.price_bucket == 2].count().pnl
            / tot_count),
        ('$3-4', price_attrib[price_attrib.price_bucket == 3].count().pnl
            / tot_count),
        ('$4-5', price_attrib[price_attrib.price_bucket == 4].count().pnl
            / tot_count),
        ('$5-10', price_attrib[price_attrib.price_bucket == 5].count().pnl
            / tot_count),
        ('$10-50', price_attrib[price_attrib.price_bucket == 6].count().pnl
            / tot_count),
        ('$50-100', price_attrib[price_attrib.price_bucket == 7].count().pnl
            / tot_count),
        ('$100+', price_attrib[price_attrib.price_bucket == 8].count().pnl
            / tot_count),
    ))
    price_heights = price_buckets.values()

    tot_pnl = price_attrib.pnl.sum()
    pnl_price_buckets = OrderedDict((
        ('$0-1', price_attrib[price_attrib.price_bucket == 0].pnl.sum()
            / tot_pnl),
        ('$1-2', price_attrib[price_attrib.price_bucket == 1].pnl.sum()
            / tot_pnl),
        ('$2-3', price_attrib[price_attrib.price_bucket == 2].pnl.sum()
            / tot_pnl),
        ('$3-4', price_attrib[price_attrib.price_bucket == 3].pnl.sum()
            / tot_pnl),
        ('$4-5', price_attrib[price_attrib.price_bucket == 4].pnl.sum()
            / tot_pnl),
        ('$5-10', price_attrib[price_attrib.price_bucket == 5].pnl.sum()
            / tot_pnl),
        ('$10-50', price_attrib[price_attrib.price_bucket == 6].pnl.sum()
            / tot_pnl),
        ('$50-100', price_attrib[price_attrib.price_bucket == 7].pnl.sum()
            / tot_pnl),
        ('$100+', price_attrib[price_attrib.price_bucket == 8].pnl.sum()
            / tot_pnl),
    ))
    pnl_price_heights = pnl_price_buckets.values()

    pnl_colors = []
    for pnl in pnl_price_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    bars1 = ax.bar(list(range(9)), price_buckets.values(),
                   color='orange', alpha=0.8, width=0.8,
                   label='Contribution to Total Number of Round Trips')
    bars2 = ax.bar(list(range(9)), pnl_price_buckets.values(),
                   color=pnl_colors, alpha=0.8, width=0.6,
                   label='Contribution to Total PnL')

    plt.title('PnL Attribution by Round Trip Entering Price',
              fontsize='medium')
    plt.xlabel('Entering Price')
    plt.ylabel('Percentage Contribution')
    plt.legend()
    plt.axvline(4.5, color='k', linestyle='--')
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_xticklabels(pnl_price_buckets.keys())
    ax.set_yticklabels(['{:3.2f}%'.format(100*y) for y in ax.get_yticks()])

    autolabel(bars1, price_heights, 'left', ax)
    autolabel(bars2, pnl_price_heights, 'right', ax)

    return ax
