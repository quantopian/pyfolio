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


# 31 visually distinct colors...
# http://phrogz.net/css/distinct-colors.html
COLORS = [
    '#f23d3d', '#828c23', '#698c83', '#594080', '#994d4d',
    '#206380', '#dd39e6', '#cc9999', '#7c8060', '#66adcc',
    '#6c7dd9', '#8a698c', '#7f6340', '#66cc7a', '#a3abd9',
    '#d9c0a3', '#bfffcc', '#542699', '#b35986', '#d4e639',
    '#b380ff', '#e0e6ac', '#a253a6', '#418020', '#ff409f',
    '#ffa940', '#83ff40', '#3d58f2', '#e3ace6', '#d9a86c',
    '#2db391'
]


def perf_attrib(factor_loadings_list,
                stock_specific_variances_list,
                factor_covariances_list,
                factor_returns_list,
                holdings_list,
                pnl_list,
                holdings_pnl_list,
                aum_list,
                date_range):

    pnl_series = pd.Series()
    exposures_df = pd.DataFrame()
    vol_weighted_exposures_df = pd.DataFrame()
    aum_series = pd.Series()
    common_factor_pnls_df = pd.DataFrame()
    specific_pnl_series = pd.Series()
    holdings_pnl_series = pd.Series()
    trading_pnl_series = pd.Series()
    common_factor_risk_series = pd.Series()
    specific_risk_series = pd.Series()
    portfolio_risk_series = pd.Series()
    MCR_common_factor_df = pd.DataFrame()
    MCR_specific_df = pd.DataFrame()
    MCR_portfolio_df = pd.DataFrame()

    perf_attrib_dict = OrderedDict([
        ('pnl', pnl_series)
        ('exposures', exposures_df)
        ('vol weighted exposures', vol_weighted_exposures_df)
        ('aum', aum_series)
        ('common factor pnl', common_factor_pnls_df)
        ('specific pnl', specific_pnl_series)
        ('holdings pnl', holdings_pnl_series)
        ('trading pnl', trading_pnl_series)
        ('common factor risk', common_factor_risk_series)
        ('specific risk', specific_risk_series)
        ('portfolio risk', portfolio_risk_series)
        ('MCR common factor', MCR_common_factor_df)
        ('MCR specific', MCR_specific_df)
        ('MCR portfolio', MCR_portfolio_df)
    ])

    for i in len(date_range):
        tup = perf_attrib_1d(factor_loadings_list[i],
                             stock_specific_variances_list[i],
                             factor_covariances_list[i],
                             factor_returns_list[i],
                             holdings_list[i],
                             pnl_list[i],
                             holdings_pnl_list[i],
                             aum_list[i],
                             date_range[i])

        for key, value in perf_attrib_dict.items():
            perf_attrib_dict[key] = perf_attrib_dict[key].append(tup[i])

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
        Factor loadings of each stock to common risk factors for the given day
        - Columns are common factors, indexed by sids
        - Example:
                momentum	size	     value
            2   0.185313    0.916532     0.174353
            24	1.791453    -1.97424     1.016321
            41  -0.14235    1.129351     -2.05923

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - Square matrix with both columns and index being common risk factors
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
        Returns associated with common risk factors for the given day
        - Returns, indexed by common risk factor
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
    exposures_1d = compute_risk_factor_exposures_1d(holdings_1d,
                                                    factor_loadings_1d)

    vol_weighted_exposures = \
        compute_vol_weighted_risk_factor_exposures_1d(exposures_1d,
                                                      factor_covariances_1d)

    common_factor_pnls_1d = compute_common_factor_pnls_1d(exposures_1d,
                                                          factor_returns_1d)

    specific_pnl_1d = compute_specific_pnl_1d(pnl_1d, common_factor_pnls_1d)

    holdings_pnl_1d = compute_holdings_pnl()

    trading_pnl = compute_trading_pnl_1d(specific_pnl_1d, holdings_pnl_1d)

    common_factor_var, specific_var, portfolio_var = \
        compute_variances_1d(holdings_1d, factor_loadings_1d,
                             factor_covariances_1d,
                             stock_specific_variances_1d)

    proportion_specific = specific_var / portfolio_var

    MCR_common_factor, MCR_specific, MCR_portfolio = \
        compute_marginal_contributions_to_risk_1d(holdings_1d,
                                                  factor_loadings_1d,
                                                  factor_covariances_1d,
                                                  stock_specific_variances_1d)

    return (
        ('total pnl', pnl_1d),
        ('factor exposure', exposures_1d),
        ('vol weighted factor exposure', vol_weighted_exposures),
        ('AUM', aum_1d),
        ('common factor pnls', common_factor_pnls_1d),
        ('specific pnl', specific_pnl_1d),
        ('holdings pnl', holdings_pnl_1d),
        ('trading pnl', trading_pnl),
        ('common factor risk', np.sqrt(common_factor_var)),
        ('specific risk', np.sqrt(specific_var)),
        ('portfolio risk', np.sqrt(portfolio_var)),
        ('proportion specific', proportion_specific),
        ('MCR common factor', MCR_common_factor),
        ('MCR specific', MCR_specific),
        ('MCR portfolio', MCR_portfolio)
    )


def compute_risk_factor_exposures_1d(holdings_1d, factor_loadings_1d):
    '''
    Computes dollar risk factor exposures

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common risk factors for the given day
        - See full explanation in perf_attrib_1d

    Returns
    -------
    exposures_1d :
    '''
    return holdings_1d.dot(factor_loadings_1d)


def plot_risk_factor_exposures(exposures_1d, ax=None):
    if ax is None:
        ax = plt.gca()

    pos_exposures = exposures_1d.copy()
    neg_exposures = exposures_1d.copy()
    pos_exposures[pos_exposures < 0] = 0
    neg_exposures[neg_exposures > 0] = 0

    pos_list = []
    neg_list = []
    for i in range(len(pos_exposures.columns)):
        pos_list.append(pos_exposures.iloc[:, i].values)
        neg_list.append(neg_exposures.iloc[:, i].values)

    ax.stackplot(exposures_1d.index, pos_list, colors=COLORS, alpha=0.8,
                 labels=pos_exposures.columns)
    ax.stackplot(exposures_1d.index, neg_list, colors=COLORS, alpha=0.8)
    ax.axhline(0, color='k')
    ax.legend(loc=2, frameon=True)
    ax.set_ylabel('Exposure ($)')
    ax.set_title('Risk Factor Exposures', fontsize='large')

    return ax


def compute_vol_weighted_risk_factor_exposures_1d(exposures_1d,
                                                  factor_covariances_1d):
    '''
    Computes volatility-weighted dollar risk factor exposures

    Parameters
    ----------
    exposures_1d : pd.Series
        Risk factor exposures of the portfolio for the given day
        - Exact output of compute_risk_factor_exposures_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d
    '''
    vol = pd.Series(data=np.diag(factor_covariances_1d),
                    index=exposures_1d.index)
    return exposures_1d.multiply(np.sqrt(vol))


def compute_common_factor_pnls_1d(exposures_1d, factor_returns_1d):
    '''
    Computes PnL due to common risk factors

    Parameters
    ----------
    exposures_1d : pd.Series
        Risk factor exposures of the portfolio for the given day
        - Exact output of compute_risk_factor_exposures_1d

    factor_returns_1d : pd.Series
        Returns associated with common risk factors for the given day
        - See full explanation in perf_attrib_1d
    '''
    return exposures_1d.multiply(factor_returns_1d)


def compute_specific_pnl_1d(pnl_1d, common_factor_pnls_1d):
    '''
    Computes PnL that is not due to common risk factors

    Parameters
    ----------
    pnl_1d : float
        Total PnL for the given day

    common_factor_pnls_1d : pd.Series
        PnL due to common risk factors for the given day
        - Exact output of compute_common_factor_pnls_1d
    '''
    return pnl_1d - common_factor_pnls_1d.sum()


def compute_holdings_pnl():
    pass


def compute_trading_pnl_1d(specific_pnl_1d, holdings_pnl_1d):
    '''
    Computes fraction of specific PnL due to trading

    Parameters
    ----------
    specific_pnl_1d : float
        Specific PnL for the given day
        - Exact output of compute_specific_pnl_1d

    holdings_pnl_1d : float
        Fraction of specific PnL due to holdings
        - See full explanation in perf_attrib_1d
    '''
    return specific_pnl_1d - holdings_pnl_1d


def compute_variances_1d(holdings_1d, factor_loadings_1d,
                         factor_covariances_1d,
                         stock_specific_variances_1d):
    '''
    Computes common factor variance, specific variance and portfolio variance
    of the algorithm.

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common risk factors for the given day
        - See full explanation in perf_attrib_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d

    stock_specific_variances_1d : pd.DataFrame
        Stock specific variances
        - See full explanation in perf_attrib_1d
    '''
    common_factor = holdings_1d.dot(factor_loadings_1d) \
        .dot(factor_covariances_1d).dot(factor_loadings_1d.transpose()) \
        .dot(holdings_1d.transpose())
    specific = holdings_1d.dot(stock_specific_variances_1d) \
        .dot(holdings_1d.transpose())
    portfolio = common_factor + specific

    return common_factor, specific, portfolio


def compute_marginal_contributions_to_risk_1d(holdings_1d, factor_loadings_1d,
                                              factor_covariances_1d,
                                              stock_specific_variances_1d):
    '''
    Compute marginal contributions to risk (MCR) to common factor risk,
    specific risk and portfolio risk.

    Parameters
    ----------
    holdings_1d : pd.Series
        Dollar value of position per asset for the given day.
        - See full explanation in perf_attrib_1d

    factor_loadings_1d : pd.DataFrame
        Factor loadings of each stock to common risk factors for the given day
        - See full explanation in perf_attrib_1d

    factor_covariances_1d : pd.DataFrame
        Risk factor variance-covariance matrix
        - See full explanation in perf_attrib_1d

    stock_specific_variances_1d : pd.DataFrame
        Stock specific variances
        - See full explanation in perf_attrib_1d
    '''
    var = factor_loadings_1d.dot(factor_covariances_1d) \
        .dot(factor_loadings_1d.transpose())
    MCR_common_factor = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    var = stock_specific_variances_1d
    MCR_specific = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    var = factor_loadings_1d.dot(factor_covariances_1d) \
        .dot(factor_loadings_1d.transpose()) + stock_specific_variances_1d
    MCR_portfolio = var.dot(holdings_1d) \
        .divide(np.sqrt(holdings_1d.transpose().dot(var).dot(holdings_1d)))

    return MCR_common_factor, MCR_specific, MCR_portfolio
