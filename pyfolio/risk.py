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
from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np


SECTORS = OrderedDict([
    (101, 'Basic Materials'),
    (102, 'Consumer Cyclical'),
    (103, 'Financial Services'),
    (104, 'Real Estate'),
    (205, 'Consumer Defensive'),
    (206, 'Healthcare'),
    (207, 'Utilities'),
    (308, 'Communication Services'),
    (309, 'Energy'),
    (310, 'Industrials'),
    (311, 'Technology')
])

CAP_BUCKETS = OrderedDict([
    ('Micro', (50000000, 300000000)),
    ('Small', (300000000, 2000000000)),
    ('Mid', (2000000000, 10000000000)),
    ('Large', (10000000000, 200000000000)),
    ('Mega', (200000000000, np.inf))
])


def compute_style_factor_exposures(positions, risk_factor):
    """
    Returns style factor exposure of an algorithm's positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in create_risk_tear_sheet

    risk_factor : pd.DataFrame
        Daily risk factor per asset.
        - DataFrame with dates as index and equities as columns
        - Example:
                         Equity(24   Equity(62
                           [AAPL])      [ABT])
        2017-04-03	  -0.51284     1.39173
        2017-04-04	  -0.73381     0.98149
        2017-04-05	  -0.90132     1.13981
    """

    positions_wo_cash = positions.drop('cash', axis='columns')
    gross_exposure = positions_wo_cash.abs().sum(axis='columns')

    style_factor_exposure = positions_wo_cash.multiply(risk_factor) \
        .divide(gross_exposure, axis='index')
    tot_style_factor_exposure = style_factor_exposure.sum(axis='columns',
                                                          skipna=True)

    return tot_style_factor_exposure


def plot_style_factor_exposures(tot_style_factor_exposure, factor_name=None,
                                ax=None):
    """
    Plots DataFrame output of compute_style_factor_exposures as a line graph

    Parameters
    ----------
    tot_style_factor_exposure : pd.Series
        Daily style factor exposures (output of compute_style_factor_exposures)
        - Time series with decimal style factor exposures
        - Example:
            2017-04-24    0.037820
            2017-04-25    0.016413
            2017-04-26   -0.021472
            2017-04-27   -0.024859

    factor_name : string
        Name of style factor, for use in graph title
        - Defaults to tot_style_factor_exposure.name
    """

    if ax is None:
        ax = plt.gca()

    if factor_name is None:
        factor_name = tot_style_factor_exposure.name

    ax.plot(tot_style_factor_exposure.index, tot_style_factor_exposure,
            label=factor_name)
    avg = tot_style_factor_exposure.mean()
    ax.axhline(avg, linestyle='-.', label='Mean = {:.3}'.format(avg))
    ax.axhline(0, color='k', linestyle='-')
    _, _, y1, y2 = plt.axis()
    lim = max(abs(y1), abs(y2))
    ax.set(title='Exposure to {}'.format(factor_name),
           ylabel='{} \n weighted exposure'.format(factor_name),
           ylim=(-lim, lim))
    ax.legend(frameon=True, framealpha=0.5)

    return ax


def compute_sector_exposures(positions, sectors, sector_dict=SECTORS):
    """
    Returns arrays of long, short and gross sector exposures of an algorithm's
    positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in compute_style_factor_exposures.

    sectors : pd.DataFrame
        Daily Morningstar sector code per asset
        - See full explanation in create_risk_tear_sheet

    sector_dict : dict or OrderedDict
        Dictionary of all sectors
        - Keys are sector codes (e.g. ints or strings) and values are sector
          names (which must be strings)
        - Defaults to Morningstar sectors
    """

    sector_ids = sector_dict.keys()

    long_exposures = []
    short_exposures = []
    gross_exposures = []
    net_exposures = []

    positions_wo_cash = positions.drop('cash', axis='columns')
    long_exposure = positions_wo_cash[positions_wo_cash > 0] \
        .sum(axis='columns')
    short_exposure = positions_wo_cash[positions_wo_cash < 0] \
        .abs().sum(axis='columns')
    gross_exposure = positions_wo_cash.abs().sum(axis='columns')

    for sector_id in sector_ids:
        in_sector = positions_wo_cash[sectors == sector_id]

        long_sector = in_sector[in_sector > 0] \
            .sum(axis='columns').divide(long_exposure)
        short_sector = in_sector[in_sector < 0] \
            .sum(axis='columns').divide(short_exposure)
        gross_sector = in_sector.abs().sum(axis='columns') \
            .divide(gross_exposure)
        net_sector = long_sector.subtract(short_sector)

        long_exposures.append(long_sector)
        short_exposures.append(short_sector)
        gross_exposures.append(gross_sector)
        net_exposures.append(net_sector)

    return long_exposures, short_exposures, gross_exposures, net_exposures


def plot_sector_exposures_longshort(long_exposures, short_exposures,
                                    sector_dict=SECTORS, ax=None):
    """
    Plots outputs of compute_sector_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures : arrays
        Arrays of long and short sector exposures (output of
        compute_sector_exposures).

    sector_dict : dict or OrderedDict
        Dictionary of all sectors
        - See full description in compute_sector_exposures
    """

    if ax is None:
        ax = plt.gca()

    if sector_dict is None:
        sector_names = SECTORS.values()
    else:
        sector_names = sector_dict.values()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 11))

    ax.stackplot(long_exposures[0].index, long_exposures,
                 labels=sector_names, colors=color_list, alpha=0.8,
                 baseline='zero')
    ax.stackplot(long_exposures[0].index, short_exposures,
                 colors=color_list, alpha=0.8, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set(title='Long and short exposures to sectors',
           ylabel='Proportion of long/short exposure in sectors')
    ax.legend(loc='upper left', frameon=True, framealpha=0.5)

    return ax


def plot_sector_exposures_gross(gross_exposures, sector_dict=None, ax=None):
    """
    Plots output of compute_sector_exposures as area charts

    Parameters
    ----------
    gross_exposures : arrays
        Arrays of gross sector exposures (output of compute_sector_exposures).

    sector_dict : dict or OrderedDict
        Dictionary of all sectors
        - See full description in compute_sector_exposures
    """

    if ax is None:
        ax = plt.gca()

    if sector_dict is None:
        sector_names = SECTORS.values()
    else:
        sector_names = sector_dict.values()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 11))

    ax.stackplot(gross_exposures[0].index, gross_exposures,
                 labels=sector_names, colors=color_list, alpha=0.8,
                 baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set(title='Gross exposure to sectors',
           ylabel='Proportion of gross exposure \n in sectors')

    return ax


def plot_sector_exposures_net(net_exposures, sector_dict=None, ax=None):
    """
    Plots output of compute_sector_exposures as line graphs

    Parameters
    ----------
    net_exposures : arrays
        Arrays of net sector exposures (output of compute_sector_exposures).

    sector_dict : dict or OrderedDict
        Dictionary of all sectors
        - See full description in compute_sector_exposures
    """

    if ax is None:
        ax = plt.gca()

    if sector_dict is None:
        sector_names = SECTORS.values()
    else:
        sector_names = sector_dict.values()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 11))

    for i in range(len(net_exposures)):
        ax.plot(net_exposures[i], color=color_list[i], alpha=0.8,
                label=sector_names[i])
    ax.set(title='Net exposures to sectors',
           ylabel='Proportion of net exposure \n in sectors')

    return ax


def compute_cap_exposures(positions, caps):
    """
    Returns arrays of long, short and gross market cap exposures of an
    algorithm's positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in compute_style_factor_exposures.

    caps : pd.DataFrame
        Daily Morningstar sector code per asset
        - See full explanation in create_risk_tear_sheet
    """

    long_exposures = []
    short_exposures = []
    gross_exposures = []
    net_exposures = []

    positions_wo_cash = positions.drop('cash', axis='columns')
    tot_gross_exposure = positions_wo_cash.abs().sum(axis='columns')
    tot_long_exposure = positions_wo_cash[positions_wo_cash > 0] \
        .sum(axis='columns')
    tot_short_exposure = positions_wo_cash[positions_wo_cash < 0] \
        .abs().sum(axis='columns')

    for bucket_name, boundaries in CAP_BUCKETS.items():
        in_bucket = positions_wo_cash[(caps >= boundaries[0]) &
                                      (caps <= boundaries[1])]

        gross_bucket = in_bucket.abs().sum(axis='columns') \
            .divide(tot_gross_exposure)
        long_bucket = in_bucket[in_bucket > 0] \
            .sum(axis='columns').divide(tot_long_exposure)
        short_bucket = in_bucket[in_bucket < 0] \
            .sum(axis='columns').divide(tot_short_exposure)
        net_bucket = long_bucket.subtract(short_bucket)

        gross_exposures.append(gross_bucket)
        long_exposures.append(long_bucket)
        short_exposures.append(short_bucket)
        net_exposures.append(net_bucket)

    return long_exposures, short_exposures, gross_exposures, net_exposures


def plot_cap_exposures_longshort(long_exposures, short_exposures, ax=None):
    """
    Plots outputs of compute_cap_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures : arrays
        Arrays of long and short market cap exposures (output of
        compute_cap_exposures).
    """

    if ax is None:
        ax = plt.gca()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    ax.stackplot(long_exposures[0].index, long_exposures,
                 labels=CAP_BUCKETS.keys(), colors=color_list, alpha=0.8,
                 baseline='zero')
    ax.stackplot(long_exposures[0].index, short_exposures, colors=color_list,
                 alpha=0.8, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set(title='Long and short exposures to market caps',
           ylabel='Proportion of long/short exposure in market cap buckets')
    ax.legend(loc='upper left', frameon=True, framealpha=0.5)

    return ax


def plot_cap_exposures_gross(gross_exposures, ax=None):
    """
    Plots outputs of compute_cap_exposures as area charts

    Parameters
    ----------
    gross_exposures : array
        Arrays of gross market cap exposures (output of compute_cap_exposures).
    """

    if ax is None:
        ax = plt.gca()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    ax.stackplot(gross_exposures[0].index, gross_exposures,
                 labels=CAP_BUCKETS.keys(), colors=color_list, alpha=0.8,
                 baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set(title='Gross exposure to market caps',
           ylabel='Proportion of gross exposure \n in market cap buckets')

    return ax


def plot_cap_exposures_net(net_exposures, ax=None):
    """
    Plots outputs of compute_cap_exposures as line graphs

    Parameters
    ----------
    net_exposures : array
        Arrays of gross market cap exposures (output of compute_cap_exposures).
    """

    if ax is None:
        ax = plt.gca()

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    cap_names = CAP_BUCKETS.keys()
    for i in range(len(net_exposures)):
        ax.plot(net_exposures[i], color=color_list[i], alpha=0.8,
                label=cap_names[i])
    ax.axhline(0, color='k', linestyle='-')
    ax.set(title='Net exposure to market caps',
           ylabel='Proportion of net exposure \n in market cap buckets')

    return ax


def compute_volume_exposures(shares_held, volumes, percentile):
    """
    Returns arrays of pth percentile of long, short and gross volume exposures
    of an algorithm's held shares

    Parameters
    ----------
    shares_held : pd.DataFrame
        Daily number of shares held by an algorithm.
        - See full explanation in create_risk_tear_sheet

    volume : pd.DataFrame
        Daily volume per asset
        - See full explanation in create_risk_tear_sheet

    percentile : float
        Percentile to use when computing and plotting volume exposures
        - See full explanation in create_risk_tear_sheet
    """

    shares_held = shares_held.replace(0, np.nan)

    shares_longed = shares_held[shares_held > 0]
    shares_shorted = -1 * shares_held[shares_held < 0]
    shares_grossed = shares_held.abs()

    longed_frac = shares_longed.divide(volumes)
    shorted_frac = shares_shorted.divide(volumes)
    grossed_frac = shares_grossed.divide(volumes)

    # NOTE: To work around a bug in `quantile` with nan-handling in
    #       pandas 0.18, use np.nanpercentile by applying to each row of
    #       the dataframe. This is fixed in pandas 0.19.
    #
    # longed_threshold = 100*longed_frac.quantile(percentile, axis='columns')
    # shorted_threshold = 100*shorted_frac.quantile(percentile, axis='columns')
    # grossed_threshold = 100*grossed_frac.quantile(percentile, axis='columns')

    longed_threshold = 100 * longed_frac.apply(
        partial(np.nanpercentile, q=100 * percentile),
        axis='columns',
    )
    shorted_threshold = 100 * shorted_frac.apply(
        partial(np.nanpercentile, q=100 * percentile),
        axis='columns',
    )
    grossed_threshold = 100 * grossed_frac.apply(
        partial(np.nanpercentile, q=100 * percentile),
        axis='columns',
    )

    return longed_threshold, shorted_threshold, grossed_threshold


def plot_volume_exposures_longshort(longed_threshold, shorted_threshold,
                                    percentile, ax=None):
    """
    Plots outputs of compute_volume_exposures as line graphs

    Parameters
    ----------
    longed_threshold, shorted_threshold : pd.Series
        Series of longed and shorted volume exposures (output of
        compute_volume_exposures).

    percentile : float
        Percentile to use when computing and plotting volume exposures.
        - See full explanation in create_risk_tear_sheet
    """

    if ax is None:
        ax = plt.gca()

    ax.plot(longed_threshold.index, longed_threshold,
            color='b', label='long')
    ax.plot(shorted_threshold.index, shorted_threshold,
            color='r', label='short')
    ax.axhline(0, color='k')
    ax.set(title='Long and short exposures to illiquidity',
           ylabel='{}th percentile of proportion of volume (%)'
           .format(100 * percentile))
    ax.legend(frameon=True, framealpha=0.5)

    return ax


def plot_volume_exposures_gross(grossed_threshold, percentile, ax=None):
    """
    Plots outputs of compute_volume_exposures as line graphs

    Parameters
    ----------
    grossed_threshold : pd.Series
        Series of grossed volume exposures (output of
        compute_volume_exposures).

    percentile : float
        Percentile to use when computing and plotting volume exposures
        - See full explanation in create_risk_tear_sheet
    """

    if ax is None:
        ax = plt.gca()

    ax.plot(grossed_threshold.index, grossed_threshold,
            color='b', label='gross')
    ax.axhline(0, color='k')
    ax.set(title='Gross exposure to illiquidity',
           ylabel='{}th percentile of \n proportion of volume (%)'
           .format(100 * percentile))
    ax.legend(frameon=True, framealpha=0.5)

    return ax
