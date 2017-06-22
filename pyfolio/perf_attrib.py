#
# Copyright 2016 Quantopian, Inc.
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
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def label_bars(bars, heights, style='left', ax=None):
    '''
    Helper function to label the height of bar graphs
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


def plot_mcap_pnl(mcap_attrib):
    tot_count = mcap_attrib.count()['pnl']
    market_cap_buckets = OrderedDict([
        ('ETFs', mcap_attrib[pd.isnull(mcap_attrib.market_cap_B)]
            .count()['pnl'] / tot_count),
        ('less than \$100m', mcap_attrib[mcap_attrib.market_cap_B < 0.1]
            .count()['pnl'] / tot_count),
        ('\$100m - \$1b', mcap_attrib[(mcap_attrib.market_cap_B > 0.1)
                                      & (mcap_attrib.market_cap_B < 1)]
            .count()['pnl'] / tot_count),
        ('\$1b - \$10b', mcap_attrib[(mcap_attrib.market_cap_B > 1)
                                     & (mcap_attrib.market_cap_B < 10)]
            .count()['pnl'] / tot_count),
        ('\$10b - \$50b', mcap_attrib[(mcap_attrib.market_cap_B > 10)
                                      & (mcap_attrib.market_cap_B < 50)]
            .count()['pnl'] / tot_count),
        ('more than \$50b', mcap_attrib[mcap_attrib.market_cap_B > 50]
            .count()['pnl'] / tot_count)
    ])
    market_cap_heights = market_cap_buckets.values()

    tot_pnl = mcap_attrib.pnl.sum()
    pnl_market_cap_buckets = OrderedDict((
        ('ETFs', mcap_attrib[pd.isnull(mcap_attrib.market_cap_B)]
            .pnl.sum() / tot_pnl),
        ('less than \$100m', mcap_attrib[mcap_attrib.market_cap_B < 0.1]
            .pnl.sum() / tot_pnl),
        ('\$100m - \$1b', mcap_attrib[(mcap_attrib.market_cap_B > 0.1)
                                      & (mcap_attrib.market_cap_B < 1)]
            .pnl.sum() / tot_pnl),
        ('\$1b - \$10b', mcap_attrib[(mcap_attrib.market_cap_B > 1)
                                     & (mcap_attrib.market_cap_B < 10)]
            .pnl.sum() / tot_pnl),
        ('\$10b - \$50b', mcap_attrib[(mcap_attrib.market_cap_B > 10)
                                      & (mcap_attrib.market_cap_B < 50)]
            .pnl.sum() / tot_pnl),
        ('more than \$50b', mcap_attrib[mcap_attrib.market_cap_B > 50]
            .pnl.sum() / tot_pnl)
    ))
    pnl_market_cap_heights = pnl_market_cap_buckets.values()

    pnl_colors = []
    for pnl in pnl_market_cap_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    fig, ax = plt.subplots(figsize=(14, 12))
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
    label_bars(bars1, market_cap_heights, 'left', ax)
    label_bars(bars2, pnl_market_cap_heights, 'right', ax)

    return fig


def compute_sector_pnl(round_trips, sectors):
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


def plot_sector_pnl(sector_attrib):
    tot_count = sector_attrib.count()['pnl']
    sector_buckets = OrderedDict((
        ('101 Basic Materials', sector_attrib[sector_attrib.sector == 101]
            .count()['pnl'] / tot_count),
        ('102 Consumer Cyclical', sector_attrib[sector_attrib.sector == 102]
            .count()['pnl'] / tot_count),
        ('103 Financial Services', sector_attrib[sector_attrib.sector == 103]
            .count()['pnl'] / tot_count),
        ('104 Real Estate', sector_attrib[sector_attrib.sector == 104]
            .count()['pnl'] / tot_count),
        ('205 Consumer Defensive', sector_attrib[sector_attrib.sector == 205]
            .count()['pnl'] / tot_count),
        ('206 Healthcare', sector_attrib[sector_attrib.sector == 206]
            .count()['pnl'] / tot_count),
        ('207 Utilities', sector_attrib[sector_attrib.sector == 207]
            .count()['pnl'] / tot_count),
        ('308 Communication Services', sector_attrib[sector_attrib.sector
                                                     == 308]
            .count()['pnl'] / tot_count),
        ('309 Energy', sector_attrib[sector_attrib.sector == 309]
            .count()['pnl'] / tot_count),
        ('310 Industrials', sector_attrib[sector_attrib.sector == 310]
            .count()['pnl'] / tot_count),
        ('311 Technology', sector_attrib[sector_attrib.sector == 311]
            .count()['pnl'] / tot_count)
    ))
    sector_heights = sector_buckets.values()

    tot_pnl = sector_attrib.pnl.sum()
    pnl_sector_buckets = OrderedDict((
        ('101 Basic Materials', sector_attrib[sector_attrib.sector == 101]
            .pnl.sum() / tot_pnl),
        ('102 Consumer Cyclical', sector_attrib[sector_attrib.sector == 102]
            .pnl.sum() / tot_pnl),
        ('103 Financial Services', sector_attrib[sector_attrib.sector == 103]
            .pnl.sum() / tot_pnl),
        ('104 Real Estate', sector_attrib[sector_attrib.sector == 104]
            .pnl.sum() / tot_pnl),
        ('205 Consumer Defensive', sector_attrib[sector_attrib.sector == 205]
            .pnl.sum() / tot_pnl),
        ('206 Healthcare', sector_attrib[sector_attrib.sector == 206]
            .pnl.sum() / tot_pnl),
        ('207 Utilities', sector_attrib[sector_attrib.sector == 207]
            .pnl.sum() / tot_pnl),
        ('308 Communication Services', sector_attrib[sector_attrib.sector
                                                     == 308]
            .pnl.sum() / tot_pnl),
        ('309 Energy', sector_attrib[sector_attrib.sector == 309]
            .pnl.sum() / tot_pnl),
        ('310 Industrials', sector_attrib[sector_attrib.sector == 310]
            .pnl.sum() / tot_pnl),
        ('311 Technology', sector_attrib[sector_attrib.sector == 311]
            .pnl.sum() / tot_pnl)
    ))
    pnl_sector_heights = pnl_sector_buckets.values()

    pnl_colors = []
    for pnl in pnl_sector_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    fig, ax = plt.subplots(figsize=(14, 12))
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

    label_bars(bars1, sector_heights, 'left', ax)
    label_bars(bars2, pnl_sector_heights, 'right', ax)

    return fig


def compute_q_univ_pnl(round_trips, q500, q1500):
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


def plot_q_univ_pnl(q_univ_attrib):
    tot_count = q_univ_attrib.count()['pnl']
    q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500].count().pnl
            / tot_count),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500)
                                | (q_univ_attrib.q_univ == 1500)].count().pnl
            / tot_count),
        ('All Q Securities', 1)
    ))
    q_univ_heights = q_univ_buckets.values()

    tot_pnl = q_univ_attrib.pnl.sum()
    pnl_q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500].pnl.sum()
            / tot_pnl),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500)
                                | (q_univ_attrib.q_univ == 1500)].pnl.sum()
            / tot_pnl),
        ('All Q Securities', 1)
    ))
    pnl_q_univ_heights = pnl_q_univ_buckets.values()

    pnl_colors = []
    for pnl in pnl_q_univ_buckets.values():
        if pnl < 0:
            pnl_colors.append('r')
        else:
            pnl_colors.append('g')

    fig, ax = plt.subplots(figsize=(14, 12))
    bars1 = ax.bar(list(range(3)), q_univ_buckets.values(),
                   color='orange', alpha=0.8, width=0.8,
                   label='Contribution to Total Number of Round Trips')
    bars2 = ax.bar(list(range(3)), pnl_q_univ_buckets.values(),
                   color=pnl_colors, alpha=0.8, width=0.6,
                   label='Contribution to Total PnL')

    ax.set_title('PnL Attribution by Q Universe (Cumulative Plot)',
                 fontsize='medium')
    ax.set_xlabel('Q Universe')
    ax.set_ylabel('Percentage Contribution')
    ax.legend()
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels(pnl_q_univ_buckets.keys())
    ax.set_yticklabels(['{:3.2f}%'.format(100*y) for y in ax.get_yticks()])

    label_bars(bars1, q_univ_heights, 'left', ax)
    label_bars(bars2, pnl_q_univ_heights, 'right', ax)

    return fig
