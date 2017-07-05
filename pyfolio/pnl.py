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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict


def label_bars(bars, heights, style='left', ax=None):
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


def plot_mcap_pnl(mcap_attrib):
    '''
    Plots output of compute_mcap_pnl as a bar chart

    Parameters
    ----------
    mcap_attrib : pd.DataFrame
        PnL attribution by market cap
        - Exact output of compute_mcap_pnl()
    '''

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
    pnl_market_cap_buckets = OrderedDict([
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
    ])
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


def plot_sector_pnl(sector_attrib):
    '''
    Plots output of compute_sector_pnl as a bar chart

    Parameters
    ----------
    sector_attrib : pd.DataFrame
        PnL attribution by sector
        - Exact output of compute_sector_pnl()
    '''

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


def plot_q_univ_pnl(q_univ_attrib):
    '''
    Plots output of compute_q_univ_pnl as a bar chart

    Parameters
    ----------
    q_univ_attrib : pd.DataFrame
        PnL attribution by Q universe
        - Exact output of compute_q_univ_pnl()
    '''

    tot_count = q_univ_attrib.count()['pnl']
    q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500]
            .count().pnl / tot_count),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500)
                                | (q_univ_attrib.q_univ == 1500)]
            .count().pnl / tot_count),
        ('All Q Securities', 1)
    ))
    q_univ_heights = q_univ_buckets.values()

    tot_pnl = q_univ_attrib.pnl.sum()
    pnl_q_univ_buckets = OrderedDict((
        ('Q500', q_univ_attrib[q_univ_attrib.q_univ == 500]
            .pnl.sum() / tot_pnl),
        ('Q1500', q_univ_attrib[(q_univ_attrib.q_univ == 500)
                                | (q_univ_attrib.q_univ == 1500)]
            .pnl.sum() / tot_pnl),
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


def compute_price_pnl(round_trips, pricing):
    '''
    Computes PnL breakdown by entering price (i.e. the price of the stock at
    the opening of the round trip)

    Parameters
    ----------
    round_trips : pd.DataFrame
        Round trips of algorithm, including open_dt, pnl and symbol
        - Output of pyfolio.round_trips.extract_round_trips()

    pricing : pd.DataFrame
        - Example:
                                      Equity(2 [ARNC])	    Equity(24 [AAPL])
        2015-01-05 00:00:00+00:00	  43.645	            134.350
        2015-01-06 00:00:00+00:00	  59.747	            101.001
        2015-01-07 00:00:00+00:00	  94.234	            197.932
    '''

    price_attrib = pd.DataFrame()

    for row in round_trips.sort_values('symbol').iterrows():
        # Set up some variables
        day = row[1].open_dt
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


def plot_price_pnl(price_attrib):
    '''
    Plots output of compute_price_pnl as a bar chart

    Parameters
    ----------
    price_attrib : pd.DataFrame
        PnL attribution by entering price of round trip
        - Exact output of compute_price_pnl()
    '''

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

    fig, ax = plt.subplots(figsize=(14, 12))
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

    label_bars(bars1, price_heights, 'left', ax)
    label_bars(bars2, pnl_price_heights, 'right', ax)

    return fig
