#
# Copyright 2018 Quantopian, Inc.
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

import warnings
from time import time

import empyrical as ep
from IPython.display import display, Markdown
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from . import _seaborn as sns
from . import capacity
from . import perf_attrib
from . import plotting
from . import pos
from . import risk
from . import round_trips
from . import timeseries
from . import txn
from . import utils

try:
    from . import bayesian
    have_bayesian = True
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)
    have_bayesian = False

FACTOR_PARTITIONS = {
    'style': ['momentum', 'size', 'value', 'reversal_short_term',
              'volatility'],
    'sector': ['basic_materials', 'consumer_cyclical', 'financial_services',
               'real_estate', 'consumer_defensive', 'health_care',
               'utilities', 'communication_services', 'energy', 'industrials',
               'technology']
}


def timer(msg_body, previous_time):
    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))

    return current_time


def create_full_tear_sheet(returns,
                           positions=None,
                           transactions=None,
                           market_data=None,
                           benchmark_rets=None,
                           slippage=None,
                           live_start_date=None,
                           sector_mappings=None,
                           bayesian=False,
                           round_trips=False,
                           estimate_intraday='infer',
                           hide_positions=False,
                           cone_std=(1.0, 1.5, 2.0),
                           bootstrap=False,
                           unadjusted_returns=None,
                           style_factor_panel=None,
                           sectors=None,
                           caps=None,
                           shares_held=None,
                           volumes=None,
                           percentile=None,
                           turnover_denom='AGB',
                           set_context=True,
                           factor_returns=None,
                           factor_loadings=None,
                           pos_in_dollars=True,
                           header_rows=None,
                           factor_partitions=FACTOR_PARTITIONS):
    """
    Generate a number of tear sheets that are useful
    for analyzing a strategy's performance.

    - Fetches benchmarks if needed.
    - Creates tear sheets for returns, and significant events.
        If possible, also creates tear sheets for position analysis,
        transaction analysis, and Bayesian analysis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'
    market_data : pd.Panel, optional
        Panel with items axis of 'price' and 'volume' DataFrames.
        The major and minor axes should match those of the
        the passed positions DataFrame (same dates and symbols).
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    bayesian: boolean, optional
        If True, causes the generation of a Bayesian tear sheet.
    round_trips: boolean, optional
        If True, causes the generation of a round trip tear sheet.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    estimate_intraday: boolean or str, optional
        Instead of using the end-of-day positions, use the point in the day
        where we have the most $ invested. This will adjust positions to
        better approximate and represent how an intraday strategy behaves.
        By default, this is 'infer', and an attempt will be made to detect
        an intraday strategy. Specifying this value will prevent detection.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    bootstrap : boolean (optional)
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.
    turnover_denom : str
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    factor_returns : pd.Dataframe, optional
        Returns by factor, with date as index and factors as columns
    factor_loadings : pd.Dataframe, optional
        Factor loadings for all days in the date range, with date and
        ticker as index, and factors as columns.
    pos_in_dollars : boolean, optional
        indicates whether positions is in dollars
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    set_context : boolean, optional
        If True, set default plotting style context.
         - See plotting.context().
    factor_partitions : dict, optional
        dict specifying how factors should be separated in perf attrib
        factor returns and risk exposures plots
        - See create_perf_attrib_tear_sheet().
    """

    if (unadjusted_returns is None) and (slippage is not None) and\
       (transactions is not None):
        unadjusted_returns = returns.copy()
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    create_returns_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        live_start_date=live_start_date,
        cone_std=cone_std,
        benchmark_rets=benchmark_rets,
        bootstrap=bootstrap,
        turnover_denom=turnover_denom,
        header_rows=header_rows,
        set_context=set_context)

    create_interesting_times_tear_sheet(returns,
                                        benchmark_rets=benchmark_rets,
                                        set_context=set_context)

    if positions is not None:
        create_position_tear_sheet(returns, positions,
                                   hide_positions=hide_positions,
                                   set_context=set_context,
                                   sector_mappings=sector_mappings,
                                   estimate_intraday=False)

        if transactions is not None:
            create_txn_tear_sheet(returns, positions, transactions,
                                  unadjusted_returns=unadjusted_returns,
                                  estimate_intraday=False,
                                  set_context=set_context)
            if round_trips:
                create_round_trip_tear_sheet(
                    returns=returns,
                    positions=positions,
                    transactions=transactions,
                    sector_mappings=sector_mappings,
                    estimate_intraday=False)

            if market_data is not None:
                create_capacity_tear_sheet(returns, positions, transactions,
                                           market_data,
                                           liquidation_daily_vol_limit=0.2,
                                           last_n_days=125,
                                           estimate_intraday=False)

        if style_factor_panel is not None:
            create_risk_tear_sheet(positions, style_factor_panel, sectors,
                                   caps, shares_held, volumes, percentile)

        if factor_returns is not None and factor_loadings is not None:
            create_perf_attrib_tear_sheet(returns, positions, factor_returns,
                                          factor_loadings, transactions,
                                          pos_in_dollars=pos_in_dollars,
                                          factor_partitions=factor_partitions)

    if bayesian:
        create_bayesian_tear_sheet(returns,
                                   live_start_date=live_start_date,
                                   benchmark_rets=benchmark_rets,
                                   set_context=set_context)


@plotting.customize
def create_simple_tear_sheet(returns,
                             positions=None,
                             transactions=None,
                             benchmark_rets=None,
                             slippage=None,
                             estimate_intraday='infer',
                             live_start_date=None,
                             turnover_denom='AGB',
                             header_rows=None):
    """
    Simpler version of create_full_tear_sheet; generates summary performance
    statistics and important plots as a single image.

    - Plots: cumulative returns, rolling beta, rolling Sharpe, underwater,
        exposure, top 10 holdings, total holdings, long/short holdings,
        daily turnover, transaction time distribution.
    - Never accept market_data input (market_data = None)
    - Never accept sector_mappings input (sector_mappings = None)
    - Never perform bootstrap analysis (bootstrap = False)
    - Never hide posistions on top 10 holdings plot (hide_positions = False)
    - Always use default cone_std (cone_std = (1.0, 1.5, 2.0))

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902
    positions : pd.DataFrame, optional
        Daily net position values.
         - Time series of dollar amount invested in each position and cash.
         - Days where stocks are not held can be represented by 0 or NaN.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'
    benchmark_rets : pd.Series, optional
        Daily returns of the benchmark, noncumulative.
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    set_context : boolean, optional
        If True, set default plotting style context.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if (slippage is not None) and (transactions is not None):
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    always_sections = 4
    positions_sections = 4 if positions is not None else 0
    transactions_sections = 2 if transactions is not None else 0
    live_sections = 1 if live_start_date is not None else 0
    benchmark_sections = 1 if benchmark_rets is not None else 0

    vertical_sections = sum([
        always_sections,
        positions_sections,
        transactions_sections,
        live_sections,
        benchmark_sections,
    ])

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    plotting.show_perf_stats(returns,
                             benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             live_start_date=live_start_date,
                             header_rows=header_rows)

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)

    ax_rolling_returns = plt.subplot(gs[:2, :])
    i = 2
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1

    plotting.plot_rolling_returns(returns,
                                  factor_returns=benchmark_rets,
                                  live_start_date=live_start_date,
                                  cone_std=(1.0, 1.5, 2.0),
                                  ax=ax_rolling_returns)
    ax_rolling_returns.set_title('Cumulative returns')

    if benchmark_rets is not None:
        plotting.plot_rolling_beta(returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_sharpe(returns, ax=ax_rolling_sharpe)

    plotting.plot_drawdown_underwater(returns, ax=ax_underwater)

    if positions is not None:
        # Plot simple positions tear sheet
        ax_exposures = plt.subplot(gs[i, :])
        i += 1
        ax_top_positions = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_holdings = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_long_short_holdings = plt.subplot(gs[i, :])
        i += 1

        positions_alloc = pos.get_percent_alloc(positions)

        plotting.plot_exposures(returns, positions, ax=ax_exposures)

        plotting.show_and_plot_top_positions(returns,
                                             positions_alloc,
                                             show_and_plot=0,
                                             hide_positions=False,
                                             ax=ax_top_positions)

        plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

        plotting.plot_long_short_holdings(returns, positions_alloc,
                                          ax=ax_long_short_holdings)

        if transactions is not None:
            # Plot simple transactions tear sheet
            ax_turnover = plt.subplot(gs[i, :])
            i += 1
            ax_txn_timings = plt.subplot(gs[i, :])
            i += 1

            plotting.plot_turnover(returns,
                                   transactions,
                                   positions,
                                   ax=ax_turnover)

            plotting.plot_txn_time_hist(transactions, ax=ax_txn_timings)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)


@plotting.customize
def create_returns_tear_sheet(returns, positions=None,
                              transactions=None,
                              live_start_date=None,
                              cone_std=(1.0, 1.5, 2.0),
                              benchmark_rets=None,
                              bootstrap=False,
                              turnover_denom='AGB',
                              header_rows=None,
                              return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's returns.

    - Fetches benchmarks, then creates the plots on a single figure.
    - Plots: rolling returns (with cone), rolling beta, rolling sharpe,
        rolling Fama-French risk factors, drawdowns, underwater plot, monthly
        and annual return plots, daily similarity plots,
        and return quantile box plot.
    - Will also print the start and end dates of the strategy,
        performance statistics, drawdown periods, and the return range.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.
        - See full explanation in create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

    plotting.show_perf_stats(returns, benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             bootstrap=bootstrap,
                             live_start_date=live_start_date,
                             header_rows=header_rows)

    plotting.show_worst_drawdown_periods(returns)

    vertical_sections = 11

    if live_start_date is not None:
        vertical_sections += 1
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    if benchmark_rets is not None:
        vertical_sections += 1

    if bootstrap:
        vertical_sections += 1

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])

    i = 2
    ax_rolling_returns_vol_match = plt.subplot(gs[i, :],
                                               sharex=ax_rolling_returns)
    i += 1
    ax_rolling_returns_log = plt.subplot(gs[i, :],
                                         sharex=ax_rolling_returns)
    i += 1
    ax_returns = plt.subplot(gs[i, :],
                             sharex=ax_rolling_returns)
    i += 1
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_volatility = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_drawdown = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_monthly_heatmap = plt.subplot(gs[i, 0])
    ax_annual_returns = plt.subplot(gs[i, 1])
    ax_monthly_dist = plt.subplot(gs[i, 2])
    i += 1
    ax_return_quantiles = plt.subplot(gs[i, :])
    i += 1

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)
    ax_rolling_returns.set_title(
        'Cumulative returns')

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=(benchmark_rets is not None),
        legend_loc=None,
        ax=ax_rolling_returns_vol_match)
    ax_rolling_returns_vol_match.set_title(
        'Cumulative returns volatility matched to benchmark')

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns_log)
    ax_rolling_returns_log.set_title(
        'Cumulative returns on logarithmic scale')

    plotting.plot_returns(
        returns,
        live_start_date=live_start_date,
        ax=ax_returns,
    )
    ax_returns.set_title(
        'Returns')

    if benchmark_rets is not None:
        plotting.plot_rolling_beta(
            returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, ax=ax_rolling_volatility)

    plotting.plot_rolling_sharpe(
        returns, ax=ax_rolling_sharpe)

    # Drawdowns
    plotting.plot_drawdown_periods(
        returns, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(
        returns=returns, ax=ax_underwater)

    plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    plotting.plot_return_quantiles(
        returns,
        live_start_date=live_start_date,
        ax=ax_return_quantiles)

    if bootstrap and (benchmark_rets is not None):
        ax_bootstrap = plt.subplot(gs[i, :])
        plotting.plot_perf_stats(returns, benchmark_rets,
                                 ax=ax_bootstrap)
    elif bootstrap:
        raise ValueError('bootstrap requires passing of benchmark_rets.')

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if return_fig:
        return fig


@plotting.customize
def create_position_tear_sheet(returns, positions,
                               show_and_plot_top_pos=2, hide_positions=False,
                               return_fig=False, sector_mappings=None,
                               transactions=None, estimate_intraday='infer'):
    """
    Generate a number of plots for analyzing a
    strategy's positions and holdings.

    - Plots: gross leverage, exposures, top positions, and holdings.
    - Will also print the top positions held.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    show_and_plot_top_pos : int, optional
        By default, this is 2, and both prints and plots the
        top 10 positions.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
        Overrides show_and_plot_top_pos to 0 to suppress text output.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if hide_positions:
        show_and_plot_top_pos = 0
    vertical_sections = 7 if sector_mappings is not None else 6

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_exposures = plt.subplot(gs[0, :])
    ax_top_positions = plt.subplot(gs[1, :], sharex=ax_exposures)
    ax_max_median_pos = plt.subplot(gs[2, :], sharex=ax_exposures)
    ax_holdings = plt.subplot(gs[3, :], sharex=ax_exposures)
    ax_long_short_holdings = plt.subplot(gs[4, :])
    ax_gross_leverage = plt.subplot(gs[5, :], sharex=ax_exposures)

    positions_alloc = pos.get_percent_alloc(positions)

    plotting.plot_exposures(returns, positions, ax=ax_exposures)

    plotting.show_and_plot_top_positions(
        returns,
        positions_alloc,
        show_and_plot=show_and_plot_top_pos,
        hide_positions=hide_positions,
        ax=ax_top_positions)

    plotting.plot_max_median_position_concentration(positions,
                                                    ax=ax_max_median_pos)

    plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

    plotting.plot_long_short_holdings(returns, positions_alloc,
                                      ax=ax_long_short_holdings)

    plotting.plot_gross_leverage(returns, positions,
                                 ax=ax_gross_leverage)

    if sector_mappings is not None:
        sector_exposures = pos.get_sector_exposures(positions,
                                                    sector_mappings)
        if len(sector_exposures.columns) > 1:
            sector_alloc = pos.get_percent_alloc(sector_exposures)
            sector_alloc = sector_alloc.drop('cash', axis='columns')
            ax_sector_alloc = plt.subplot(gs[6, :], sharex=ax_exposures)
            plotting.plot_sector_allocations(returns, sector_alloc,
                                             ax=ax_sector_alloc)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if return_fig:
        return fig


@plotting.customize
def create_txn_tear_sheet(returns, positions, transactions,
                          unadjusted_returns=None, estimate_intraday='infer',
                          return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's transactions.

    Plots: turnover, daily volume, and a histogram of daily volume.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    unadjusted_returns : pd.Series, optional
        Daily unadjusted returns of the strategy, noncumulative.
        Will plot additional swippage sweep analysis.
         - See pyfolio.plotting.plot_swippage_sleep and
           pyfolio.plotting.plot_slippage_sensitivity
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    vertical_sections = 6 if unadjusted_returns is not None else 4

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_turnover = plt.subplot(gs[0, :])
    ax_daily_volume = plt.subplot(gs[1, :], sharex=ax_turnover)
    ax_turnover_hist = plt.subplot(gs[2, :])
    ax_txn_timings = plt.subplot(gs[3, :])

    plotting.plot_turnover(
        returns,
        transactions,
        positions,
        ax=ax_turnover)

    plotting.plot_daily_volume(returns, transactions, ax=ax_daily_volume)

    try:
        plotting.plot_daily_turnover_hist(transactions, positions,
                                          ax=ax_turnover_hist)
    except ValueError:
        warnings.warn('Unable to generate turnover plot.', UserWarning)

    plotting.plot_txn_time_hist(transactions, ax=ax_txn_timings)

    if unadjusted_returns is not None:
        ax_slippage_sweep = plt.subplot(gs[4, :])
        plotting.plot_slippage_sweep(unadjusted_returns,
                                     positions,
                                     transactions,
                                     ax=ax_slippage_sweep
                                     )
        ax_slippage_sensitivity = plt.subplot(gs[5, :])
        plotting.plot_slippage_sensitivity(unadjusted_returns,
                                           positions,
                                           transactions,
                                           ax=ax_slippage_sensitivity
                                           )
    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if return_fig:
        return fig


@plotting.customize
def create_round_trip_tear_sheet(returns, positions, transactions,
                                 sector_mappings=None,
                                 estimate_intraday='infer', return_fig=False):
    """
    Generate a number of figures and plots describing the duration,
    frequency, and profitability of trade "round trips."
    A round trip is started when a new long or short position is
    opened and is only completed when the number of shares in that
    position returns to or crosses zero.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    transactions_closed = round_trips.add_closing_transactions(positions,
                                                               transactions)
    # extract_round_trips requires BoD portfolio_value
    trades = round_trips.extract_round_trips(
        transactions_closed,
        portfolio_value=positions.sum(axis='columns') / (1 + returns)
    )

    if len(trades) < 5:
        warnings.warn(
            """Fewer than 5 round-trip trades made.
               Skipping round trip tearsheet.""", UserWarning)
        return

    round_trips.print_round_trip_stats(trades)

    plotting.show_profit_attribution(trades)

    if sector_mappings is not None:
        sector_trades = round_trips.apply_sector_mappings_to_round_trips(
            trades, sector_mappings)
        plotting.show_profit_attribution(sector_trades)

    fig = plt.figure(figsize=(14, 3 * 6))

    gs = gridspec.GridSpec(3, 2, wspace=0.5, hspace=0.5)

    ax_trade_lifetimes = plt.subplot(gs[0, :])
    ax_prob_profit_trade = plt.subplot(gs[1, 0])
    ax_holding_time = plt.subplot(gs[1, 1])
    ax_pnl_per_round_trip_dollars = plt.subplot(gs[2, 0])
    ax_pnl_per_round_trip_pct = plt.subplot(gs[2, 1])

    plotting.plot_round_trip_lifetimes(trades, ax=ax_trade_lifetimes)

    plotting.plot_prob_profit_trade(trades, ax=ax_prob_profit_trade)

    trade_holding_times = [x.days for x in trades['duration']]
    sns.distplot(trade_holding_times, kde=False, ax=ax_holding_time)
    ax_holding_time.set(xlabel='Holding time in days')

    sns.distplot(trades.pnl, kde=False, ax=ax_pnl_per_round_trip_dollars)
    ax_pnl_per_round_trip_dollars.set(xlabel='PnL per round-trip trade in $')

    sns.distplot(trades.returns.dropna() * 100, kde=False,
                 ax=ax_pnl_per_round_trip_pct)
    ax_pnl_per_round_trip_pct.set(
        xlabel='Round-trip returns in %')

    gs.tight_layout(fig)

    if return_fig:
        return fig


@plotting.customize
def create_interesting_times_tear_sheet(
        returns, benchmark_rets=None, legend_loc='best', return_fig=False):
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    benchmark_rets must be passed, as it is meaningless to analyze performance
    during interesting times without some benchmark to refer to.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    rets_interesting = timeseries.extract_interesting_date_ranges(returns)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    utils.print_table(pd.DataFrame(rets_interesting)
                      .describe().transpose()
                      .loc[:, ['mean', 'min', 'max']] * 100,
                      name='Stress Events',
                      float_format='{0:.2f}%'.format)

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

        bmark_interesting = timeseries.extract_interesting_date_ranges(
            benchmark_rets)

    num_plots = len(rets_interesting)
    # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    num_rows = int((num_plots + 1) / 2.0)
    fig = plt.figure(figsize=(14, num_rows * 6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.items()):
        # i=0 -> 0, i=1 -> 0, i=2 -> 1 ;; i=0 -> 0, i=1 -> 1, i=2 -> 0
        ax = plt.subplot(gs[int(i / 2.0), i % 2])

        ep.cum_returns(rets_period).plot(
            ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)

        if benchmark_rets is not None:
            ep.cum_returns(bmark_interesting[name]).plot(
                ax=ax, color='gray', label='benchmark', alpha=0.6)
            ax.legend(['Algo',
                       'benchmark'],
                      loc=legend_loc, frameon=True, framealpha=0.5)
        else:
            ax.legend(['Algo'],
                      loc=legend_loc, frameon=True, framealpha=0.5)

        ax.set_title(name)
        ax.set_ylabel('Returns')
        ax.set_xlabel('')

    if return_fig:
        return fig


@plotting.customize
def create_capacity_tear_sheet(returns, positions, transactions,
                               market_data,
                               liquidation_daily_vol_limit=0.2,
                               trade_daily_vol_limit=0.05,
                               last_n_days=utils.APPROX_BDAYS_PER_MONTH * 6,
                               days_to_liquidate_limit=1,
                               estimate_intraday='infer'):
    """
    Generates a report detailing portfolio size constraints set by
    least liquid tickers. Plots a "capacity sweep," a curve describing
    projected sharpe ratio given the slippage penalties that are
    applied at various capital bases.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    market_data : pd.Panel
        Panel with items axis of 'price' and 'volume' DataFrames.
        The major and minor axes should match those of the
        the passed positions DataFrame (same dates and symbols).
    liquidation_daily_vol_limit : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position in the
        "days to liquidation" analysis.
    trade_daily_vol_limit : float
        Flag daily transaction totals that exceed proportion of
        daily bar.
    last_n_days : integer
        Compute max position allocation and dollar volume for only
        the last N days of the backtest
    days_to_liquidate_limit : integer
        Display all tickers with greater max days to liquidation.
    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in create_full_tear_sheet.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    print("Max days to liquidation is computed for each traded name "
          "assuming a 20% limit on daily bar consumption \n"
          "and trailing 5 day mean volume as the available bar volume.\n\n"
          "Tickers with >1 day liquidation time at a"
          " constant $1m capital base:")

    max_days_by_ticker = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5)
    max_days_by_ticker.index = (
        max_days_by_ticker.index.map(utils.format_asset))

    print("Whole backtest:")
    utils.print_table(
        max_days_by_ticker[max_days_by_ticker.days_to_liquidate >
                           days_to_liquidate_limit])

    max_days_by_ticker_lnd = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5,
        last_n_days=last_n_days)
    max_days_by_ticker_lnd.index = (
        max_days_by_ticker_lnd.index.map(utils.format_asset))

    print("Last {} trading days:".format(last_n_days))
    utils.print_table(
        max_days_by_ticker_lnd[max_days_by_ticker_lnd.days_to_liquidate > 1])

    llt = capacity.get_low_liquidity_transactions(transactions, market_data)
    llt.index = llt.index.map(utils.format_asset)

    print('Tickers with daily transactions consuming >{}% of daily bar \n'
          'all backtest:'.format(trade_daily_vol_limit * 100))
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    llt = capacity.get_low_liquidity_transactions(
        transactions, market_data, last_n_days=last_n_days)

    print("Last {} trading days:".format(last_n_days))
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    bt_starting_capital = positions.iloc[0].sum() / (1 + returns.iloc[0])
    fig, ax_capacity_sweep = plt.subplots(figsize=(14, 6))
    plotting.plot_capacity_sweep(returns, transactions, market_data,
                                 bt_starting_capital,
                                 min_pv=100000,
                                 max_pv=300000000,
                                 step_size=1000000,
                                 ax=ax_capacity_sweep)


@plotting.customize
def create_bayesian_tear_sheet(returns, benchmark_rets=None,
                               live_start_date=None, samples=2000,
                               return_fig=False, stoch_vol=False,
                               progressbar=True):
    """
    Generate a number of Bayesian distributions and a Bayesian
    cone plot of returns.

    Plots: Sharpe distribution, annual volatility distribution,
    annual alpha distribution, beta distribution, predicted 1 and 5
    day returns distributions, and a cumulative returns cone plot.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The point in time when the strategy began live
        trading, after its backtest period.
    samples : int, optional
        Number of posterior samples to draw.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    stoch_vol : boolean, optional
        If True, run and plot the stochastic volatility model
    progressbar : boolean, optional
        If True, show a progress bar
    """

    if not have_bayesian:
        raise NotImplementedError(
            "Bayesian tear sheet requirements not found.\n"
            "Run 'pip install pyfolio[bayesian]' to install "
            "bayesian requirements."
        )

    if live_start_date is None:
        raise NotImplementedError(
            'Bayesian tear sheet requires setting of live_start_date'
        )

    live_start_date = ep.utils.get_utc_timestamp(live_start_date)
    df_train = returns.loc[returns.index < live_start_date]
    df_test = returns.loc[returns.index >= live_start_date]

    # Run T model with missing data
    print("Running T model")
    previous_time = time()
    # track the total run time of the Bayesian tear sheet
    start_time = previous_time

    trace_t, ppc_t = bayesian.run_model('t', df_train,
                                        returns_test=df_test,
                                        samples=samples, ppc=True,
                                        progressbar=progressbar)
    previous_time = timer("T model", previous_time)

    # Compute BEST model
    print("\nRunning BEST model")
    trace_best = bayesian.run_model('best', df_train,
                                    returns_test=df_test,
                                    samples=samples,
                                    progressbar=progressbar)
    previous_time = timer("BEST model", previous_time)

    # Plot results

    fig = plt.figure(figsize=(14, 10 * 2))
    gs = gridspec.GridSpec(9, 2, wspace=0.3, hspace=0.3)

    axs = []
    row = 0

    # Plot Bayesian cone
    ax_cone = plt.subplot(gs[row, :])
    bayesian.plot_bayes_cone(df_train, df_test, ppc_t, ax=ax_cone)
    previous_time = timer("plotting Bayesian cone", previous_time)

    # Plot BEST results
    row += 1
    axs.append(plt.subplot(gs[row, 0]))
    axs.append(plt.subplot(gs[row, 1]))
    row += 1
    axs.append(plt.subplot(gs[row, 0]))
    axs.append(plt.subplot(gs[row, 1]))
    row += 1
    axs.append(plt.subplot(gs[row, 0]))
    axs.append(plt.subplot(gs[row, 1]))
    row += 1
    # Effect size across two
    axs.append(plt.subplot(gs[row, :]))

    bayesian.plot_best(trace=trace_best, axs=axs)
    previous_time = timer("plotting BEST results", previous_time)

    # Compute Bayesian predictions
    row += 1
    ax_ret_pred_day = plt.subplot(gs[row, 0])
    ax_ret_pred_week = plt.subplot(gs[row, 1])
    day_pred = ppc_t[:, 0]
    p5 = scipy.stats.scoreatpercentile(day_pred, 5)
    sns.distplot(day_pred,
                 ax=ax_ret_pred_day
                 )
    ax_ret_pred_day.axvline(p5, linestyle='--', linewidth=3.)
    ax_ret_pred_day.set_xlabel('Predicted returns 1 day')
    ax_ret_pred_day.set_ylabel('Frequency')
    ax_ret_pred_day.text(0.4, 0.9, 'Bayesian VaR = %.2f' % p5,
                         verticalalignment='bottom',
                         horizontalalignment='right',
                         transform=ax_ret_pred_day.transAxes)
    previous_time = timer("computing Bayesian predictions", previous_time)

    # Plot Bayesian VaRs
    week_pred = (
        np.cumprod(ppc_t[:, :5] + 1, 1) - 1)[:, -1]
    p5 = scipy.stats.scoreatpercentile(week_pred, 5)
    sns.distplot(week_pred,
                 ax=ax_ret_pred_week
                 )
    ax_ret_pred_week.axvline(p5, linestyle='--', linewidth=3.)
    ax_ret_pred_week.set_xlabel('Predicted cum returns 5 days')
    ax_ret_pred_week.set_ylabel('Frequency')
    ax_ret_pred_week.text(0.4, 0.9, 'Bayesian VaR = %.2f' % p5,
                          verticalalignment='bottom',
                          horizontalalignment='right',
                          transform=ax_ret_pred_week.transAxes)
    previous_time = timer("plotting Bayesian VaRs estimate", previous_time)

    # Run alpha beta model
    if benchmark_rets is not None:
        print("\nRunning alpha beta model")
        benchmark_rets = benchmark_rets.loc[df_train.index]
        trace_alpha_beta = bayesian.run_model('alpha_beta', df_train,
                                              bmark=benchmark_rets,
                                              samples=samples,
                                              progressbar=progressbar)
        previous_time = timer("running alpha beta model", previous_time)

        # Plot alpha and beta
        row += 1
        ax_alpha = plt.subplot(gs[row, 0])
        ax_beta = plt.subplot(gs[row, 1])
        sns.distplot((1 + trace_alpha_beta['alpha'][100:])**252 - 1,
                     ax=ax_alpha)
        sns.distplot(trace_alpha_beta['beta'][100:], ax=ax_beta)
        ax_alpha.set_xlabel('Annual Alpha')
        ax_alpha.set_ylabel('Belief')
        ax_beta.set_xlabel('Beta')
        ax_beta.set_ylabel('Belief')
        previous_time = timer("plotting alpha beta model", previous_time)

    if stoch_vol:
        # run stochastic volatility model
        returns_cutoff = 400
        print(
            "\nRunning stochastic volatility model on "
            "most recent {} days of returns.".format(returns_cutoff)
        )
        if df_train.size > returns_cutoff:
            df_train_truncated = df_train[-returns_cutoff:]
        _, trace_stoch_vol = bayesian.model_stoch_vol(df_train_truncated)
        previous_time = timer(
            "running stochastic volatility model", previous_time)

        # plot latent volatility
        row += 1
        ax_volatility = plt.subplot(gs[row, :])
        bayesian.plot_stoch_vol(
            df_train_truncated, trace=trace_stoch_vol, ax=ax_volatility)
        previous_time = timer(
            "plotting stochastic volatility model", previous_time)

    total_time = time() - start_time
    print("\nTotal runtime was {:.2f} seconds.".format(total_time))

    gs.tight_layout(fig)

    if return_fig:
        return fig


@plotting.customize
def create_risk_tear_sheet(positions,
                           style_factor_panel=None,
                           sectors=None,
                           caps=None,
                           shares_held=None,
                           volumes=None,
                           percentile=None,
                           returns=None,
                           transactions=None,
                           estimate_intraday='infer',
                           return_fig=False):
    '''
    Creates risk tear sheet: computes and plots style factor exposures, sector
    exposures, market cap exposures and volume exposures.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - DataFrame with dates as index, equities as columns
        - Last column is cash held
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])             cash
        2017-04-03	-108062.40 	  4401.540     2.247757e+07
        2017-04-04	-108852.00	  4373.820     2.540999e+07
        2017-04-05	-119968.66	  4336.200     2.839812e+07

    style_factor_panel : pd.Panel
        Panel where each item is a DataFrame that tabulates style factor per
        equity per day.
        - Each item has dates as index, equities as columns
        - Example item:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	  -0.51284     1.39173
        2017-04-04	  -0.73381     0.98149
        2017-04-05	  -0.90132	   1.13981

    sectors : pd.DataFrame
        Daily Morningstar sector code per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	     311.0       206.0
        2017-04-04	     311.0       206.0
        2017-04-05	     311.0	     206.0

    caps : pd.DataFrame
        Daily market cap per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03     1.327160e+10     6.402460e+10
        2017-04-04	   1.329620e+10     6.403694e+10
        2017-04-05	   1.297464e+10	    6.397187e+10

    shares_held : pd.DataFrame
        Daily number of shares held by an algorithm.
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03             1915            -2595
        2017-04-04	           1968            -3272
        2017-04-05	           2104            -3917

    volumes : pd.DataFrame
        Daily volume per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03      34940859.00       4665573.80
        2017-04-04	    35603329.10       4818463.90
        2017-04-05	    41846731.75	      4129153.10

    percentile : float
        Percentile to use when computing and plotting volume exposures.
        - Defaults to 10th percentile
    '''

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    idx = positions.index & style_factor_panel.iloc[0].index & sectors.index \
        & caps.index & shares_held.index & volumes.index
    positions = positions.loc[idx]

    vertical_sections = 0
    if style_factor_panel is not None:
        vertical_sections += len(style_factor_panel.items)
        new_style_dict = {}
        for item in style_factor_panel.items:
            new_style_dict.update({item:
                                   style_factor_panel.loc[item].loc[idx]})
        style_factor_panel = pd.Panel()
        style_factor_panel = style_factor_panel.from_dict(new_style_dict)
    if sectors is not None:
        vertical_sections += 4
        sectors = sectors.loc[idx]
    if caps is not None:
        vertical_sections += 4
        caps = caps.loc[idx]
    if (shares_held is not None) & (volumes is not None) \
                                 & (percentile is not None):
        vertical_sections += 3
        shares_held = shares_held.loc[idx]
        volumes = volumes.loc[idx]

    if percentile is None:
        percentile = 0.1

    fig = plt.figure(figsize=[14, vertical_sections * 6])
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)

    if style_factor_panel is not None:
        style_axes = []
        style_axes.append(plt.subplot(gs[0, :]))
        for i in range(1, len(style_factor_panel.items)):
            style_axes.append(plt.subplot(gs[i, :], sharex=style_axes[0]))

        j = 0
        for name, df in style_factor_panel.iteritems():
            sfe = risk.compute_style_factor_exposures(positions, df)
            risk.plot_style_factor_exposures(sfe, name, style_axes[j])
            j += 1

    if sectors is not None:
        i += 1
        ax_sector_longshort = plt.subplot(gs[i:i+2, :], sharex=style_axes[0])
        i += 2
        ax_sector_gross = plt.subplot(gs[i, :], sharex=style_axes[0])
        i += 1
        ax_sector_net = plt.subplot(gs[i, :], sharex=style_axes[0])
        long_exposures, short_exposures, gross_exposures, net_exposures \
            = risk.compute_sector_exposures(positions, sectors)
        risk.plot_sector_exposures_longshort(long_exposures, short_exposures,
                                             ax=ax_sector_longshort)
        risk.plot_sector_exposures_gross(gross_exposures, ax=ax_sector_gross)
        risk.plot_sector_exposures_net(net_exposures, ax=ax_sector_net)

    if caps is not None:
        i += 1
        ax_cap_longshort = plt.subplot(gs[i:i+2, :], sharex=style_axes[0])
        i += 2
        ax_cap_gross = plt.subplot(gs[i, :], sharex=style_axes[0])
        i += 1
        ax_cap_net = plt.subplot(gs[i, :], sharex=style_axes[0])
        long_exposures, short_exposures, gross_exposures, net_exposures \
            = risk.compute_cap_exposures(positions, caps)
        risk.plot_cap_exposures_longshort(long_exposures, short_exposures,
                                          ax_cap_longshort)
        risk.plot_cap_exposures_gross(gross_exposures, ax_cap_gross)
        risk.plot_cap_exposures_net(net_exposures, ax_cap_net)

    if volumes is not None:
        i += 1
        ax_vol_longshort = plt.subplot(gs[i:i+2, :], sharex=style_axes[0])
        i += 2
        ax_vol_gross = plt.subplot(gs[i, :], sharex=style_axes[0])
        longed_threshold, shorted_threshold, grossed_threshold \
            = risk.compute_volume_exposures(positions, volumes, percentile)
        risk.plot_volume_exposures_longshort(longed_threshold,
                                             shorted_threshold, percentile,
                                             ax_vol_longshort)
        risk.plot_volume_exposures_gross(grossed_threshold, percentile,
                                         ax_vol_gross)

    for ax in fig.axes:
        plt.setp(ax.get_xticklabels(), visible=True)

    if return_fig:
        return fig


@plotting.customize
def create_perf_attrib_tear_sheet(returns,
                                  positions,
                                  factor_returns,
                                  factor_loadings,
                                  transactions=None,
                                  pos_in_dollars=True,
                                  return_fig=False,
                                  factor_partitions=FACTOR_PARTITIONS):
    """
    Generate plots and tables for analyzing a strategy's performance.

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
        Factor loadings for all days in the date range, with date
        and ticker as index, and factors as columns.

    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
         - Default is None.

    pos_in_dollars : boolean, optional
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.

    factor_partitions : dict
        dict specifying how factors should be separated in factor returns
        and risk exposures plots
        - Example:
          {'style': ['momentum', 'size', 'value', ...],
           'sector': ['technology', 'materials', ... ]}
    """
    portfolio_exposures, perf_attrib_data = perf_attrib.perf_attrib(
        returns, positions, factor_returns, factor_loadings, transactions,
        pos_in_dollars=pos_in_dollars
    )

    display(Markdown("## Performance Relative to Common Risk Factors"))

    # aggregate perf attrib stats and show summary table
    perf_attrib.show_perf_attrib_stats(returns, positions, factor_returns,
                                       factor_loadings, transactions,
                                       pos_in_dollars)

    # one section for the returns plot, and for each factor grouping
    # one section for factor returns, and one for risk exposures
    vertical_sections = 1 + 2 * max(len(factor_partitions), 1)
    current_section = 0

    fig = plt.figure(figsize=[14, vertical_sections * 6])

    gs = gridspec.GridSpec(vertical_sections, 1,
                           wspace=0.5, hspace=0.5)

    perf_attrib.plot_returns(perf_attrib_data,
                             ax=plt.subplot(gs[current_section]))
    current_section += 1

    if factor_partitions is not None:

        for factor_type, partitions in factor_partitions.iteritems():

            columns_to_select = perf_attrib_data.columns.intersection(
                partitions
            )

            perf_attrib.plot_factor_contribution_to_perf(
                perf_attrib_data[columns_to_select],
                ax=plt.subplot(gs[current_section]),
                title=(
                    'Cumulative common {} returns attribution'
                ).format(factor_type)
            )
            current_section += 1

        for factor_type, partitions in factor_partitions.iteritems():

            perf_attrib.plot_risk_exposures(
                portfolio_exposures[portfolio_exposures.columns
                                    .intersection(partitions)],
                ax=plt.subplot(gs[current_section]),
                title='Daily {} factor exposures'.format(factor_type)
            )
            current_section += 1

    else:

        perf_attrib.plot_factor_contribution_to_perf(
            perf_attrib_data,
            ax=plt.subplot(gs[current_section])
        )
        current_section += 1

        perf_attrib.plot_risk_exposures(
            portfolio_exposures,
            ax=plt.subplot(gs[current_section])
        )

    gs.tight_layout(fig)

    if return_fig:
        return fig
