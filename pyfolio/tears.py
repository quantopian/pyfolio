#
# Copyright 2015 Quantopian, Inc.
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

from . import timeseries
from . import utils
from . import pos
from . import txn
from . import plotting
from .plotting import plotting_context

try:
    from . import bayesian
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)

import numpy as np
import scipy.stats
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from time import time

def timer(msg_body, previous_time):

    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))

    return current_time


def create_full_tear_sheet(returns, positions=None, transactions=None,
                           benchmark_rets=None,
                           gross_lev=None,
                           slippage=None,
                           live_start_date=None, bayesian=False,
                           hide_positions=False,
                           sector_mappings=None,
                           cone_std=(1.0, 1.5, 2.0), set_context=True):
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
        Daily transaction quantity (txn_shares) and dollar amount
        (txn_volume).
        - Example:
            index         txn_volume      txn_shares
            2004-01-09    99288.441805    6361
            2004-01-12    1226.039520     102
            2004-01-13    752.354630      73
    gross_lev : pd.Series, optional
        The leverage of a strategy.
         - Time series of the sum of long and short exposure per share
            divided by net asset value.
         - Example:
            2009-12-04    0.999932
            2009-12-07    0.999783
            2009-12-08    0.999880
            2009-12-09    1.000283
    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.
        - See txn.adjust_returns_for_slippage for more details.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    bayesian: boolean, optional
        If True, causes the generation of a Bayesian tear sheet.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    set_context : boolean, optional
        If True, set default plotting style context.
         - See plotting.context().
    """

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')

    # If the strategy's history is longer than the benchmark's, limit strategy
    if returns.index[0] < benchmark_rets.index[0]:
        returns = returns[returns.index > benchmark_rets.index[0]]

    if slippage is not None and transactions is not None:
        turnover = txn.get_turnover(transactions, positions,
                                    period=None, average=False)
        unadjusted_returns = returns.copy()
        returns = txn.adjust_returns_for_slippage(returns, turnover, slippage)
    else:
        unadjusted_returns = None

    create_returns_tear_sheet(
        returns,
        live_start_date=live_start_date,
        cone_std=cone_std,
        benchmark_rets=benchmark_rets,
        set_context=set_context
    )

    create_interesting_times_tear_sheet(returns,
                                        benchmark_rets=benchmark_rets,
                                        set_context=set_context)

    if positions is not None:
        create_position_tear_sheet(returns, positions,
                                   gross_lev=gross_lev,
                                   hide_positions=hide_positions,
                                   set_context=set_context)

        if transactions is not None:
            create_txn_tear_sheet(returns, positions, transactions,
                                  unadjusted_returns=unadjusted_returns,
                                  set_context=set_context)

    if bayesian:
        create_bayesian_tear_sheet(returns,
                                   live_start_date=live_start_date,
                                   benchmark_rets=benchmark_rets,
                                   set_context=set_context)


@plotting_context
def create_returns_tear_sheet(returns, live_start_date=None,
                              cone_std=(1.0, 1.5, 2.0),
                              benchmark_rets=None,
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
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    set_context : boolean, optional
        If True, set default plotting style context.
    """

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')
        # If the strategy's history is longer than the benchmark's, limit
        # strategy
        if returns.index[0] < benchmark_rets.index[0]:
            returns = returns[returns.index > benchmark_rets.index[0]]

    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    print("Entire data start date: " + str(df_cum_rets
                                           .index[0].strftime('%Y-%m-%d')))
    print("Entire data end date: " + str(df_cum_rets
                                         .index[-1].strftime('%Y-%m-%d')))

    print('\n')

    plotting.show_perf_stats(returns, benchmark_rets,
                             live_start_date=live_start_date)

    if live_start_date is not None:
        vertical_sections = 11
        live_start_date = utils.get_utc_timestamp(live_start_date)
    else:
        vertical_sections = 10

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])
    ax_rolling_returns_vol_match = plt.subplot(gs[2, :],
                                               sharex=ax_rolling_returns)
    ax_rolling_beta = plt.subplot(gs[3, :], sharex=ax_rolling_returns)
    ax_rolling_sharpe = plt.subplot(gs[4, :], sharex=ax_rolling_returns)
    ax_rolling_risk = plt.subplot(gs[5, :], sharex=ax_rolling_returns)
    ax_drawdown = plt.subplot(gs[6, :], sharex=ax_rolling_returns)
    ax_underwater = plt.subplot(gs[7, :], sharex=ax_rolling_returns)
    ax_monthly_heatmap = plt.subplot(gs[8, 0])
    ax_annual_returns = plt.subplot(gs[8, 1])
    ax_monthly_dist = plt.subplot(gs[8, 2])
    ax_return_quantiles = plt.subplot(gs[9, :])

    if live_start_date is not None:
        ax_daily_similarity_scale = plt.subplot(gs[10, 0])
        ax_daily_similarity_no_var = plt.subplot(gs[10, 1])
        ax_daily_similarity_no_var_no_mean = plt.subplot(gs[10, 2])

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=True,
        ax=ax_rolling_returns_vol_match)
    ax_rolling_returns_vol_match.set_title(
        'Cumulative returns volatility matched to benchmark.')

    plotting.plot_rolling_beta(
        returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_sharpe(
        returns, ax=ax_rolling_sharpe)

    plotting.plot_rolling_fama_french(
        returns, ax=ax_rolling_risk)

    # Drawdowns
    plotting.plot_drawdown_periods(
        returns, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(
        returns=returns, ax=ax_underwater)

    plotting.show_worst_drawdown_periods(returns)

    df_weekly = timeseries.aggregate_returns(returns, 'weekly')
    df_monthly = timeseries.aggregate_returns(returns, 'monthly')

    print('\n')
    plotting.show_return_range(returns, df_weekly)

    plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    plotting.plot_return_quantiles(
        returns,
        df_weekly,
        df_monthly,
        ax=ax_return_quantiles)

    if live_start_date is not None:
        returns_backtest = returns[returns.index < live_start_date]
        returns_live = returns[returns.index > live_start_date]

        plotting.plot_daily_returns_similarity(
            returns_backtest,
            returns_live,
            title='Daily Returns Similarity',
            ax=ax_daily_similarity_scale)
        plotting.plot_daily_returns_similarity(
            returns_backtest,
            returns_live,
            scale_kws={'with_std': False},
            title='Similarity without\nvariance normalization',
            ax=ax_daily_similarity_no_var)
        plotting.plot_daily_returns_similarity(
            returns_backtest,
            returns_live,
            scale_kws={'with_std': False,
                       'with_mean': False},
            title='Similarity without variance\nand mean normalization',
            ax=ax_daily_similarity_no_var_no_mean)

    plt.show()
    if return_fig:
        return fig


@plotting_context
def create_position_tear_sheet(returns, positions, gross_lev=None,
                               show_and_plot_top_pos=2, hide_positions=False,
                               return_fig=False, sector_mappings=None):
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
    gross_lev : pd.Series, optional
        The leverage of a strategy.
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
    set_context : boolean, optional
        If True, set default plotting style context.
    sector_mapping: dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.
    """

    if hide_positions:
        show_and_plot_top_pos = 0
    vertical_sections = 5 if sector_mappings is not None else 4

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_gross_leverage = plt.subplot(gs[0, :])
    ax_exposures = plt.subplot(gs[1, :], sharex=ax_gross_leverage)
    ax_top_positions = plt.subplot(gs[2, :], sharex=ax_gross_leverage)
    ax_holdings = plt.subplot(gs[3, :], sharex=ax_gross_leverage)

    positions_alloc = pos.get_percent_alloc(positions)

    if gross_lev is not None:
        plotting.plot_gross_leverage(returns, gross_lev, ax=ax_gross_leverage)

    plotting.plot_exposures(returns, positions_alloc, ax=ax_exposures)

    plotting.show_and_plot_top_positions(
        returns,
        positions_alloc,
        show_and_plot=show_and_plot_top_pos,
        hide_positions=hide_positions,
        ax=ax_top_positions)

    plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

    if sector_mappings is not None:
        sector_exposures = pos.get_sector_exposures(positions, sector_mappings)

        sector_alloc = pos.get_percent_alloc(sector_exposures)
        sector_alloc = sector_alloc.drop('cash', axis='columns')
        ax_sector_alloc = plt.subplot(gs[4, :], sharex=ax_gross_leverage)
        plotting.plot_sector_allocations(returns, sector_alloc,
                                         ax=ax_sector_alloc)

    plt.show()
    if return_fig:
        return fig


@plotting_context
def create_txn_tear_sheet(returns, positions, transactions,
                          unadjusted_returns=None, return_fig=False):
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
         Daily transaction volume and dollar ammount.
         - See full explanation in create_full_tear_sheet.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    set_context : boolean, optional
        If True, set default plotting style context.
    """
    vertical_sections = 5 if unadjusted_returns is not None else 3

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_turnover = plt.subplot(gs[0, :])
    ax_daily_volume = plt.subplot(gs[1, :], sharex=ax_turnover)
    ax_turnover_hist = plt.subplot(gs[2, :])

    plotting.plot_turnover(
        returns,
        transactions,
        positions,
        ax=ax_turnover)

    plotting.plot_daily_volume(returns, transactions, ax=ax_daily_volume)

    try:
        plotting.plot_daily_turnover_hist(transactions, positions,
                                          ax=ax_turnover_hist)
    except AttributeError:
        warnings.warn('Unable to generate turnover plot.', UserWarning)

    if unadjusted_returns is not None:
        ax_slippage_sweep = plt.subplot(gs[3, :])
        plotting.plot_slippage_sweep(unadjusted_returns,
                                     transactions,
                                     positions,
                                     ax=ax_slippage_sweep
                                     )
        ax_slippage_sensitivity = plt.subplot(gs[4, :])
        plotting.plot_slippage_sensitivity(unadjusted_returns,
                                           transactions,
                                           positions,
                                           ax=ax_slippage_sensitivity
                                           )

    plt.show()
    if return_fig:
        return fig


@plotting_context
def create_interesting_times_tear_sheet(
        returns, benchmark_rets=None, legend_loc='best', return_fig=False):
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    set_context : boolean, optional
        If True, set default plotting style context.
    """
    rets_interesting = timeseries.extract_interesting_date_ranges(returns)

    if len(rets_interesting) == 0:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    print('\nStress Events')
    print(np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[
          :, ['mean', 'min', 'max']], 3))

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')
        # If the strategy's history is longer than the benchmark's, limit
        # strategy
        if returns.index[0] < benchmark_rets.index[0]:
            returns = returns[returns.index > benchmark_rets.index[0]]

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
        timeseries.cum_returns(rets_period).plot(
            ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)
        timeseries.cum_returns(bmark_interesting[name]).plot(
            ax=ax, color='gray', label='SPY', alpha=0.6)
        ax.legend(['algo',
                   'SPY'],
                  loc=legend_loc)
        ax.set_title(name, size=14)
        ax.set_ylabel('Returns')
        ax.set_xlabel('')

    plt.show()
    if return_fig:
        return fig


@plotting_context
def create_bayesian_tear_sheet(returns, benchmark_rets=None,
                               live_start_date=None, samples=2000,
                               return_fig=False, stoch_vol=False):
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
    set_context : boolean, optional
        If True, set default plotting style context.
    stoch_vol : boolean, optional
        If True, run and plot the stochastic volatility model
    """

    if live_start_date is None:
        raise NotImplementedError(
            'Bayesian tear sheet requires setting of live_start_date'
        )

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY',
                                               start=returns.index[0],
                                               end=returns.index[-1])

    live_start_date = utils.get_utc_timestamp(live_start_date)
    df_train = returns.loc[returns.index < live_start_date]
    df_test = returns.loc[returns.index >= live_start_date]

    # Run T model with missing data
    print("Running T model")
    previous_time = time()
    # track the total run time of the Bayesian tear sheet
    start_time = previous_time

    trace_t = bayesian.run_model('t', df_train, returns_test=df_test,
                                 samples=samples)
    previous_time = timer("T model", previous_time)

    # Compute BEST model
    print("\nRunning BEST model")
    trace_best = bayesian.run_model('best', df_train,
                                    returns_test=df_test,
                                    samples=samples)
    previous_time = timer("BEST model", previous_time)

    # Plot results

    fig = plt.figure(figsize=(14, 10 * 2))
    gs = gridspec.GridSpec(9, 2, wspace=0.3, hspace=0.3)

    axs = []
    row = 0

    # Plot Bayesian cone
    ax_cone = plt.subplot(gs[row, :])
    bayesian.plot_bayes_cone(df_train, df_test,
                             trace=trace_t,
                             ax=ax_cone)
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
    day_pred = trace_t['returns_missing'][:, 0]
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
        np.cumprod(trace_t['returns_missing'][:, :5] + 1, 1) - 1)[:, -1]
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
    print("\nRunning alpha beta model")
    benchmark_rets = benchmark_rets.loc[df_train.index]
    trace_alpha_beta = bayesian.run_model('alpha_beta', df_train,
                                          bmark=benchmark_rets,
                                          samples=samples)
    previous_time = timer("running alpha beta model", previous_time)

    # Plot alpha and beta
    row += 1
    ax_alpha = plt.subplot(gs[row, 0])
    ax_beta = plt.subplot(gs[row, 1])
    sns.distplot((1 + trace_alpha_beta['alpha'][100:])**252 - 1, ax=ax_alpha)
    ax_alpha.set_xlabel('Annual Alpha')
    ax_alpha.set_ylabel('Belief')
    sns.distplot(trace_alpha_beta['beta'][100:], ax=ax_beta)
    ax_beta.set_xlabel('Beta')
    ax_beta.set_ylabel('Belief')
    previous_time = timer("plotting alpha beta model", previous_time)

    if stoch_vol:
        # run stochastic volatility model
        print("\nRunning stochastic volatility model on most recent 400 days of returns")
        returns_cutoff = 400
        if df_train.size > returns_cutoff:
            df_train_truncated = df_train[-returns_cutoff:]
        trace_stoch_vol = bayesian.model_stoch_vol(df_train_truncated)
        previous_time = timer("running stochastic volatility model", previous_time)

        # plot log(sigma) and log(nu)
        print("\nPlotting stochastic volatility model")
        row += 1
        ax_sigma_log = plt.subplot(gs[row, 0])
        ax_nu_log = plt.subplot(gs[row, 1])
        sigma_log = trace_stoch_vol['sigma_log']
        sns.distplot(sigma_log, ax=ax_sigma_log)
        ax_sigma_log.set_xlabel('log(Sigma)')
        ax_sigma_log.set_ylabel('Belief')
        nu_log = trace_stoch_vol['nu_log']
        sns.distplot(nu_log, ax=ax_nu_log)
        ax_nu_log.set_xlabel('log(nu)')
        ax_nu_log.set_ylabel('Belief')

        # plot latent volatility
        row += 1
        ax_volatility = plt.subplot(gs[row, :])
        bayesian.plot_stoch_vol(df_train_truncated, trace=trace_stoch_vol, ax=ax_volatility)
        previous_time = timer("plotting stochastic volatility model", previous_time)

    total_time = time() - start_time
    print("\nTotal runtime was {:.2f} seconds.").format(total_time)

    gs.tight_layout(fig)

    plt.show()
    if return_fig:
        return fig
