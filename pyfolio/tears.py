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


import warnings

from . import timeseries
from . import utils
from . import pos
from . import plotting
try:
    from . import bayesian
except ImportError:
    warnings.warn("Could not import bayesian submodule due to missing pymc3 dependency.", ImportWarning)

import numpy as np
import scipy.stats
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def create_returns_tear_sheet(returns, live_start_date=None, backtest_days_pct=0.5, cone_std=1.0, benchmark_rets=None, benchmark2_rets=None, return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's returns.

    - Fetches benchmarks, then creates the plots on a single figure.
    - Plots: rolling returns (with cone), rolling beta, rolling sharpe, rolling Fama-French risk factors, drawdowns, underwater plot, monthly and annual return plots, daily similarity plots, and return quantile box plot.
    - Will also print the start and end dates of the strategy, performance statistics, drawdown periods, and the return range.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after its backtest period.
    backtest_days_pct : float, optional
        The fraction of the returns data that comes from backtesting (versus live trading).
         - Only used if live_start_date is left blank.
    cone_std : float, optional
        The standard deviation to use for the cone plots.
    benchmark_rets : pd.Series, optional
        Daily non-cumulative returns of the first benchmark.
    benchmark2_rets : pd.Series, optional
        Daily non-cumulative returns of the second benchmark.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')
        # If the strategy's history is longer than the benchmark's, limit strategy
        if returns.index[0] < benchmark_rets.index[0]:
            returns = returns[returns.index > benchmark_rets.index[0]]
    if benchmark2_rets is None:
        benchmark2_rets = utils.get_symbol_rets('IEF')  # 7df_c-10yr Bond ETF.

    risk_factors = utils.load_portfolio_risk_factors().dropna(axis=0)

    plotting.set_plot_defaults()

    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)

    print("Entire data start date: " + str(df_cum_rets.index[0]))
    print("Entire data end date: " + str(df_cum_rets.index[-1]))

    if live_start_date is None:
            live_start_date = returns.index[int(len(returns)*backtest_days_pct)]

    print('\n')

    plotting.show_perf_stats(returns, live_start_date, benchmark_rets)

    fig = plt.figure(figsize=(14, 10*6))
    gs = gridspec.GridSpec(10, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])
    ax_rolling_beta = plt.subplot(gs[2, :], sharex=ax_rolling_returns)
    ax_rolling_sharpe = plt.subplot(gs[3, :], sharex=ax_rolling_returns)
    ax_rolling_risk = plt.subplot(gs[4, :], sharex=ax_rolling_returns)
    ax_drawdown = plt.subplot(gs[5, :], sharex=ax_rolling_returns)
    ax_underwater = plt.subplot(gs[6, :], sharex=ax_rolling_returns)
    ax_monthly_heatmap = plt.subplot(gs[7, 0])
    ax_annual_returns = plt.subplot(gs[7, 1])
    ax_monthly_dist = plt.subplot(gs[7, 2])
    ax_daily_similarity_scale = plt.subplot(gs[8, 0])
    ax_daily_similarity_no_var = plt.subplot(gs[8, 1])
    ax_daily_similarity_no_var_no_mean = plt.subplot(gs[8, 2])
    ax_return_quantiles = plt.subplot(gs[9, :])


    plotting.plot_rolling_returns(
        returns,
        benchmark_rets=benchmark_rets,
        benchmark2_rets=benchmark2_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)

    plotting.plot_rolling_beta(
        returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_sharpe(
        returns, ax=ax_rolling_sharpe)

    plotting.plot_rolling_risk_factors(
        returns, risk_factors, ax=ax_rolling_risk)

    # Drawdowns
    plotting.plot_drawdown_periods(
        returns, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(
        returns=returns, ax=ax_underwater)

    plotting.show_worst_drawdown_periods(returns)


    returns_backtest = returns[returns.index < live_start_date]
    returns_live = returns[returns.index > live_start_date]

    df_weekly = timeseries.aggregate_returns(returns, 'weekly')
    df_monthly = timeseries.aggregate_returns(returns, 'monthly')

    print('\n')
    plotting.show_return_range(returns, df_weekly)

    plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    plotting.plot_daily_returns_similarity(returns_backtest,
                                           returns_live,
                                           title='Daily Returns Similarity',
                                           ax=ax_daily_similarity_scale)
    plotting.plot_daily_returns_similarity(returns_backtest,
                                           returns_live,
                                           scale_kws={'with_std': False},
                                           title='Similarity without\nvariance normalization',
                                           ax=ax_daily_similarity_no_var)
    plotting.plot_daily_returns_similarity(returns_backtest,
                                           returns_live,
                                           scale_kws={'with_std': False,
                                                      'with_mean': False},
                                           title='Similarity without variance\nand mean normalization',
                                           ax=ax_daily_similarity_no_var_no_mean)

    plotting.plot_return_quantiles(returns, df_weekly, df_monthly, ax=ax_return_quantiles)

    if return_fig:
        return fig


def create_position_tear_sheet(returns, positions_val, gross_lev=None, return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's positions and holdings.

    - Plots: gross leverage, exposures, top positions, and holdings.
    - Will also print the top positions held.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    positions_val : pd.DataFrame
        The positions that the strategy takes over time.
    gross_lev : pd.Series, optional
         The sum of long and short exposure per share divided by net asset value.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    fig = plt.figure(figsize=(14, 4*6))
    gs = gridspec.GridSpec(4, 3, wspace=0.5, hspace=0.5)
    ax_gross_leverage = plt.subplot(gs[0, :])
    ax_exposures = plt.subplot(gs[1, :], sharex=ax_gross_leverage)
    ax_top_positions = plt.subplot(gs[2, :], sharex=ax_gross_leverage)
    ax_holdings = plt.subplot(gs[3, :], sharex=ax_gross_leverage)

    positions_alloc = pos.get_portfolio_alloc(positions_val)

    if gross_lev is not None:
        plotting.plot_gross_leverage(returns, gross_lev, ax=ax_gross_leverage)

    plotting.plot_exposures(returns, positions_alloc, ax=ax_exposures)

    plotting.show_and_plot_top_positions(returns, positions_alloc, ax=ax_top_positions)

    plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

    if return_fig:
        return fig


def create_txn_tear_sheet(returns, positions_val, transactions, return_fig=False):
    """
    Generate a number of plots for analyzing a strategy's transactions.

    Plots: turnover, daily volume, and a histogram of daily volume.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    positions_val : pd.DataFrame
        The positions that the strategy takes over time.
    transactions : pd.DataFrame
         A strategy's transactions. See pos.make_transaction_frame(transactions).
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    fig = plt.figure(figsize=(14, 3*6))
    gs = gridspec.GridSpec(3, 3, wspace=0.5, hspace=0.5)
    ax_turnover = plt.subplot(gs[0, :])
    ax_daily_volume = plt.subplot(gs[1, :], sharex=ax_turnover)
    ax_daily_volume_hist = plt.subplot(gs[2, :])

    plotting.plot_turnover(returns, transactions, positions_val, ax=ax_turnover)

    plotting.plot_daily_volume(returns, transactions, ax=ax_daily_volume)

    plotting.plot_volume_per_day_hist(transactions, ax=ax_daily_volume_hist)

    if return_fig:
        return fig


def create_interesting_times_tear_sheet(returns, benchmark_rets=None, legend_loc='best', return_fig=False):
    """
    Generate a number of returns plots around interesting points in time, like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure, 9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing bubble burst, EZB IR, Great Recession (August 2007, March and September of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    benchmark_rets : pd.Series, optional
        Daily non-cumulative returns of a benchmark.
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    rets_interesting = timeseries.extract_interesting_date_ranges(returns)
    print('\nStress Events')
    print(np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[:, ['mean', 'min', 'max']], 3))

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')
        # If the strategy's history is longer than the benchmark's, limit strategy
        if returns.index[0] < benchmark_rets.index[0]:
            returns = returns[returns.index > benchmark_rets.index[0]]

    bmark_interesting = timeseries.extract_interesting_date_ranges(
        benchmark_rets)

    num_plots = len(rets_interesting)
    num_rows = int((num_plots+1)/2.0) # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    fig = plt.figure(figsize=(14, num_rows*6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.items()):

        ax = plt.subplot(gs[int(i/2.0), i%2]) # i=0 -> 0, i=1 -> 0, i=2 -> 1 ;; i=0 -> 0, i=1 -> 1, i=2 -> 0
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

    if return_fig:
        return fig


def create_bayesian_tear_sheet(returns, bmark, live_start_date=None, backtest_days_pct=0.5, return_fig=False):
    """
    Generate a number of Bayesian distributions and a Beyesian cone plot of returns.

    Plots: Sharpe distribution, annual volatility distribution, annual alpha distribution, beta distribution, predicted 1 and 5 day returns distributions, and a cumulative returns cone plot.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    bmark : pd.Series
        Daily non-cumulative returns of a benchmark.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after its backtest period.
    backtest_days_pct : float, optional
        The fraction of the returns data that comes from backtesting (versus live trading).
         - Only used if live_start_date is left blank.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    if live_start_date is None:
            live_start_date = returns.index[int(len(returns)*backtest_days_pct)]

    fig = plt.figure(figsize=(14, 10*2))
    gs = gridspec.GridSpec(4, 2, wspace=0.3, hspace=0.3)

    row = 0
    ax_sharpe = plt.subplot(gs[row, 0])
    ax_vol = plt.subplot(gs[row, 1])

    df_train = returns.loc[returns.index < live_start_date]
    df_test = returns.loc[returns.index >= live_start_date]
    trace_t = bayesian.run_model('t', df_train, df_test=df_test,
                                 samples=2000)

    sns.distplot(trace_t['sharpe'][100:], ax=ax_sharpe)
    #ax_sharpe.set_title('Bayesian T-Sharpe Ratio')
    ax_sharpe.set_xlabel('Sharpe Ratio')
    ax_sharpe.set_ylabel('Belief')
    sns.distplot(trace_t['annual volatility'][100:], ax=ax_vol)
    #ax_vol.set_title('Annual Volatility')
    ax_vol.set_xlabel('Annual Volatility')
    ax_vol.set_ylabel('Belief')

    bmark = bmark.loc[df_train.index]
    trace_alpha_beta = bayesian.run_model('alpha_beta', df_train,
                                          bmark=bmark, samples=2000)

    row += 1
    ax_alpha = plt.subplot(gs[row, 0])
    ax_beta = plt.subplot(gs[row, 1])
    sns.distplot((1 + trace_alpha_beta['alpha'][100:])**252 - 1, ax=ax_alpha)
    #ax_sharpe.set_title('Alpha')
    ax_alpha.set_xlabel('Annual Alpha')
    ax_alpha.set_ylabel('Belief')
    sns.distplot(trace_alpha_beta['beta'][100:], ax=ax_beta)
    #ax_beta.set_title('Beta')
    ax_beta.set_xlabel('Beta')
    ax_beta.set_ylabel('Belief')

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


    week_pred = (np.cumprod(trace_t['returns_missing'][:, :5] + 1, 1) - 1)[:, -1]
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

    row += 1
    ax_cone = plt.subplot(gs[row, :])

    bayesian.plot_bayes_cone(df_train, df_test,
                             trace=trace_t,
                             ax=ax_cone)

    if return_fig:
        return fig


def create_full_tear_sheet(returns, positions=None, transactions=None,
                           gross_lev=None,
                           live_start_date=None, bayesian=False,
                           backtest_days_pct=0.5, cone_std=1.0):
    """
    Generate a number of tear sheets that are useful for analyzing a strategy's performance.

    - Fetches benchmarks if needed.
    - Creates tear sheets for returns, and significant events. If possible, also creates tear sheets for position analysis, transaction analysis, and Bayesian analysis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, non-cumulative.
    positions : pd.DataFrame, optional
        The positions that the strategy takes over time.
    transactions : pd.DataFrame, optional
        A strategy's transactions. See pos.make_transaction_frame(transactions).
    gross_lev : pd.Series, optional
        The sum of long and short exposure per share divided by net asset value.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after its backtest period.
    bayesian: boolean, optional
        If True, causes the generation of a Bayesian tear sheet.
    backtest_days_pct : float, optional
        The fraction of the returns data that comes from backtesting (versus live trading).
         - Only used if live_start_date is left blank.
    cone_std : float, optional
        The standard deviation to use for the cone plots.
    """

    benchmark_rets = utils.get_symbol_rets('SPY')
    benchmark2_rets = utils.get_symbol_rets('IEF')  # 7-10yr Bond ETF.

    # If the strategy's history is longer than the benchmark's, limit strategy
    if returns.index[0] < benchmark_rets.index[0]:
        returns = returns[returns.index > benchmark_rets.index[0]]

    create_returns_tear_sheet(returns, live_start_date=live_start_date, backtest_days_pct=backtest_days_pct, cone_std=cone_std, benchmark_rets=benchmark_rets, benchmark2_rets=benchmark2_rets)

    create_interesting_times_tear_sheet(returns, benchmark_rets=benchmark_rets)

    if positions is not None:
        create_position_tear_sheet(returns, positions, gross_lev=gross_lev)

        if transactions is not None:
            create_txn_tear_sheet(returns, positions, transactions)

    if bayesian:
        create_bayesian_tear_sheet(returns, benchmark_rets, live_start_date=live_start_date, backtest_days_pct=backtest_days_pct)
