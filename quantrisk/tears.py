from __future__ import division

from . import timeseries
from . import utils
from . import positions
from . import plotting
try:
    from . import bayesian
except ImportError:
    warnings.warn("Could not import bayesian submodule due to missing pymc3 dependency.", ImportWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def create_returns_tear_sheet(df_rets, algo_create_date=None, backtest_days_pct=0.5, cone_std=1.0, benchmark_rets=None, benchmark2_rets=None):

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')
    if benchmark2_rets is None:
        benchmark2_rets = utils.get_symbol_rets('IEF')  # 7-10yr Bond ETF.

    risk_factors = utils.load_portfolio_risk_factors().dropna(axis=0)

    plotting.set_plot_defaults()

    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)

    print "Entire data start date: " + str(df_cum_rets.index[0])
    print "Entire data end date: " + str(df_cum_rets.index[-1])

    if algo_create_date is None:
            algo_create_date = df_rets.index[ int(len(df_rets)*backtest_days_pct) ]

    print '\n'

    plotting.show_perf_stats(df_rets, algo_create_date, benchmark_rets)

    plt.figure(figsize=(14, 10*6))
    gs = gridspec.GridSpec(10, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])
    ax_rolling_beta = plt.subplot(gs[2, :], sharex=ax_rolling_returns)
    #plt.setp(ax_rolling_beta.get_xticklabels(), visible=False)
    ax_rolling_sharpe = plt.subplot(gs[3, :], sharex=ax_rolling_returns)
    #plt.setp(ax_rolling_sharpe.get_xticklabels(), visible=False)
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
        df_rets,
        benchmark_rets=benchmark_rets,
        benchmark2_rets=benchmark2_rets,
        live_start_date=algo_create_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)

    plotting.plot_rolling_beta(
        df_cum_rets, df_rets, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_sharp(
        df_cum_rets, df_rets, ax=ax_rolling_sharpe)

    plotting.plot_rolling_risk_factors(
        df_cum_rets, df_rets, risk_factors, ax=ax_rolling_risk)

    # Drawdowns
    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)
    plotting.plot_drawdown_periods(
        df_rets, df_cum_rets, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(
        df_cum_rets=df_cum_rets, ax=ax_underwater)

    plotting.show_worst_drawdown_periods(df_rets)
    ####

    df_rets_backtest = df_rets[df_rets.index < algo_create_date]
    df_rets_live = df_rets[df_rets.index > algo_create_date]

    df_weekly = timeseries.aggregate_returns(df_rets, 'weekly')
    df_monthly = timeseries.aggregate_returns(df_rets, 'monthly')

    plotting.show_return_range(df_rets, df_weekly)

    plotting.plot_monthly_returns_heatmap(df_rets, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(df_rets, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(df_rets, ax=ax_monthly_dist)

    plotting.plot_daily_returns_similarity(df_rets_backtest,
                                           df_rets_live,
                                           ax=ax_daily_similarity_scale)
    plotting.plot_daily_returns_similarity(df_rets_backtest,
                                           df_rets_live,
                                           scale_kws={'with_std': False},
                                           title='Similarity without\nvariance normalization',
                                           ax=ax_daily_similarity_no_var)
    plotting.plot_daily_returns_similarity(df_rets_backtest,
                                           df_rets_live,
                                           scale_kws={'with_std': False,
                                                      'with_mean': False},
                                           title='Similarity without variance\nand mean normalization',
                                           ax=ax_daily_similarity_no_var_no_mean)

    plotting.plot_return_quantiles(df_rets, df_weekly, df_monthly, ax=ax_return_quantiles)


def create_position_tear_sheet(df_rets, df_pos_val, gross_lev=None):

    plt.figure(figsize=(14, 4*6))
    gs = gridspec.GridSpec(4, 3, wspace=0.5, hspace=0.5)
    ax_gross_leverage = plt.subplot(gs[0, :])
    ax_exposures = plt.subplot(gs[1, :], sharex=ax_gross_leverage)
    ax_top_positions = plt.subplot(gs[2, :], sharex=ax_gross_leverage)
    ax_holdings = plt.subplot(gs[3, :], sharex=ax_gross_leverage)

    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)
    df_pos_alloc = positions.get_portfolio_alloc(df_pos_val)

    plotting.plot_gross_leverage(df_cum_rets, gross_lev, ax=ax_gross_leverage)

    plotting.plot_exposures(df_cum_rets, df_pos_alloc, ax=ax_exposures)

    plotting.show_and_plot_top_positions(df_cum_rets, df_pos_alloc, ax=ax_top_positions)

    plotting.plot_holdings(df_pos_alloc, df_rets, ax=ax_holdings)


def create_txn_tear_sheet(df_rets, df_pos_val, df_txn):

    plt.figure(figsize=(14, 3*6))
    gs = gridspec.GridSpec(3, 3, wspace=0.5, hspace=0.5)
    ax_turnover = plt.subplot(gs[0, :])
    ax_daily_volume = plt.subplot(gs[1, :], sharex=ax_turnover)
    ax_daily_volume_hist = plt.subplot(gs[2, :])

    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)

    plotting.plot_turnover(df_cum_rets, df_txn, df_pos_val, ax=ax_turnover)

    plotting.plot_daily_volume(df_cum_rets, df_txn, ax=ax_daily_volume)

    plotting.plot_volume_per_day_hist(df_txn, ax=ax_daily_volume_hist)


def create_interesting_times_tear_sheet(df_rets, benchmark_rets=None, legend_loc='best'):

    rets_interesting = timeseries.extract_interesting_date_ranges(df_rets)
    print '\nStress Events'
    print np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[:, ['mean', 'min', 'max']], 3)

    if benchmark_rets is None:
        benchmark_rets = utils.get_symbol_rets('SPY')

    bmark_interesting = timeseries.extract_interesting_date_ranges(
        benchmark_rets)

    num_plots = len(rets_interesting)
    num_rows = int((num_plots+1)/2.0) # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    plt.figure(figsize=(14, num_rows*6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.iteritems()):

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
        ax.set_xlabel('Date')

def create_bayesian_tear_sheet(df_rets, bmark, live_start_date, plot_train_len=50):
    plt.figure(figsize=(14, 10*2))
    gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5)
    ax_sharpe = plt.subplot(gs[0, 0])
    ax_vol = plt.subplot(gs[0, 1])
    ax_cone = plt.subplot(gs[1, :])

    df_train = df_rets.loc[df_rets.index < live_start_date]
    df_test = df_rets.loc[df_rets.index >= live_start_date]
    bmark = bmark.loc[df_rets.index]

    trace_t = bayesian.run_model('t', df_train, df_test=df_test,
                                 samples=2000)

    sns.distplot(trace_t['sharpe'], ax=ax_sharpe)
    ax_sharpe.set_title('Bayesian T-Sharpe Ratio')
    ax_sharpe.set_xlabel('Sharpe Ratio')
    ax_sharpe.set_ylabel('Belief')
    sns.distplot(trace_t['annual volatility'], ax=ax_vol)
    ax_vol.set_title('Annual Volatility')
    ax_vol.set_xlabel('Volatility')
    ax_vol.set_ylabel('Belief')

    trace_alpha_beta = bayesian.run_model('alpha_beta', df_train,
                                          bmark=bmark, samples=2000)

    bayesian._plot_bayes_cone(df_train, df_test,
                              trace_alpha_beta['returns_missing'],
                              ax=ax_cone)


def create_full_tear_sheet(df_rets, df_pos=None, df_txn=None,
                           gross_lev=None, fetcher_urls='',
                           algo_create_date=None, bayesian=False,
                           backtest_days_pct=0.5, cone_std=1.0):

    benchmark_rets = utils.get_symbol_rets('SPY')
    benchmark2_rets = utils.get_symbol_rets('IEF')  # 7-10yr Bond ETF.

    create_returns_tear_sheet(df_rets, algo_create_date=algo_create_date, backtest_days_pct=backtest_days_pct, cone_std=cone_std, benchmark_rets=benchmark_rets, benchmark2_rets=benchmark2_rets)

    create_interesting_times_tear_sheet(df_rets, benchmark_rets=benchmark_rets)

    if df_pos is not None:
        create_position_tear_sheet(df_rets, df_pos, gross_lev=gross_lev)

        if df_txn is not None:
            create_txn_tear_sheet(df_rets, df_pos, df_txn)

    if bayesian:
        create_bayesian_tear_sheet(df_rets, benchmark_rets, live_start_date=algo_create_date)
