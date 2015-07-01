from __future__ import division

import timeseries
import utils
import positions
import plotting
import internals

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

    # if your directory structure isn't exactly the same as the research server you can manually specify the location
    # of the directory holding the risk factor data
    # risk_factors = load_portfolio_risk_factors(local_risk_factor_path)
    risk_factors = internals.load_portfolio_risk_factors().dropna(axis=0)

    plotting.set_plot_defaults()

    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)

    print "Entire data start date: " + str(df_cum_rets.index[0])
    print "Entire data end date: " + str(df_cum_rets.index[-1])

    if algo_create_date is None:
            algo_create_date = df_rets.index[ int(len(df_rets)*backtest_days_pct) ]

    print '\n'

    plotting.show_perf_stats(df_rets, algo_create_date, benchmark_rets)

    fig = plt.figure(figsize=(14, 10*6))
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
    ax_daily_similarity = plt.subplot(gs[8, :])
    ax_return_quantiles = plt.subplot(gs[9, :])


    plotting.plot_rolling_returns(
        df_cum_rets,
        df_rets,
        benchmark_rets,
        benchmark2_rets,
        algo_create_date,
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

    ####
    df_rets_backtest = df_rets[df_rets.index < algo_create_date]
    df_rets_live = df_rets[df_rets.index > algo_create_date]

    df_weekly = timeseries.aggregate_returns(df_rets, 'weekly')
    df_monthly = timeseries.aggregate_returns(df_rets, 'monthly')

    plotting.show_return_range(df_rets, df_weekly)

    plotting.plot_monthly_returns_heatmap(df_rets, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(df_rets, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(df_rets, ax=ax_monthly_dist)
    
    plotting.plot_daily_returns_similarity(df_rets_backtest, df_rets_live, ax=ax_daily_similarity)
    
    plotting.plot_return_quantiles(df_rets, df_weekly, df_monthly, ax=ax_return_quantiles)

    print '\nWorst Drawdown Periods'
    drawdown_df = timeseries.gen_drawdown_table(df_rets, top=5)
    drawdown_df['peak date'] = pd.to_datetime(drawdown_df['peak date'],unit='D')
    drawdown_df['valley date'] = pd.to_datetime(drawdown_df['valley date'],unit='D')
    drawdown_df['recovery date'] = pd.to_datetime(drawdown_df['recovery date'],unit='D')
    drawdown_df['net drawdown in %'] = map( utils.round_two_dec_places, drawdown_df['net drawdown in %'] )
    print drawdown_df.sort('net drawdown in %', ascending=False)


def create_position_tear_sheet(df_rets, df_pos_val, gross_lev=None):
    
    fig = plt.figure(figsize=(14, 4*6))
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
    
    fig = plt.figure(figsize=(14, 3*6))
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

    fig = plt.figure(figsize=(31, 19))
    for i, (name, rets_period) in enumerate(rets_interesting.iteritems()):
        ax = fig.add_subplot(6, 3, i + 1)
        timeseries.cum_returns(rets_period).plot(
            ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)
        timeseries.cum_returns(bmark_interesting[name]).plot(
            ax=ax, color='gray', label='SPY', alpha=0.6)
        plt.legend(['algo',
                    'SPY'],
                   loc=legend_loc)
        ax.set_title(name, size=14)
        ax.set_ylabel('', size=12)
    ax.legend()

def create_full_tear_sheet(df_rets, df_pos=None, df_txn=None,
                           gross_lev=None, fetcher_urls='',
                           algo_create_date=None,
                           backtest_days_pct=0.5, cone_std=1.0):

    benchmark_rets = utils.get_symbol_rets('SPY')
    benchmark2_rets = utils.get_symbol_rets('IEF')  # 7-10yr Bond ETF.

    create_returns_tear_sheet(df_rets, algo_create_date=algo_create_date, backtest_days_pct=backtest_days_pct, cone_std=cone_std, benchmark_rets=benchmark_rets, benchmark2_rets=benchmark2_rets)

    create_interesting_times_tear_sheet(df_rets, benchmark_rets=benchmark_rets)

    if df_pos is not None:
        create_position_tear_sheet(df_rets, df_pos, gross_lev=gross_lev)

        if df_txn is not None:
            create_txn_tear_sheet(df_rets, df_pos, df_txn)
