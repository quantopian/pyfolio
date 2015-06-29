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
    ax1 = plt.subplot(gs[:2, :]) # Rolling returns
    ax2 = plt.subplot(gs[2, :]) # Rolling beta
    #plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(gs[3, :], sharex=ax2) # Rolling sharpe
    #plt.setp(ax3.get_xticklabels(), visible=False)
    ax4 = plt.subplot(gs[4, :], sharex=ax2) # Rolling risk factors
    ax5 = plt.subplot(gs[5, 0])
    ax6 = plt.subplot(gs[5, 1])
    ax7 = plt.subplot(gs[5, 2])
    ax8 = plt.subplot(gs[6, :])
    ax9 = plt.subplot(gs[7, :])
    ax10 = plt.subplot(gs[8, :])
    ax11 = plt.subplot(gs[9, :])

    plotting.plot_rolling_returns(
        df_cum_rets, df_rets, benchmark_rets, benchmark2_rets, algo_create_date, cone_std=cone_std, ax=ax1)

    plotting.plot_rolling_beta(df_cum_rets, df_rets, benchmark_rets, ax=ax2)

    plotting.plot_rolling_sharp(df_cum_rets, df_rets, ax=ax3)

    plotting.plot_rolling_risk_factors(
        df_cum_rets, df_rets, risk_factors, ax=ax4)

    plotting.plot_monthly_returns_heatmap(df_rets, ax=ax5)
    plotting.plot_annual_returns(df_rets, ax=ax6)
    plotting.plot_monthly_returns_dist(df_rets, ax=ax7)

    df_rets_backtest = df_rets[df_rets.index < algo_create_date]
    df_rets_live = df_rets[df_rets.index > algo_create_date]

    plotting.plot_daily_returns_similarity(df_rets_backtest, df_rets_live, ax=ax8)

    df_weekly = timeseries.aggregate_returns(df_rets, 'weekly')
    df_monthly = timeseries.aggregate_returns(df_rets, 'monthly')

    plotting.plot_return_quantiles(df_rets, df_weekly, df_monthly, ax=ax9)

    plotting.show_return_range(df_rets, df_weekly)

    # Drawdowns
    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)
    plotting.plot_drawdown_periods(df_rets, df_cum_rets, top=5, ax=ax10)

    plotting.plot_drawdown_underwater(df_cum_rets=df_cum_rets, ax=ax11)

    print '\nWorst Drawdown Periods'
    drawdown_df = timeseries.gen_drawdown_table(df_rets, top=5)
    drawdown_df['peak date'] = pd.to_datetime(drawdown_df['peak date'],unit='D')
    drawdown_df['valley date'] = pd.to_datetime(drawdown_df['valley date'],unit='D')
    drawdown_df['recovery date'] = pd.to_datetime(drawdown_df['recovery date'],unit='D')
    drawdown_df['net drawdown in %'] = map( utils.round_two_dec_places, drawdown_df['net drawdown in %'] )
    print drawdown_df.sort('net drawdown in %', ascending=False)


def create_position_tear_sheet(df_rets, df_pos_val, gross_lev=None):
    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)

    plotting.plot_gross_leverage(df_cum_rets, gross_lev)

    df_pos_alloc = positions.get_portfolio_alloc(df_pos_val)

    plotting.plot_exposures(df_cum_rets, df_pos_alloc)

    plotting.show_and_plot_top_positions(df_cum_rets, df_pos_alloc)

    plotting.plot_holdings(df_pos_alloc)


def create_txn_tear_sheet(df_rets, df_pos_val, df_txn):
    df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1)

    plotting.plot_turnover(df_cum_rets, df_txn, df_pos_val)

    plotting.plot_daily_volume(df_cum_rets, df_txn)

    plotting.plot_volume_per_day_hist(df_txn)

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
