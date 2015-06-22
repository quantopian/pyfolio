from __future__ import division

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import utils
import timeseries
import positions

def set_plot_defaults():
    # the below just sets some nice default plotting/charting colors/styles
    matplotlib.style.use('fivethirtyeight')
    sns.set_context("talk", font_scale=1.0)
    sns.set_palette("Set1", 10, 1.0)
    matplotlib.style.use('bmh')
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['axes.facecolor'] = '0.995'
    matplotlib.rcParams['figure.facecolor'] = '0.97'


def plot_rolling_risk_factors(
        algo_ts,
        df_rets,
        risk_factors,
        rolling_beta_window=63*2,
        legend_loc='best'):
    from matplotlib.ticker import FuncFormatter
    y_axis_formatter = FuncFormatter(utils.one_dec_places)

    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    num_months_str = '%.0f' % (rolling_beta_window / 21)

    plt.title(
        "Rolling Fama-French Single Factor Betas (" +
        num_months_str +
        '-month)',
        fontsize=16)
    plt.ylabel('beta', fontsize=14)

    rolling_risk_multifactor = timeseries.rolling_multifactor_beta(
        df_rets,
        risk_factors.ix[
            :,
            [
                'SMB',
                'HML',
                'UMD']],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB = timeseries.rolling_beta(
        df_rets,
        risk_factors['SMB'],
        rolling_window=rolling_beta_window)
    rolling_beta_HML = timeseries.rolling_beta(
        df_rets,
        risk_factors['HML'],
        rolling_window=rolling_beta_window)
    rolling_beta_UMD = timeseries.rolling_beta(
        df_rets,
        risk_factors['UMD'],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB.plot(color='steelblue', alpha=0.7, ax=ax)
    rolling_beta_HML.plot(color='orangered', alpha=0.7, ax=ax)
    rolling_beta_UMD.plot(color='forestgreen', alpha=0.7, ax=ax)

    plt.axhline(0.0, color='black')
    plt.legend(['Small-Caps (SMB)',
                'High-Growth (HML)',
                'Momentum (UMD)'],
               loc=legend_loc)
    plt.ylim((-2.0, 2.0))

    # plt.figure(figsize=(13,3))
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    (rolling_risk_multifactor[
     'const'] * 252).plot(color='forestgreen', alpha=0.5, lw=3, label=False, ax=ax)
    plt.axhline(
        (rolling_risk_multifactor['const'] * 252).mean(),
        color='darkgreen',
        alpha=0.8,
        lw=3,
        ls='--')
    plt.axhline(0.0, color='black')

    plt.ylabel('alpha', fontsize=14)
    plt.xlim( (df_rets.index[0], df_rets.index[-1]) )
    plt.ylim((-.40, .40))
    plt.title(
        'Multi-factor Alpha (vs. Factors: Small-Cap, High-Growth, Momentum)',
        fontsize=16)


def plot_cone_chart(
        cone_df,
        warm_up_days_pct,
        lines_to_plot=['line'],
        in_sample_color='grey',
        oos_color='coral',
        plot_cone_lines=True):
    if plot_cone_lines:
        cone_df[lines_to_plot].plot(
            alpha=0.3,
            color='k',
            ls='-',
            lw=2,
            label='')

    warm_up_x_end = int(len(cone_df) * warm_up_days_pct)

    plt.fill_between(
        cone_df.index[
            :warm_up_x_end], cone_df.sd_down[
            :warm_up_x_end], cone_df.sd_up[
                :warm_up_x_end], color=in_sample_color, alpha=0.15)
    plt.fill_between(
        cone_df.index[
            warm_up_x_end:], cone_df.sd_down[
            warm_up_x_end:], cone_df.sd_up[
                warm_up_x_end:], color=oos_color, alpha=0.15)


def plot_calendar_returns_info_graphic(daily_rets_ts, x_dim=15, y_dim=6):
    #cumulate_returns = lambda x: cum_returns(x)[-1]

    #rets_df = pd.DataFrame(index=daily_rets_ts.index, data=daily_rets_ts.values)

    #rets_df['dt'] = map( pd.to_datetime, rets_df.index )
    #rets_df['month'] = map( lambda x: x.month, rets_df.dt )
    #rets_df['year'] = map( lambda x: x.year, rets_df.dt )

    # monthly_ret = rets_df.groupby(['year','month'])[0].sum()
    #monthly_ret = rets_df.groupby(['year','month'])[0].apply(cumulate_returns)
    #monthly_ret_table = monthly_ret.unstack()
    #monthly_ret_table['Annual Return'] = monthly_ret_table.sum(axis=1)
    #monthly_ret_table['Annual Return'] = monthly_ret_table.apply(cumulate_returns, axis=1)
    #ann_ret_df = pd.DataFrame(index=monthly_ret_table['Annual Return'].index, data=monthly_ret_table['Annual Return'].values)

    ann_ret_df = pd.DataFrame(
        timeseries.aggregate_returns(
            daily_rets_ts,
            'yearly'))

    monthly_ret_table = timeseries.aggregate_returns(daily_rets_ts, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack()
    monthly_ret_table = np.round(monthly_ret_table, 3)

    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={
            "size": 12},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn)
    ax1.set_ylabel(' ')
    ax1.set_xlabel("Monthly Returns (%)")

    ax2 = fig.add_subplot(1, 3, 2)
    # sns.barplot(ann_ret_df.index, ann_ret_df[0], ci=None)
    ax2.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax2, kind='barh', alpha=0.70)
    ax2.axvline(0.0, color='black', linestyle='-', lw=3)
    # sns.heatmap(ann_ret_df*100.0, annot=True, annot_kws={"size": 14, "weight":'bold'}, center=0.0, cbar=False, cmap=matplotlib.cm.RdYlGn)
    ax2.set_ylabel(' ')
    ax2.set_xlabel("Annual Returns (%)")
    ax2.legend(['mean'])

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(
        100 *
        monthly_ret_table.dropna().values.flatten(),
        color='orangered',
        alpha=0.80,
        bins=20)
    #sns.distplot(100*monthly_ret_table.values.flatten(), bins=20, ax=ax3, color='orangered', kde=True)
    ax3.axvline(
        100 *
        monthly_ret_table.dropna().values.flatten().mean(),
        color='gold',
        linestyle='--',
        lw=4,
        alpha=1.0)
    ax3.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    ax3.legend(['mean'])
    ax3.set_xlabel("Distribution of Monthly Returns (%)")


"""def plot_avg_holdings(df_pos):
    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.groupby([lambda x: x.year, lambda x: x.month]).apply(
        lambda x: np.mean([len(x[x != 0]) for _, x in x.iterrows()])).unstack()
    sns.heatmap(df_holdings, annot=True, cbar=False)
    plt.title('Average # of holdings per month')
    plt.xlabel('month')
    plt.ylabel('year')"""


def plot_holdings(df_pos, end_date=None, legend_loc='best'):
    plt.figure(figsize=(13, 6))
    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_month = df_holdings.resample('1M', how='mean')
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5)
    df_holdings_by_month.plot(color='orangered', alpha=0.5, lw=2)
    plt.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3,
        alpha=1.0)
    if end_date is not None:
        plt.xlim((df_holdings.index[0], end_date))
        
    plt.legend(['Daily holdings',
                'Average daily holdings, by month',
                'Average daily holdings, net'],
               loc=legend_loc)
    plt.title('# of Holdings Per Day')
    

def plot_drawdowns(df_rets, algo_ts=None, top=10):
    df_drawdowns = timeseries.gen_drawdown_table(df_rets)
    
    # algo_ts - 1 = cum_returns when startingvalue=None
    
    if algo_ts is None:
        algo_ts = timeseries.cum_returns(df_rets, starting_value=1)
    
    running_max = np.maximum.accumulate(algo_ts - 1)
    underwater = running_max - (algo_ts - 1)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13, 6))
    (100 * (algo_ts - 1)).plot(ax=ax1)
    (-100 * underwater).plot(ax=ax2, kind='area', color='darkred', alpha=0.4)
    lim = ax1.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = df_rets.iloc[-1].index
        ax1.fill_between((peak, recovery),
                         lim[0],
                         lim[1],
                         alpha=.5,
                         color=colors[i])

    plt.suptitle('Top %i draw down periods' % top)
    ax1.set_ylabel('returns in %')
    ax2.set_ylabel('drawdown in %')
    ax2.set_title('Underwater plot')


def show_perf_stats(df_rets, algo_create_date, benchmark_rets):
    df_rets_backtest = df_rets[ df_rets.index < algo_create_date]
    df_rets_live = df_rets[ df_rets.index > algo_create_date]

    print 'Out-of-Sample Months: ' + str( int( len(df_rets_live) / 21) )
    print 'Backtest Months: ' + str( int( len(df_rets_backtest) / 21) )

    perf_stats_backtest = np.round(timeseries.perf_stats(df_rets_backtest, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_backtest_ab = np.round(timeseries.calc_alpha_beta(df_rets_backtest, benchmark_rets), 2)
    perf_stats_backtest.loc['alpha'] = perf_stats_backtest_ab[0]
    perf_stats_backtest.loc['beta'] = perf_stats_backtest_ab[1]
    perf_stats_backtest.columns = ['Backtest']

    perf_stats_live = np.round(timeseries.perf_stats(df_rets_live, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_live_ab = np.round(timeseries.calc_alpha_beta(df_rets_live, benchmark_rets), 2)
    perf_stats_live.loc['alpha'] = perf_stats_live_ab[0]
    perf_stats_live.loc['beta'] = perf_stats_live_ab[1]
    perf_stats_live.columns = ['Out_of_Sample']

    perf_stats_all = np.round(timeseries.perf_stats(df_rets, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_all_ab = np.round(timeseries.calc_alpha_beta(df_rets, benchmark_rets), 2)
    perf_stats_all.loc['alpha'] = perf_stats_all_ab[0]
    perf_stats_all.loc['beta'] = perf_stats_all_ab[1]
    perf_stats_all.columns = ['All_History']

    perf_stats_both = perf_stats_backtest.join(perf_stats_live, how='inner')
    perf_stats_both = perf_stats_both.join(perf_stats_all, how='inner')

    print perf_stats_both

def plot_rolling_returns(algo_ts, df_rets, benchmark_rets, benchmark2_rets, algo_create_date, timeseries_input_only=True, legend_loc='best'):
    #future_cone_stdev = 1.5

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    timeseries.cum_returns(benchmark_rets[algo_ts.index], 1.0).plot(ax=ax, lw=2, color='gray', label='', alpha=0.60)
    timeseries.cum_returns(benchmark2_rets[algo_ts.index], 1.0).plot(ax=ax, lw=2, color='gray', label='', alpha=0.35)

    if not timeseries_input_only and algo_ts.index[-1] <= algo_create_date:
        algo_ts.plot(lw=3, color='forestgreen', label='', alpha=0.6)
        plt.legend(['S&P500',
                    '7-10yr Bond',
                    'Algo backtest'],
                   loc=legend_loc)
    else:
        algo_ts[:algo_create_date].plot(lw=3, color='forestgreen', label='', alpha=0.6)
        algo_ts[algo_create_date:].plot(lw=4, color='red', label='', alpha=0.6)

        #cone_df = timeseries.cone_rolling(df_rets, num_stdev=future_cone_stdev, cone_fit_end_date=algo_create_date)

        #cone_df_fit = cone_df[ cone_df.index < algo_create_date]
        #cone_df_live = cone_df[ cone_df.index > algo_create_date]
        #cone_df_live = cone_df_live[ cone_df_live.index < df_rets.index[-1] ]
        #cone_df_future = cone_df[ cone_df.index > df_rets.index[-1] ]

        #cone_df_fit['line'].plot(ls='--', lw=2, color='forestgreen', alpha=0.7)
        #cone_df_live['line'].plot(ls='--', lw=2, color='coral', alpha=0.7)
        #cone_df_future['line'].plot(ls='--', lw=2, color='navy', alpha=0.7)

        #ax.fill_between(cone_df_live.index,
        #                cone_df_live.sd_down,
        #                cone_df_live.sd_up,
        #                color='coral', alpha=0.20)

        #ax.fill_between(cone_df_future.index,
        #                cone_df_future.sd_down,
        #                cone_df_future.sd_up,
        #                color='navy', alpha=0.15)

        plt.axhline(1.0 , linestyle='--', color='black', lw=2)
        plt.ylabel('Cumulative returns', fontsize=14)
        plt.xlim((algo_ts.index[0], algo_ts.index[-1]))

        if timeseries_input_only:
            plt.legend(['S&P500',
                        '7-10yr Bond',
                        'Portfolio'],
                       loc=legend_loc)
        else:
            plt.legend(['S&P500',
                        '7-10yr Bond',
                        'Algo backtest',
                        'Algo LIVE'],
                       loc=legend_loc)

def plot_rolling_beta(algo_ts, df_rets, benchmark_rets, rolling_beta_window=63, legend_loc='best'):
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    fig = plt.figure(figsize=(13,3))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    plt.title("Rolling Portfolio Beta to SP500",fontsize=16)
    plt.ylabel('beta', fontsize=14)
    rb_1 = timeseries.rolling_beta(df_rets, benchmark_rets, rolling_window=rolling_beta_window*2)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax)
    rb_2 = timeseries.rolling_beta(df_rets, benchmark_rets, rolling_window=rolling_beta_window*3)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax)
    plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )
    plt.ylim((-2.5, 2.5))
    plt.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    plt.axhline(0.0, color='black', linestyle='-', lw=2)

    #plt.fill_between(cone_df_future.index,
    #                rb_1.mean() + future_cone_stdev*np.std(rb_1),
    #                rb_1.mean() - future_cone_stdev*np.std(rb_1),
    #                color='steelblue', alpha=0.2)

    plt.legend(['6-mo',
                '12-mo'],
               loc=legend_loc)

def plot_rolling_sharp(algo_ts, df_rets, rolling_sharpe_window=63*2):
    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    fig = plt.figure(figsize=(13, 3))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(df_rets, rolling_sharpe_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered')
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    plt.title('Rolling Sharpe ratio (6-month)', fontsize=16)
    plt.axhline(rolling_sharpe_ts.mean(), color='orangered', linestyle='--', lw=3)
    plt.axhline(0.0, color='black', linestyle='-', lw=3)

    #plt.fill_between(cone_df_future.index,
    #                rolling_sharpe_ts.mean() + future_cone_stdev*np.std(rolling_sharpe_ts),
    #                rolling_sharpe_ts.mean() - future_cone_stdev*np.std(rolling_sharpe_ts),
    #                color='orangered', alpha=0.15)

    plt.xlim((algo_ts.index[0], algo_ts.index[-1]))
    plt.ylim((-3.0, 6.0))
    plt.ylabel('Sharpe ratio', fontsize=14)

def plot_gross_leverage(algo_ts, gross_lev):
    fig = plt.figure(figsize=(13, 3))
    gross_lev.plot(alpha=0.8, lw=0.5, color='g', legend=False)
    #plt.axhline(0.0, color='black', lw=2)
    plt.axhline(np.mean(gross_lev.iloc[:,0]), color='g', linestyle='--', lw=3, alpha=1.0)
    plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )
    plt.title('Gross Leverage')
    plt.ylabel('Gross Leverage', fontsize=14)

def plot_exposures(algo_ts, df_pos_alloc):
    fig = plt.figure(figsize=(13, 3))
    df_long_short = positions.get_long_short_pos(df_pos_alloc)
    df_long_short.plot(kind='area', color=['lightblue','green','coral'], alpha=1.0)
    plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )
    plt.title("Long/Short/Cash Exposure")
    plt.ylabel('Exposure', fontsize=14)

def show_and_plot_top_positions(algo_ts, df_pos_alloc, show_and_plot=2):
    # show_and_plot allows for both showing info and plot, or doing only one. plot:0, show:1, both:2 (default 2).
    df_top_long, df_top_short, df_top_abs = positions.get_top_long_short_abs(df_pos_alloc)

    if show_and_plot == 0 or show_and_plot == 2:
        print"\n"
        print 'Top 10 long positions of all time (and max%)'
        print pd.DataFrame(df_top_long).index.values
        print np.round(pd.DataFrame(df_top_long)[0].values, 3)
        print"\n"

        print 'Top 10 short positions of all time (and max%)'
        print pd.DataFrame(df_top_short).index.values
        print np.round(pd.DataFrame(df_top_short)[0].values, 3)
        print"\n"

        print 'Top 10 positions of all time (and max%)'
        print pd.DataFrame(df_top_abs).index.values
        print np.round(pd.DataFrame(df_top_abs)[0].values, 3)
        print"\n"

        _, _, df_top_abs_all = positions.get_top_long_short_abs(df_pos_alloc, top=1000)
        print 'All positions ever held'
        print pd.DataFrame(df_top_abs_all).index.values
        print np.round(pd.DataFrame(df_top_abs_all)[0].values, 3)
        print"\n"

    if show_and_plot == 1 or show_and_plot == 2:
        fig = plt.figure(figsize=(13, 3))
        df_pos_alloc[df_top_abs.index].plot(title='Portfolio allocation over time, only top 10 holdings', alpha=0.4)#kind='area')
        plt.ylabel('Exposure by Stock', fontsize=14)
        # plt.figure(figsize=(13, 6))
        plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )

def plot_return_quantiles(df_rets, df_weekly, df_monthly):
    fig = plt.figure(figsize=(13, 6))
    sns.boxplot([df_rets, df_weekly, df_monthly], names=['daily', 'weekly', 'monthly'])
    plt.title('Return quantiles')

def show_return_range(df_rets, df_weekly):
    var_daily = timeseries.var_cov_var_normal(1e7, .05, df_rets.mean(), df_rets.std())
    var_weekly = timeseries.var_cov_var_normal(1e7, .05, df_weekly.mean(), df_weekly.std())
    two_sigma_daily = df_rets.mean() - 2*df_rets.std()
    two_sigma_weekly = df_weekly.mean() - 2*df_weekly.std()

    var_sigma = pd.Series([two_sigma_daily, two_sigma_weekly],
                          index=['2-sigma returns daily', '2-sigma returns weekly'])

    print np.round(var_sigma, 3)

def plot_interesting_times(df_rets, benchmark_rets, legend_loc='best'):
    rets_interesting = timeseries.extract_interesting_date_ranges(df_rets)
    print '\nStress Events'
    print np.round(pd.DataFrame(rets_interesting).describe().transpose().loc[:,['mean','min','max']], 3)

    bmark_interesting = timeseries.extract_interesting_date_ranges(benchmark_rets)

    fig = plt.figure(figsize=(31,19))
    for i, (name, rets_period) in enumerate(rets_interesting.iteritems()):
        ax = fig.add_subplot(6, 3, i+1)
        timeseries.cum_returns(rets_period).plot(ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)
        timeseries.cum_returns(bmark_interesting[name]).plot(ax=ax, color='gray', label='SPY', alpha=0.6)
        plt.legend(['algo',
                    'SPY'],
                   loc=legend_loc)
        ax.set_title(name, size=14)
        ax.set_ylabel('', size=12)
    ax.legend()

def plot_turnover(algo_ts, df_txn, df_pos_val, legend_loc='best'):
    fig = plt.figure(figsize=(13, 4))
    df_turnover = df_txn.txn_volume / df_pos_val.abs().sum(axis='columns')
    df_turnover_by_month = df_turnover.resample('1M', how='mean')
    df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5)
    df_turnover_by_month.plot(color='orangered', alpha=0.5, lw=2)
    plt.axhline(df_turnover.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    plt.legend(['Daily turnover',
                'Average daily turnover, by month',
                'Average daily turnover, net'],
               loc=legend_loc)
    plt.title('Daily turnover')
    plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )
    plt.ylim((0, 1))
    plt.ylabel('% turn-over')

def plot_daily_volume(algo_ts, df_txn):
    fig = plt.figure(figsize=(13, 4))
    df_txn.txn_shares.plot(alpha=1.0, lw=0.5)
    plt.axhline(df_txn.txn_shares.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    plt.title('Daily volume traded')
    plt.xlim( (algo_ts.index[0], algo_ts.index[-1]) )
    plt.ylabel('# shares traded')

def plot_volume_per_day_hist(df_txn):
    fig = plt.figure(figsize=(13, 4))
    sns.distplot(df_txn.txn_volume)
    plt.title('Histogram of daily trading volume')
