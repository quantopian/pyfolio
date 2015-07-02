from __future__ import division
import warnings

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn import preprocessing

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
        df_cum_rets,
        df_rets,
        risk_factors,
        rolling_beta_window=63 * 2,
        legend_loc='best',
        ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    num_months_str = '%.0f' % (rolling_beta_window / 21)

    ax.set_title(
        "Rolling Fama-French Single Factor Betas (" +
        num_months_str +
        '-month)')
    ax.set_ylabel('beta')

    rolling_risk_multifactor = timeseries.rolling_multifactor_beta(
        df_rets,
        risk_factors.loc[:, ['SMB', 'HML', 'UMD']],
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

    rolling_beta_SMB.plot(color='steelblue', alpha=0.7, ax=ax, **kwargs)
    rolling_beta_HML.plot(color='orangered', alpha=0.7, ax=ax, **kwargs)
    rolling_beta_UMD.plot(color='forestgreen', alpha=0.7, ax=ax, **kwargs)
    (rolling_risk_multifactor['const'] * 252).plot(
        color='forestgreen',
        alpha=0.5,
        lw=3,
        label=False,
        ax=ax,
        **kwargs)

    ax.axhline(0.0, color='black')
    ax.legend(['Small-Caps (SMB)',
                'High-Growth (HML)',
                'Momentum (UMD)'],
               loc=legend_loc)
    ax.set_ylim((-2.0, 2.0))

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.axhline(
        (rolling_risk_multifactor['const'] * 252).mean(),
        color='darkgreen',
        alpha=0.8,
        lw=3,
        ls='--')
    ax.axhline(0.0, color='black')

    ax.set_ylabel('Alpha')
    ax.set_ylim((-.40, .40))
    ax.set_xlabel('Date')
    ax.set_title(
        'Multi-factor Alpha (vs. Factors: Small-Cap, High-Growth, Momentum)')
    return ax

def plot_cone_chart(
        cone_df,
        warm_up_days_pct,
        lines_to_plot=['line'],
        in_sample_color='grey',
        oos_color='coral',
        plot_cone_lines=True,
        ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if plot_cone_lines:
        cone_df[lines_to_plot].plot(
            alpha=0.3,
            color='k',
            ls='-',
            lw=2,
            label='',
            ax=ax, **kwargs)

    warm_up_x_end = int(len(cone_df) * warm_up_days_pct)

    ax.fill_between(
        cone_df.index[
            :warm_up_x_end], cone_df.sd_down[
            :warm_up_x_end], cone_df.sd_up[
                :warm_up_x_end], color=in_sample_color, alpha=0.15)
    ax.fill_between(
        cone_df.index[
            warm_up_x_end:], cone_df.sd_down[
            warm_up_x_end:], cone_df.sd_up[
                warm_up_x_end:], color=oos_color, alpha=0.15)
    return ax

def plot_monthly_returns_heatmap(daily_rets_ts, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = timeseries.aggregate_returns(daily_rets_ts, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack()
    monthly_ret_table = np.round(monthly_ret_table, 3)

    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={
            "size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax, **kwargs)
    ax.set_ylabel('Month')
    ax.set_xlabel('Year')
    ax.set_title("Monthly Returns (%)")
    return ax

def plot_annual_returns(daily_rets_ts, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()
        
    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)
    
    ann_ret_df = pd.DataFrame(
        timeseries.aggregate_returns(
            daily_rets_ts,
            'yearly'))

    ax.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax, kind='barh', alpha=0.70, **kwargs)
    ax.axvline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Year')
    ax.set_xlabel('Returns')
    ax.set_title("Annual Returns")
    ax.legend(['mean'])
    return ax

def plot_monthly_returns_dist(daily_rets_ts, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)
    
    monthly_ret_table = timeseries.aggregate_returns(daily_rets_ts, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack()
    monthly_ret_table = np.round(monthly_ret_table, 3)
    ax.hist(
        100*monthly_ret_table.dropna().values.flatten(),
        color='orangered',
        alpha=0.80,
        bins=20,
        **kwargs)

    ax.axvline(
        100 *
        monthly_ret_table.dropna().values.flatten().mean(),
        color='gold',
        linestyle='--',
        lw=4,
        alpha=1.0)
    ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    ax.legend(['mean'])
    ax.set_ylabel('Number of months')
    ax.set_xlabel('Returns')
    ax.set_title("Distribution of Monthly Returns")
    return ax

"""def plot_avg_holdings(df_pos):
    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.groupby([lambda x: x.year, lambda x: x.month]).apply(
        lambda x: np.mean([len(x[x != 0]) for _, x in x.iterrows()])).unstack()
    sns.heatmap(df_holdings, annot=True, cbar=False)
    ax.set_title('Average # of holdings per month')
    ax.set_xlabel('month')
    ax.set_ylabel('year')"""


def plot_holdings(df_pos, df_rets, legend_loc='best', ax=None, **kwargs):
    
    if ax is None:
        ax = plt.gca()

    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_month = df_holdings.resample('1M', how='mean')
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax, **kwargs)
    df_holdings_by_month.plot(color='orangered', alpha=0.5, lw=2, ax=ax, **kwargs)
    ax.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3,
        alpha=1.0)
    
    ax.set_xlim((df_rets.index[0], df_rets.index[-1]))

    ax.legend(['Daily holdings',
                'Average daily holdings, by month',
                'Average daily holdings, net'],
               loc=legend_loc)
    ax.set_title('Holdings per Day')
    ax.set_ylabel('Amount of holdings per day')
    ax.set_xlabel('Date')
    return ax

def plot_drawdown_periods(df_rets, df_cum_rets=None, top=10, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        
    if df_cum_rets is None:
        df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1.0)
    df_drawdowns = timeseries.gen_drawdown_table(df_rets, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = df_rets.index[-1]
        ax.fill_between((peak, recovery),
                         lim[0],
                         lim[1],
                         alpha=.4,
                         color=colors[i])

    ax.set_title('Top %i Drawdown Periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Algo'], 'upper left')
    ax.set_xlabel('Date')
    return ax


def plot_drawdown_underwater(df_rets=None, df_cum_rets=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        
    if df_cum_rets is None:
        df_cum_rets = timeseries.cum_returns(df_rets, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ( (running_max - df_cum_rets) / running_max )
    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    ax.set_ylabel('Drawdown')
    ax.set_title('Underwater Plot')
    ax.set_xlabel('Date')
    return ax


def show_perf_stats(df_rets, algo_create_date, benchmark_rets):
    df_rets_backtest = df_rets[df_rets.index < algo_create_date]
    df_rets_live = df_rets[df_rets.index > algo_create_date]

    print 'Out-of-Sample Months: ' + str(int(len(df_rets_live) / 21))
    print 'Backtest Months: ' + str(int(len(df_rets_backtest) / 21))

    perf_stats_backtest = np.round(timeseries.perf_stats(
        df_rets_backtest, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_backtest_ab = np.round(
        timeseries.calc_alpha_beta(df_rets_backtest, benchmark_rets), 2)
    perf_stats_backtest.loc['alpha'] = perf_stats_backtest_ab[0]
    perf_stats_backtest.loc['beta'] = perf_stats_backtest_ab[1]
    perf_stats_backtest.columns = ['Backtest']

    perf_stats_live = np.round(timeseries.perf_stats(
        df_rets_live, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_live_ab = np.round(
        timeseries.calc_alpha_beta(df_rets_live, benchmark_rets), 2)
    perf_stats_live.loc['alpha'] = perf_stats_live_ab[0]
    perf_stats_live.loc['beta'] = perf_stats_live_ab[1]
    perf_stats_live.columns = ['Out_of_Sample']

    perf_stats_all = np.round(timeseries.perf_stats(
        df_rets, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_all_ab = np.round(
        timeseries.calc_alpha_beta(df_rets, benchmark_rets), 2)
    perf_stats_all.loc['alpha'] = perf_stats_all_ab[0]
    perf_stats_all.loc['beta'] = perf_stats_all_ab[1]
    perf_stats_all.columns = ['All_History']

    perf_stats_both = perf_stats_backtest.join(perf_stats_live, how='inner')
    perf_stats_both = perf_stats_both.join(perf_stats_all, how='inner')

    print perf_stats_both

    diff_pct = timeseries.out_of_sample_vs_in_sample_returns_kde(timeseries.cum_returns(df_rets_backtest , 1.0),
                                                             timeseries.cum_returns(df_rets_live, 1.0) )

    consistency_pct = int( 100*(1.0 - diff_pct) )
    print "\n" + str(consistency_pct) + "%" + " :Similarity between Backtest vs. Out-of-Sample (daily returns distribution)\n"



def plot_rolling_returns(
                    df_rets,
                    benchmark_rets=None,
                    benchmark2_rets=None,
                    live_start_date=None,
                    timeseries_input_only=True,
                    cone_std=None,
                    legend_loc='best',
                    ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    df_cum_rets = timeseries.cum_returns(df_rets, 1.0)

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if benchmark_rets is not None:
        timeseries.cum_returns(benchmark_rets[df_cum_rets.index], 1.0).plot(
            ax=ax, lw=2, color='gray', label='', alpha=0.60, **kwargs)
    if benchmark2_rets is not None:
        timeseries.cum_returns(benchmark2_rets[df_cum_rets.index], 1.0).plot(
            ax=ax, lw=2, color='gray', label='', alpha=0.35, **kwargs)

    if (live_start_date is None) or (df_cum_rets.index[-1] <= live_start_date):
        df_cum_rets.plot(lw=3, color='forestgreen', label='', alpha=0.6, ax=ax, **kwargs)
        ax.legend(['Algo backtest',
                   'S&P500',
                    '7-10yr Bond'],
                   loc=legend_loc)
    else:
        df_cum_rets[:live_start_date].plot(
            lw=3, color='forestgreen', label='', alpha=0.6,
            ax=ax, **kwargs)
        df_cum_rets[live_start_date:].plot(
            lw=4, color='red', label='', alpha=0.6,
            ax=ax, **kwargs)

        ax.legend(['Algo backtest',
                   'Algo LIVE',
                   'S&P500',
                   '7-10yr Bond'],
                  loc=legend_loc)

        if cone_std is not None:
            cone_df = timeseries.cone_rolling(df_rets, num_stdev=cone_std, cone_fit_end_date=live_start_date)

            cone_df_fit = cone_df[ cone_df.index < live_start_date]

            cone_df_live = cone_df[ cone_df.index > live_start_date]
            cone_df_live = cone_df_live[ cone_df_live.index < df_rets.index[-1] ]

            cone_df_future = cone_df[ cone_df.index > df_rets.index[-1] ]

            #cone_df['line'].plot(ax=ax, ls='--', lw=2, color='forestgreen', alpha=0.7)
            cone_df_fit['line'].plot(ax=ax, ls='--', lw=2, color='forestgreen', alpha=0.7, **kwargs)
            cone_df_live['line'].plot(ax=ax, ls='--', lw=2, color='red', alpha=0.7, **kwargs)
            cone_df_future['line'].plot(ax=ax, ls='--', lw=2, color='navy', alpha=0.7, **kwargs)

            ax.fill_between(cone_df_live.index,
                            cone_df_live.sd_down,
                            cone_df_live.sd_up,
                            color='red', alpha=0.30)

            ax.fill_between(cone_df_future.index,
                            cone_df_future.sd_down,
                            cone_df_future.sd_up,
                            color='navy', alpha=0.25)

        ax.axhline(1.0, linestyle='--', color='black', lw=2)
        ax.set_ylabel('Cumulative returns')
        ax.set_title('Cumulative Returns')
        ax.set_xlabel('Date')

    return ax


def plot_rolling_beta(df_cum_rets, df_rets, benchmark_rets, rolling_beta_window=63, legend_loc='best', ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling Portfolio Beta to S&P 500")
    ax.set_ylabel('Beta')
    rb_1 = timeseries.rolling_beta(
        df_rets, benchmark_rets, rolling_window=rolling_beta_window * 2)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = timeseries.rolling_beta(
        df_rets, benchmark_rets, rolling_window=rolling_beta_window * 3)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.set_ylim((-2.5, 2.5))
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    # ax.fill_between(cone_df_future.index,
    #                rb_1.mean() + future_cone_stdev*np.std(rb_1),
    #                rb_1.mean() - future_cone_stdev*np.std(rb_1),
    #                color='steelblue', alpha=0.2)
    ax.set_xlabel('Date')
    ax.legend(['6-mo',
                '12-mo'],
               loc=legend_loc)
    return ax


def plot_rolling_sharp(df_cum_rets, df_rets, rolling_sharpe_window=63 * 2, legend_loc='best', ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(
        df_rets, rolling_sharpe_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax, **kwargs)

    ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.axhline(rolling_sharpe_ts.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    # ax.fill_between(cone_df_future.index,
    #                rolling_sharpe_ts.mean() + future_cone_stdev*np.std(rolling_sharpe_ts),
    #                rolling_sharpe_ts.mean() - future_cone_stdev*np.std(rolling_sharpe_ts),
    #                color='orangered', alpha=0.15)

    ax.set_ylim((-3.0, 6.0))
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Date')
    ax.legend(['Sharpe', 'Average'],
               loc=legend_loc)
    return ax


def plot_gross_leverage(df_cum_rets, gross_lev, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    gross_lev.plot(alpha=0.8, lw=0.5, color='g', legend=False, ax=ax, **kwargs)
    #ax.axhline(0.0, color='black', lw=2)
    ax.axhline(
        np.mean(gross_lev.iloc[:, 0]), color='g', linestyle='--', lw=3, alpha=1.0)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_title('Gross Leverage')
    ax.set_ylabel('Gross Leverage')
    ax.set_xlabel('Date')
    return ax


def plot_exposures(df_cum_rets, df_pos_alloc, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    df_long_short = positions.get_long_short_pos(df_pos_alloc)
    # Area plots can not work with negative values.
    # TODO Investigate what we want to do in case of negative cash.
    if np.any(df_long_short.cash < 0):
        warnings.warn('Negative cash, taking absolute for area plot.')
        df_long_short = df_long_short.abs()
    df_long_short.plot(
        kind='area', color=['lightblue', 'green', 'coral'], alpha=1.0,
        ax=ax, **kwargs)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_title("Long/Short/Cash Exposure")
    ax.set_ylabel('Exposure')
    ax.set_xlabel('Date')
    return ax


def show_and_plot_top_positions(df_cum_rets, df_pos_alloc, show_and_plot=2, legend_loc='real_best', ax=None, **kwargs):
    # show_and_plot allows for both showing info and plot, or doing only one.
    # plot:0, show:1, both:2 (default 2).
    df_top_long, df_top_short, df_top_abs = positions.get_top_long_short_abs(
        df_pos_alloc)

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

        _, _, df_top_abs_all = positions.get_top_long_short_abs(
            df_pos_alloc, top=1000)
        print 'All positions ever held'
        print pd.DataFrame(df_top_abs_all).index.values
        print np.round(pd.DataFrame(df_top_abs_all)[0].values, 3)
        print"\n"

    if show_and_plot == 1 or show_and_plot == 2:

        if ax is None:
            ax = plt.gca()

        df_pos_alloc[df_top_abs.index].plot(
            title='Portfolio Allocation Over Time, Only Top 10 Holdings', alpha=0.4,
            ax=ax, **kwargs)
        
        # Place legend below plot, shrink plot by 20%
        if legend_loc == 'real_best':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, -0.14), ncol=5)
        else:
            ax.legend(loc=legend_loc)
        
        ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
        ax.set_ylabel('Exposure by stock')
        return ax


def plot_return_quantiles(df_rets, df_weekly, df_monthly, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    sns.boxplot([df_rets, df_weekly, df_monthly],
                names=['daily', 'weekly', 'monthly'],
                ax=ax, **kwargs)
    ax.set_title('Return quantiles')
    return ax


def show_return_range(df_rets, df_weekly):
    var_daily = timeseries.var_cov_var_normal(
        1e7, .05, df_rets.mean(), df_rets.std())
    var_weekly = timeseries.var_cov_var_normal(
        1e7, .05, df_weekly.mean(), df_weekly.std())
    two_sigma_daily = df_rets.mean() - 2 * df_rets.std()
    two_sigma_weekly = df_weekly.mean() - 2 * df_weekly.std()

    var_sigma = pd.Series([two_sigma_daily, two_sigma_weekly],
                          index=['2-sigma returns daily', '2-sigma returns weekly'])

    print np.round(var_sigma, 3)


def plot_turnover(df_cum_rets, df_txn, df_pos_val, legend_loc='best', ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        
    df_turnover = df_txn.txn_volume / df_pos_val.abs().sum(axis='columns')
    df_turnover_by_month = df_turnover.resample('1M', how='mean')
    df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5, ax=ax, **kwargs)
    df_turnover_by_month.plot(color='orangered', alpha=0.5, lw=2, ax=ax, **kwargs)
    ax.axhline(
        df_turnover.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    ax.legend(['Daily turnover',
                'Average daily turnover, by month',
                'Average daily turnover, net'],
               loc=legend_loc)
    ax.set_title('Daily Turnover')
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_ylim((0, 1))
    ax.set_ylabel('Turnover')
    ax.set_xlabel('Date')
    return ax


def plot_daily_volume(df_cum_rets, df_txn, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    df_txn.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    ax.axhline(df_txn.txn_shares.mean(), color='steelblue',
                linestyle='--', lw=3, alpha=1.0)
    ax.set_title('Daily Trading Volume')
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_ylabel('Amount of shares traded')
    ax.set_xlabel('Date')
    return ax


def plot_volume_per_day_hist(df_txn, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    sns.distplot(df_txn.txn_volume, ax=ax, **kwargs)
    ax.set_title('Distribution of Daily Trading Volume')
    ax.set_xlabel('Volume')
    return ax

def plot_daily_returns_similarity(df_rets_backtest, df_rets_live, title="Daily Returns Similarity", scale_kws=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()
    if scale_kws is None:
        scale_kws = {}

    sns.kdeplot(preprocessing.scale(df_rets_backtest, **scale_kws),
                bw='scott', shade=True, label='backtest',
                color='forestgreen', ax=ax, **kwargs)
    sns.kdeplot(preprocessing.scale(df_rets_live, **scale_kws),
                bw='scott', shade=True, label='out-of-sample',
                color='red', ax=ax, **kwargs)
    ax.set_title(title)

    return ax

def show_worst_drawdown_periods(df_rets, top=5):
    print '\nWorst Drawdown Periods'
    drawdown_df = timeseries.gen_drawdown_table(df_rets, top=top)
    drawdown_df['peak date'] = pd.to_datetime(drawdown_df['peak date'],unit='D')
    drawdown_df['valley date'] = pd.to_datetime(drawdown_df['valley date'],unit='D')
    drawdown_df['recovery date'] = pd.to_datetime(drawdown_df['recovery date'],unit='D')
    drawdown_df['net drawdown in %'] = map( utils.round_two_dec_places, drawdown_df['net drawdown in %'] )
    print drawdown_df.sort('net drawdown in %', ascending=False)
    
    