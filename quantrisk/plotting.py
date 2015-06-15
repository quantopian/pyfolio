from __future__ import division

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import utils
import timeseries


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
        my_algo_returns,
        risk_factors,
        rolling_beta_window=63,
        legend_loc='lower left',
        end_date=None):
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
        my_algo_returns,
        risk_factors.ix[
            :,
            [
                'SMB',
                'HML',
                'UMD']],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB = timeseries.rolling_beta(
        my_algo_returns,
        risk_factors['SMB'],
        rolling_window=rolling_beta_window)
    rolling_beta_HML = timeseries.rolling_beta(
        my_algo_returns,
        risk_factors['HML'],
        rolling_window=rolling_beta_window)
    rolling_beta_UMD = timeseries.rolling_beta(
        my_algo_returns,
        risk_factors['UMD'],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB.plot(color='steelblue', alpha=0.7, ax=ax)
    rolling_beta_HML.plot(color='orangered', alpha=0.7, ax=ax)
    rolling_beta_UMD.plot(color='forestgreen', alpha=0.7, ax=ax)

    if end_date is None:
        plt.xlim((my_algo_returns.index[0], my_algo_returns.index[-1]))
    else:
        plt.xlim((my_algo_returns.index[0], end_date))
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
    if end_date is None:
        plt.xlim((my_algo_returns.index[0], my_algo_returns.index[-1]))
    else:
        plt.xlim((my_algo_returns.index[0], end_date))
    plt.ylabel('alpha', fontsize=14)
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


def plot_avg_holdings(df_pos):
    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.groupby([lambda x: x.year, lambda x: x.month]).apply(
        lambda x: np.mean([len(x[x != 0]) for _, x in x.iterrows()])).unstack()
    sns.heatmap(df_holdings, annot=True, cbar=False)
    plt.title('Average # of holdings per month')
    plt.xlabel('month')
    plt.ylabel('year')


def plot_holdings(df_pos, end_date=None):
    plt.figure(figsize=(13, 6))
    df_pos = df_pos.copy().drop('cash', axis='columns')
    df_holdings = df_pos.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5)
    plt.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3,
        alpha=1.0)
    if end_date is not None:
        plt.xlim((df_holdings.index[0], end_date))
    plt.title('# of Holdings Per Day')


def plot_drawdowns(df_rets, top=10):
    df_drawdowns = timeseries.gen_drawdown_table(df_rets)

    df_cum = timeseries.cum_returns(df_rets)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13, 6))
    (100 * df_cum).plot(ax=ax1)
    (-100 * underwater).plot(ax=ax2, kind='area', color='darkred', alpha=0.4)
    lim = ax1.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        ax1.fill_between((peak, recovery),
                         lim[0],
                         lim[1],
                         alpha=.5,
                         color=colors[i])

    plt.suptitle('Top %i draw down periods' % top)
    ax1.set_ylabel('returns in %')
    ax2.set_ylabel('drawdown in %')
    ax2.set_title('Underwater plot')
