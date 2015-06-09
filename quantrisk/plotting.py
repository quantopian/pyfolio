from __future__ import division

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from collections import *
from operator import *
import string

import quant_utils.timeseries as qts
from quant_utils.timeseries import cum_returns as cum_returns

import statsmodels.api as sm

import pandas.io.data as web
import statsmodels.tsa.stattools as ts

from zipline.utils import tradingcalendar

import datetime
from datetime import datetime
from datetime import timedelta
import pytz
import time

import quant_utils.timeseries as qts

utc = pytz.UTC
indexTradingCal = pd.DatetimeIndex(tradingcalendar.trading_days)
indexTradingCal = indexTradingCal.normalize()


def set_plot_defaults():
    # the below just sets some nice default plotting/charting colors/styles
    matplotlib.style.use('fivethirtyeight')
    sns.set_context("talk", font_scale=1.0)
    sns.set_palette("Set1", 10, 1.0)
    matplotlib.style.use('bmh')
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['axes.facecolor'] = '0.995'
    matplotlib.rcParams['figure.facecolor'] = '0.97'


def color_legend_texts(leg):
    # param 'leg' should be of object type, leg = ax.legend()
    # Color legend texts based on color of corresponding lines"""
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        txt.set_color(line.get_color())


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

    ann_ret_df = pd.DataFrame(qts.aggregate_returns(daily_rets_ts, 'yearly'))

    #monthly_ret = rets_df.groupby(['year','month'])[0].sum()
    #monthly_ret = rets_df.groupby(['year','month'])[0].apply(cumulate_returns)
    #monthly_ret_table = monthly_ret.unstack()

    monthly_ret_table = qts.aggregate_returns(daily_rets_ts, 'monthly')
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


def plot_scatter(
        x1Values,
        y1Values,
        moreXvaluesList=None,
        moreYvaluesList=None,
        autoLabelValues1=False,
        autoLabelMoreValues=False,
        autoLabelMoreValuesIndexes=[0],
        plotTitle=None,
        xAxisLabel=None,
        yAxisLabel=None,
        showLegend=True,
        legendLocation='upper left',
        legendLabel1='series1',
        legendLabelMoreList=None,
        showRegressionLine=False,
        seriesToUseForRegression=1,
        colorOrder=[
            'blue',
            'orange',
            'red',
            'black',
            'pink',
            'gray',
            'yellow',
            'purple',
            'darkred',
            'darkblue'],
        transparency=1.0):

    x = np.array(x1Values)
    y = np.array(y1Values)
    x = np.reshape(x, len(x))
    y = np.reshape(y, len(y))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    pointColor = 1

    ax1.scatter(
        x,
        y,
        s=200,
        color=colorOrder[0],
        alpha=transparency,
        marker="o",
        label=legendLabel1)
    if autoLabelValues1:
        for i in range(x.size):
            ax1.annotate(str(i), xy=(x[i] * 1.01, y[i] * 1.01), size=20)

    if (moreXvaluesList is not None) and (moreYvaluesList is not None):
        count = 0
        for xyTemp in zip(moreXvaluesList, moreYvaluesList):
            xTemp = np.array(xyTemp[0])
            yTemp = np.array(xyTemp[1])
            xTemp = np.reshape(xTemp, len(xTemp))
            yTemp = np.reshape(yTemp, len(yTemp))
            if legendLabelMoreList is None:
                tempLegendLabel = 'series' + str(count + 2)
            else:
                tempLegendLabel = legendLabelMoreList[count]
            ax1.scatter(
                xTemp,
                yTemp,
                color=colorOrder[pointColor],
                alpha=transparency,
                s=200,
                marker="o",
                label=tempLegendLabel)
            pointColor += 1
            if autoLabelMoreValues:
                if count in autoLabelMoreValuesIndexes:
                    for i in range(xTemp.size):
                        ax1.annotate(
                            str(i), xy=(
                                xTemp[i] * 1.01, yTemp[i] * 1.01), size=20)
            count += 1

    if showRegressionLine:
        if seriesToUseForRegression == 1:
            regX = x
            regY = y
        else:
            regX = np.array(moreXvaluesList[seriesToUseForRegression - 2])
            regY = np.array(moreYvaluesList[seriesToUseForRegression - 2])
            regX = np.reshape(regX, len(regX))
            regY = np.reshape(regY, len(regY))
        m, b = np.polyfit(regX, regY, 1)
        ax1.plot(
            regX,
            m * regX + b,
            '-',
            color=colorOrder[
                seriesToUseForRegression - 1],
            alpha=transparency)

    if plotTitle is not None:
        plt.title(plotTitle)
    if showLegend:
        plt.legend(loc=legendLocation)

    if x1Values.name is not None:
        plt.xlabel(x1Values.name)
    if y1Values.name is not None:
        plt.ylabel(y1Values.name)

    if xAxisLabel is not None:
        plt.xlabel(xAxisLabel)
    if yAxisLabel is not None:
        plt.ylabel(yAxisLabel)

    plt.show()


def plot_heatmap(df, title_str='', cmap=plt.cm.RdYlGn,
                 show_cell_value=True,
                 round_cell_value_places=1, cell_value_color='black',
                 cell_value_size=14,
                 axis_tick_font_size=16):
    """
    This creates our heatmap using our sharpe ratio dataframe
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values, cmap=cmap, interpolation='nearest')
    # axim = ax.imshow(df.values, cmap=cmap)

    # ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index, size=axis_tick_font_size)
    # ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90, size=axis_tick_font_size)

    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)

    ax.grid(False)

    '''
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(list(df.columns))

    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(list(df.index))
    '''

    ax.set_title(title_str)

    if show_cell_value:
        data = df.values
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                temp_x = round(x, round_cell_value_places)
                temp_y = round(y, round_cell_value_places)
                # format_str = '%.2f'
                format_str = '%.' + str(round_cell_value_places) + 'f'
                plt.text(x, y, format_str % data[y, x],  # data[y,x] +0.05 , data[y,x] + 0.05
                         # plt.text(x + 0.5 , y + 0.5, '%.4f' % data[y, x],
                         # #data[y,x] +0.05 , data[y,x] + 0.05
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=cell_value_size,
                         color=cell_value_color)

    plt.colorbar(axim)


def plot_performance_pivot_table_heatmaps(results_df, x_axis_var, y_axis_var):

    heatmap_x = x_axis_var
    heatmap_y = y_axis_var

    results_df_sharpe = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='sharpe_ratio')
    results_df_max_drawdown = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='max_drawdown')
    results_df_annual_return = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='annual_return')
    results_df_alpha = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='alpha')
    results_df_beta = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='beta')
    results_df_volatility = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='annual_volatility')
    results_df_stability = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='stability')
    results_df_calmar = results_df.pivot_table(
        index=heatmap_y,
        columns=heatmap_x,
        values='calmar_ratio')

    fig = plt.figure(figsize=(17, 19))

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_title("Sharpe Ratio", fontsize=16)
    ax1 = sns.heatmap(
        results_df_sharpe,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Greens)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.set_title("Annual Return", fontsize=16)
    ax2 = sns.heatmap(
        results_df_annual_return,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Greens)

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.set_title("Max Drawdown")
    ax3 = sns.heatmap(
        results_df_max_drawdown,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Reds_r)

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.set_title("Volatility", fontsize=16)
    ax4 = sns.heatmap(
        results_df_volatility,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Reds)

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_title("Beta", fontsize=16)
    ax5 = sns.heatmap(
        results_df_beta,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Blues)

    ax6 = fig.add_subplot(3, 3, 6,)
    ax6.set_title("Alpha", fontsize=16)
    ax6 = sns.heatmap(
        results_df_alpha,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Greens)

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_title("Stability", fontsize=16)
    ax7 = sns.heatmap(
        results_df_stability,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Blues)

    ax7 = fig.add_subplot(3, 3, 8)
    ax7.set_title("Calmar Ratio", fontsize=16)
    ax7 = sns.heatmap(
        results_df_calmar,
        annot=True,
        annot_kws={
            "size": 14},
        cmap=plt.cm.Greens)

    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': 14})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': 14})


def plot_rolling_risk_factors(
        my_algo_returns,
        risk_factors,
        rolling_beta_window=63,
        legend_loc='lower left'):
    plt.figure(figsize=(13, 4))
    plt.title("Rolling Fama-French Single Factor Betas", fontsize=16)
    plt.ylabel('beta', fontsize=12)

    rolling_risk_multifactor = qts.rolling_multifactor_beta(
        my_algo_returns,
        risk_factors.ix[
            :,
            [
                'SMB',
                'HML',
                'UMD']],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB = qts.rolling_beta(
        my_algo_returns,
        risk_factors['SMB'],
        rolling_window=rolling_beta_window)
    rolling_beta_HML = qts.rolling_beta(
        my_algo_returns,
        risk_factors['HML'],
        rolling_window=rolling_beta_window)
    rolling_beta_UMD = qts.rolling_beta(
        my_algo_returns,
        risk_factors['UMD'],
        rolling_window=rolling_beta_window)

    rolling_beta_SMB.plot(color='steelblue', alpha=0.7)
    rolling_beta_HML.plot(color='orangered', alpha=0.7)
    rolling_beta_UMD.plot(color='forestgreen', alpha=0.7)

    plt.xlim((my_algo_returns.index[0], my_algo_returns.index[-1]))
    plt.axhline(0.0, color='black')
    plt.legend(['Small-Caps (SMB)',
                'High-Growth (HML)',
                'Momentum (UMD)'],
               loc=legend_loc)

    plt.figure(figsize=(13, 3))

    (rolling_risk_multifactor[
     'const'] * 252).plot(color='forestgreen', alpha=0.5, lw=3, label=False)
    plt.axhline(
        (rolling_risk_multifactor['const'] * 252).mean(),
        color='darkgreen',
        alpha=0.8,
        lw=3,
        ls='--')
    plt.axhline(0.0, color='black')
    plt.xlim((my_algo_returns.index[0], my_algo_returns.index[-1]))
    plt.ylabel('alpha', fontsize=12)
    plt.title(
        'Multi-factor Alpha (vs. Factors: Small-Cap, High-Growth, Momentum)',
        fontsize=16)
