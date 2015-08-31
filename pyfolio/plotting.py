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

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn import preprocessing

from . import utils
from . import timeseries
from . import pos

from .utils import BDAYS_PER_MONTH

from functools import wraps


def plotting_context(func):
    """Decorator to set plotting context during function call."""
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def context(context='notebook', font_scale=1.5, rc=None):
    """Create pyfolio default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5,
                     'axes.facecolor': '0.995',
                     'figure.facecolor': '0.97'}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.context(font_scale=2):
    >>>    pyfolio.create_full_tear_sheet()

    See also
    --------
    For more information, see seaborn.plotting_context().

"""
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5,
                  'axes.facecolor': '0.995',
                  'figure.facecolor': '0.97'}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale,
                                rc=rc)


def plot_rolling_fama_french(
        returns,
        risk_factors=None,
        rolling_window=BDAYS_PER_MONTH * 6,
        legend_loc='best',
        ax=None, **kwargs):
    """Plots rolling Fama-French single factor betas.

    Specifically, plots SMB, HML, and UMD vs. date with a legend.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    risk_factors : pd.DataFrame, optional
        data set containing the Fama-French risk factors. See
        utils.load_portfolio_risk_factors.
    rolling_window : int, optional
        The days window over which to compute the beta.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    num_months_str = '%.0f' % (rolling_window / 21)

    ax.set_title(
        "Rolling Fama-French Single Factor Betas (" +
        num_months_str +
        '-month)')
    ax.set_ylabel('beta')

    rolling_beta = timeseries.rolling_fama_french(
        returns,
        risk_factors=risk_factors,
        rolling_window=rolling_window)

    rolling_beta.plot(alpha=0.7, ax=ax, **kwargs)

    ax.axhline(0.0, color='black')
    ax.legend(['Small-Caps (SMB)',
               'High-Growth (HML)',
               'Momentum (UMD)'],
              loc=legend_loc)
    ax.set_ylim((-2.0, 2.0))

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.axhline(0.0, color='black')
    ax.set_xlabel('')

    return ax


def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = timeseries.aggregate_returns(returns,
                                                     'monthly')
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
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly Returns (%)")
    return ax


def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    ann_ret_df = pd.DataFrame(
        timeseries.aggregate_returns(
            returns,
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


def plot_monthly_returns_dist(returns, ax=None, **kwargs):
    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major', labelsize=10)

    monthly_ret_table = timeseries.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack()
    monthly_ret_table = np.round(monthly_ret_table, 3)
    ax.hist(
        100 * monthly_ret_table.dropna().values.flatten(),
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


def plot_holdings(returns, positions, legend_loc='best', ax=None, **kwargs):
    """Plots total amount of stocks with an active position, either short
    or long.

    Displays daily total, daily average per month, and all-time daily
    average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    positions = positions.copy().drop('cash', axis='columns')
    df_holdings = positions.apply(lambda x: np.sum(x != 0), axis='columns')
    df_holdings_by_month = df_holdings.resample('1M', how='mean')
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax, **kwargs)
    df_holdings_by_month.plot(
        color='orangered',
        alpha=0.5,
        lw=2,
        ax=ax,
        **kwargs)
    ax.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3,
        alpha=1.0)

    ax.set_xlim((returns.index[0], returns.index[-1]))

    ax.legend(['Daily holdings',
               'Average daily holdings, by month',
               'Average daily holdings, net'],
              loc=legend_loc)
    ax.set_title('Holdings per Day')
    ax.set_ylabel('Amount of holdings per day')
    ax.set_xlabel('')
    return ax


def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = timeseries.cum_returns(returns, starting_value=1.0)
    df_drawdowns = timeseries.gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])

    ax.set_title('Top %i Drawdown Periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], 'upper left')
    ax.set_xlabel('')
    return ax


def plot_drawdown_underwater(returns, ax=None, **kwargs):
    """Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = timeseries.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    ax.set_ylabel('Drawdown')
    ax.set_title('Underwater Plot')
    ax.set_xlabel('')
    return ax


def show_perf_stats(returns, benchmark_rets, live_start_date=None):
    """Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.

    """

    if live_start_date is not None:
        live_start_date = utils.get_utc_timestamp(live_start_date)
        returns_backtest = returns[returns.index < live_start_date]
        returns_live = returns[returns.index > live_start_date]

        perf_stats_live = np.round(timeseries.perf_stats(
            returns_live, returns_style='arithmetic'), 2)
        perf_stats_live_ab = np.round(
            timeseries.calc_alpha_beta(returns_live, benchmark_rets), 2)
        perf_stats_live.loc['alpha'] = perf_stats_live_ab[0]
        perf_stats_live.loc['beta'] = perf_stats_live_ab[1]
        perf_stats_live.columns = ['Out_of_Sample']

        perf_stats_all = np.round(timeseries.perf_stats(
            returns, returns_style='arithmetic'), 2)
        perf_stats_all_ab = np.round(
            timeseries.calc_alpha_beta(returns, benchmark_rets), 2)
        perf_stats_all.loc['alpha'] = perf_stats_all_ab[0]
        perf_stats_all.loc['beta'] = perf_stats_all_ab[1]
        perf_stats_all.columns = ['All_History']

        print('Out-of-Sample Months: ' + str(int(len(returns_live) / 21)))
    else:
        returns_backtest = returns

    print('Backtest Months: ' + str(int(len(returns_backtest) / 21)))

    perf_stats = np.round(timeseries.perf_stats(
        returns_backtest, returns_style='arithmetic'), 2)
    perf_stats_ab = np.round(
        timeseries.calc_alpha_beta(returns_backtest, benchmark_rets), 2)
    perf_stats.loc['alpha'] = perf_stats_ab[0]
    perf_stats.loc['beta'] = perf_stats_ab[1]
    perf_stats.columns = ['Backtest']

    if live_start_date is not None:
        perf_stats = perf_stats.join(perf_stats_live,
                                     how='inner')
        perf_stats = perf_stats.join(perf_stats_all,
                                     how='inner')

    print(perf_stats)


def plot_rolling_returns(
        returns,
        benchmark_rets=None,
        live_start_date=None,
        cone_std=None,
        legend_loc='best',
        ax=None, **kwargs):
    """Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a linear cone plot may be added to the out-of-sample
    returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    cone_std : float, optional
        The standard deviation to use for the cone plots.
         - The cone is a normal distribution with this standard deviation
             centered around a linear regression.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

"""

    if ax is None:
        ax = plt.gca()

    df_cum_rets = timeseries.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if benchmark_rets is not None:
        timeseries.cum_returns(benchmark_rets[df_cum_rets.index], 1.0).plot(
            lw=2, color='gray', label='S&P500', alpha=0.60, ax=ax, **kwargs)
    if live_start_date is not None:
        live_start_date = utils.get_utc_timestamp(live_start_date)

    if (live_start_date is None) or (df_cum_rets.index[-1] <=
                                     live_start_date):
        df_cum_rets.plot(lw=3, color='forestgreen', alpha=0.6,
                         label='Backtest', ax=ax, **kwargs)
    else:
        df_cum_rets[:live_start_date].plot(
            lw=3, color='forestgreen', alpha=0.6,
            label='Backtest', ax=ax, **kwargs)
        df_cum_rets[live_start_date:].plot(
            lw=4, color='red', alpha=0.6,
            label='Live', ax=ax, **kwargs)

        if cone_std is not None:
            cone_df = timeseries.cone_rolling(
                returns,
                num_stdev=cone_std,
                cone_fit_end_date=live_start_date)

            cone_df_fit = cone_df[cone_df.index < live_start_date]

            cone_df_live = cone_df[cone_df.index > live_start_date]
            cone_df_live = cone_df_live[cone_df_live.index < returns.index[-1]]

            cone_df_fit['line'].plot(
                ax=ax,
                ls='--',
                label='Backtest trend',
                lw=2,
                color='forestgreen',
                alpha=0.7,
                **kwargs)
            cone_df_live['line'].plot(
                ax=ax,
                ls='--',
                label='Predicted trend',
                lw=2,
                color='red',
                alpha=0.7,
                **kwargs)

            ax.fill_between(cone_df_live.index,
                            cone_df_live.sd_down,
                            cone_df_live.sd_up,
                            color='red', alpha=0.30)

    ax.axhline(1.0, linestyle='--', color='black', lw=2)
    ax.set_ylabel('Cumulative returns')
    ax.set_title('Cumulative Returns')
    ax.legend(loc=legend_loc)
    ax.set_xlabel('')

    return ax


def plot_rolling_beta(returns, benchmark_rets, rolling_beta_window=63,
                      legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    rolling_beta_window : int, optional
        The days window over which to compute the beta.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling Portfolio Beta to S&P 500")
    ax.set_ylabel('Beta')
    rb_1 = timeseries.rolling_beta(
        returns, benchmark_rets, rolling_window=rolling_beta_window * 2)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = timeseries.rolling_beta(
        returns, benchmark_rets, rolling_window=rolling_beta_window * 3)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.set_ylim((-2.5, 2.5))
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    ax.set_xlabel('')
    ax.legend(['6-mo',
               '12-mo'],
              loc=legend_loc)
    return ax


def plot_rolling_sharpe(returns, rolling_sharpe_window=63 * 2,
                        legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_sharpe_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(
        returns, rolling_sharpe_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
                           **kwargs)

    ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.axhline(
        rolling_sharpe_ts.mean(),
        color='steelblue',
        linestyle='--',
        lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylim((-3.0, 6.0))
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('')
    ax.legend(['Sharpe', 'Average'],
              loc=legend_loc)
    return ax


def plot_gross_leverage(returns, gross_lev, ax=None, **kwargs):
    """Plots gross leverage versus date.

    Gross leverage is the sum of long and short exposure per share
    divided by net asset value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    gross_lev : pd.Series, optional
        The leverage of a strategy.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

"""

    if ax is None:
        ax = plt.gca()

    gross_lev.plot(alpha=0.8, lw=0.5, color='g', legend=False, ax=ax,
                   **kwargs)

    ax.axhline(gross_lev.mean(), color='g', linestyle='--', lw=3,
               alpha=1.0)

    ax.set_title('Gross Leverage')
    ax.set_ylabel('Gross Leverage')
    ax.set_xlabel('')
    return ax


def plot_exposures(returns, positions_alloc, ax=None, **kwargs):
    """Plots a cake chart of long, short, and cash exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See
        pos.get_portfolio_alloc.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

"""

    if ax is None:
        ax = plt.gca()

    df_long_short = pos.get_long_short_pos(positions_alloc)

    if np.any(df_long_short.cash < 0):
        warnings.warn('Negative cash, taking absolute for area plot.')
        df_long_short = df_long_short.abs()
    df_long_short.plot(
        kind='area', color=['lightblue', 'green', 'coral'], alpha=1.0,
        ax=ax, **kwargs)
    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_title("Long/Short/Cash Exposure")
    ax.set_ylabel('Exposure')
    ax.set_xlabel('')
    return ax


def show_and_plot_top_positions(returns, positions_alloc,
                                show_and_plot=2,
                                legend_loc='real_best', ax=None,
                                **kwargs):
    """Prints and/or plots the exposures of the top 10 held positions of
    all time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_portfolio_alloc.
    show_and_plot : int, optional
        By default, this is 2, and both prints and plots.
        If this is 0, it will only plot; if 1, it will only print.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
        By default, the legend will display below the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes, conditional
        The axes that were plotted on.

    """

    df_top_long, df_top_short, df_top_abs = pos.get_top_long_short_abs(
        positions_alloc)

    if show_and_plot == 1 or show_and_plot == 2:
        print("\n")
        print('Top 10 long positions of all time (and max%)')
        print(pd.DataFrame(df_top_long).index.values)
        print(np.round(pd.DataFrame(df_top_long)[0].values, 3))
        print("\n")

        print('Top 10 short positions of all time (and max%)')
        print(pd.DataFrame(df_top_short).index.values)
        print(np.round(pd.DataFrame(df_top_short)[0].values, 3))
        print("\n")

        print('Top 10 positions of all time (and max%)')
        print(pd.DataFrame(df_top_abs).index.values)
        print(np.round(pd.DataFrame(df_top_abs)[0].values, 3))
        print("\n")

        _, _, df_top_abs_all = pos.get_top_long_short_abs(
            positions_alloc, top=9999)
        print('All positions ever held')
        print(pd.DataFrame(df_top_abs_all).index.values)
        print(np.round(pd.DataFrame(df_top_abs_all)[0].values, 3))
        print("\n")

    if show_and_plot == 0 or show_and_plot == 2:

        if ax is None:
            ax = plt.gca()

        positions_alloc[df_top_abs.index].plot(
            title='Portfolio Allocation Over Time, Only Top 10 Holdings',
            alpha=0.4, ax=ax, **kwargs)

        # Place legend below plot, shrink plot by 20%
        if legend_loc == 'real_best':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(
                loc='upper center', frameon=True, bbox_to_anchor=(
                    0.5, -0.14), ncol=5)
        else:
            ax.legend(loc=legend_loc)

        df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
        ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
        ax.set_ylabel('Exposure by stock')
        return ax


def plot_return_quantiles(returns, df_weekly, df_monthly, ax=None, **kwargs):
    """Creates a box plot of daily, weekly, and monthly return
    distributions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    df_weekly : pd.Series
        Weekly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    df_monthly : pd.Series
        Monthly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    sns.boxplot(data=[returns, df_weekly, df_monthly],
                ax=ax, **kwargs)
    ax.set_xticklabels(['daily', 'weekly', 'monthly'])
    ax.set_title('Return quantiles')
    return ax


def show_return_range(returns, df_weekly):
    """
    Print monthly return and weekly return standard deviations.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    df_weekly : pd.Series
        Weekly returns of the strategy, noncumulative.
         - See timeseries.aggregate_returns.
    """

    two_sigma_daily = returns.mean() - 2 * returns.std()
    two_sigma_weekly = df_weekly.mean() - 2 * df_weekly.std()

    var_sigma = pd.Series([two_sigma_daily, two_sigma_weekly],
                          index=['2-sigma returns daily',
                                 '2-sigma returns weekly'])

    print(np.round(var_sigma, 3))


def plot_turnover(returns, transactions, positions,
                  legend_loc='best', ax=None, **kwargs):
    """Plots turnover vs. date.

    Turnover is the number of shares traded for a period as a fraction
    of total shares.

    Displays daily total, daily average per month, and all-time daily
    average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_turnover = pos.get_turnover(transactions, positions)
    df_turnover_by_month = df_turnover.resample("M")
    df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5, ax=ax, **kwargs)
    df_turnover_by_month.plot(
        color='orangered',
        alpha=0.5,
        lw=2,
        ax=ax,
        **kwargs)
    ax.axhline(
        df_turnover.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    ax.legend(['Daily turnover',
               'Average daily turnover, by month',
               'Average daily turnover, net'],
              loc=legend_loc)
    ax.set_title('Daily Turnover')
    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_ylim((0, 1))
    ax.set_ylabel('Turnover')
    ax.set_xlabel('')
    return ax


def plot_daily_volume(returns, transactions, ax=None, **kwargs):
    """Plots trading volume per day vs. date.

    Also displays all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    transactions.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    ax.axhline(transactions.txn_shares.mean(), color='steelblue',
               linestyle='--', lw=3, alpha=1.0)
    ax.set_title('Daily Trading Volume')
    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_ylabel('Amount of shares traded')
    ax.set_xlabel('')
    return ax


def plot_volume_per_day_hist(transactions, ax=None, **kwargs):
    """Plots a histogram of trading volume per day.

    Parameters
    ----------
    transactions : pd.DataFrame
        Daily transaction volume and dollar ammount.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    sns.distplot(transactions.txn_volume, ax=ax, **kwargs)
    ax.set_title('Distribution of Daily Trading Volume')
    ax.set_xlabel('Volume')
    return ax


def plot_daily_returns_similarity(returns_backtest, returns_live,
                                  title='', scale_kws=None, ax=None,
                                  **kwargs):
    """Plots overlapping distributions of in-sample (backtest) returns
    and out-of-sample (live trading) returns.

    Parameters
    ----------
    returns_backtest : pd.Series
        Daily returns of the strategy's backtest, noncumulative.
    returns_live : pd.Series
        Daily returns of the strategy's live trading, noncumulative.
    title : str, optional
        The title to use for the plot.
    scale_kws : dict, optional
        Additional arguments passed to preprocessing.scale.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()
    if scale_kws is None:
        scale_kws = {}

    sns.kdeplot(preprocessing.scale(returns_backtest, **scale_kws),
                bw='scott', shade=True, label='backtest',
                color='forestgreen', ax=ax, **kwargs)
    sns.kdeplot(preprocessing.scale(returns_live, **scale_kws),
                bw='scott', shade=True, label='out-of-sample',
                color='red', ax=ax, **kwargs)
    ax.set_title(title)

    return ax


def show_worst_drawdown_periods(returns, top=5):
    """Prints information about the worst drawdown periods.

    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).

    """

    print('\nWorst Drawdown Periods')
    drawdown_df = timeseries.gen_drawdown_table(returns, top=top)
    drawdown_df['peak date'] = pd.to_datetime(
        drawdown_df['peak date'],
        unit='D')
    drawdown_df['valley date'] = pd.to_datetime(
        drawdown_df['valley date'],
        unit='D')
    drawdown_df['recovery date'] = pd.to_datetime(
        drawdown_df['recovery date'],
        unit='D')
    drawdown_df['net drawdown in %'] = list(
        map(utils.round_two_dec_places, drawdown_df['net drawdown in %']))
    print(drawdown_df.sort('net drawdown in %', ascending=False))
