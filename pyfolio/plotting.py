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

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines

from sklearn import preprocessing

from . import utils
from . import timeseries
from . import pos
from . import _seaborn as sns
from . import txn

from .utils import APPROX_BDAYS_PER_MONTH

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
        factor_returns=None,
        rolling_window=APPROX_BDAYS_PER_MONTH * 6,
        legend_loc='best',
        ax=None, **kwargs):
    """Plots rolling Fama-French single factor betas.

    Specifically, plots SMB, HML, and UMD vs. date with a legend.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.DataFrame, optional
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

    ax.set_title(
        "Rolling Fama-French Single Factor Betas (%.0f-month)" % (
            rolling_window / APPROX_BDAYS_PER_MONTH
        )
    )

    ax.set_ylabel('beta')

    rolling_beta = timeseries.rolling_fama_french(
        returns,
        factor_returns=factor_returns,
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

    ax.hist(
        100 * monthly_ret_table,
        color='orangered',
        alpha=0.80,
        bins=20,
        **kwargs)

    ax.axvline(
        100 * monthly_ret_table.mean(),
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
    ax.legend(['Portfolio'], loc='upper left')
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


def show_perf_stats(returns, factor_returns, live_start_date=None):
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
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.

    """

    if live_start_date is not None:
        live_start_date = utils.get_utc_timestamp(live_start_date)
        returns_backtest = returns[returns.index < live_start_date]
        returns_live = returns[returns.index > live_start_date]

        perf_stats_live = np.round(timeseries.perf_stats(
            returns_live,
            factor_returns=factor_returns), 2)
        perf_stats_live.columns = ['Out_of_Sample']

        perf_stats_all = np.round(timeseries.perf_stats(
            returns,
            factor_returns=factor_returns), 2)
        perf_stats_all.columns = ['All_History']

        print('Out-of-Sample Months: ' +
              str(int(len(returns_live) / APPROX_BDAYS_PER_MONTH)))
    else:
        returns_backtest = returns

    print('Backtest Months: ' +
          str(int(len(returns_backtest) / APPROX_BDAYS_PER_MONTH)))

    perf_stats = np.round(timeseries.perf_stats(
        returns_backtest,
        factor_returns=factor_returns), 2)
    perf_stats.columns = ['Backtest']

    if live_start_date is not None:
        perf_stats = perf_stats.join(perf_stats_live,
                                     how='inner')
        perf_stats = perf_stats.join(perf_stats_all,
                                     how='inner')

    print(perf_stats)


def plot_rolling_returns(returns,
                         factor_returns=None,
                         live_start_date=None,
                         cone_std=None,
                         legend_loc='best',
                         volatility_match=False,
                         cone_function=timeseries.forecast_cone_bootstrap,
                         ax=None, **kwargs):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of a risk factor.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
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

    ax.set_ylabel('Cumulative returns')
    ax.set_xlabel('')

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of'
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = timeseries.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.one_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = timeseries.cum_returns(
            factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.plot(lw=2, color='gray',
                                label=factor_returns.name, alpha=0.60,
                                ax=ax, **kwargs)

    if live_start_date is not None:
        live_start_date = utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(lw=3, color='forestgreen', alpha=0.6,
                        label='Backtest', ax=ax, **kwargs)

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(lw=4, color='red', alpha=0.6,
                             label='Live', ax=ax, **kwargs)

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1])

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)

            for std in cone_std:
                ax.fill_between(cone_bounds.index,
                                cone_bounds[float(std)],
                                cone_bounds[float(-std)],
                                color='steelblue', alpha=0.5)

    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    ax.axhline(1.0, linestyle='--', color='black', lw=2)

    return ax


def plot_rolling_beta(returns, factor_returns, legend_loc='best',
                      ax=None, **kwargs):
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
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

    ax.set_title("Rolling Portfolio Beta to " + str(factor_returns.name))
    ax.set_ylabel('Beta')
    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.set_ylim((-2.5, 2.5))
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    ax.set_xlabel('')
    ax.legend(['6-mo',
               '12-mo'],
              loc=legend_loc)
    return ax


def plot_rolling_sharpe(returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                        legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_window : int, optional
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
        returns, rolling_window)
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
    """Plots a cake chart of the long and short exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See
        pos.get_percent_alloc.
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

    df_long_short.plot(
        kind='area', color=['lightblue', 'green'], alpha=1.0,
        ax=ax, **kwargs)
    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_title("Long/Short Exposure")
    ax.set_ylabel('Exposure')
    ax.set_xlabel('')
    return ax


def show_and_plot_top_positions(returns, positions_alloc,
                                show_and_plot=2, hide_positions=False,
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
        Portfolio allocation of positions. See pos.get_percent_alloc.
    show_and_plot : int, optional
        By default, this is 2, and both prints and plots.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
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

        if hide_positions:
            ax.legend_.remove()

        return ax


def plot_max_median_position_concentration(positions, ax=None, **kwargs):
    """
    Plots the max and median of long and short position concentrations
    over the time.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.gcf()

    alloc_summary = pos.get_max_median_position_concentration(positions)
    colors = ['mediumblue', 'steelblue', 'tomato', 'firebrick']
    alloc_summary.plot(linewidth=1, color=colors, alpha=0.6, ax=ax)

    ax.legend(loc='center left')
    ax.set_ylabel('Exposure')
    ax.set_title('Long/Short Max and Median Position Concentration')

    return ax


def plot_sector_allocations(returns, sector_alloc, ax=None, **kwargs):
    """Plots the sector exposures of the portfolio over time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    sector_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_sector_alloc.
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
        ax = plt.gcf()

    sector_alloc.plot(title='Sector Allocation Over Time',
                      alpha=0.4, ax=ax, **kwargs)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        loc='upper center', frameon=True, bbox_to_anchor=(
            0.5, -0.14), ncol=5)

    ax.set_xlim((sector_alloc.index[0], sector_alloc.index[-1]))
    ax.set_ylabel('Exposure by sector')

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
        Prices and amounts of executed trades. One row per trade.
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

    df_turnover = txn.get_turnover(positions, transactions)
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


def plot_slippage_sweep(returns, transactions, positions,
                        slippage_params=(3, 8, 10, 12, 15, 20, 50),
                        ax=None, **kwargs):
    """Plots a equity curves at different per-dollar slippage assumptions.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    slippage_params: tuple
        Slippage pameters to apply to the return time series (in
        basis points).
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

    turnover = txn.get_turnover(positions, transactions,
                                period=None, average=False)

    slippage_sweep = pd.DataFrame()
    for bps in slippage_params:
        adj_returns = txn.adjust_returns_for_slippage(returns, turnover, bps)
        label = str(bps) + " bps"
        slippage_sweep[label] = timeseries.cum_returns(adj_returns, 1)

    slippage_sweep.plot(alpha=1.0, lw=0.5, ax=ax)

    ax.set_title('Cumulative Returns Given Additional Per-Dollar Slippage')
    ax.set_ylabel('')

    ax.legend(loc='center left')

    return ax


def plot_slippage_sensitivity(returns, transactions, positions,
                              ax=None, **kwargs):
    """Plots curve relating per-dollar slippage to average annual returns.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
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

    turnover = txn.get_turnover(positions, transactions,
                                period=None, average=False)
    avg_returns_given_slippage = pd.Series()
    for bps in range(1, 100):
        adj_returns = txn.adjust_returns_for_slippage(returns, turnover, bps)
        avg_returns = timeseries.annual_return(
            adj_returns)
        avg_returns_given_slippage.loc[bps] = avg_returns

    avg_returns_given_slippage.plot(alpha=1.0, lw=2, ax=ax)

    ax.set(title='Average Annual Returns Given Additional Per-Dollar Slippage',
           xticks=np.arange(0, 100, 10),
           ylabel='Average Annual Return',
           xlabel='Per-Dollar Slippage (bps)')

    return ax


def plot_daily_turnover_hist(transactions, positions,
                             ax=None, **kwargs):
    """Plots a histogram of daily turnover rates.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
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
    turnover = txn.get_turnover(positions, transactions, period=None)
    sns.distplot(turnover, ax=ax, **kwargs)
    ax.set_title('Distribution of Daily Turnover Rates')
    ax.set_xlabel('Turnover Rate')
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
        Prices and amounts of executed trades. One row per trade.
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
    daily_txn = txn.get_txn_vol(transactions)
    daily_txn.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    ax.axhline(daily_txn.txn_shares.mean(), color='steelblue',
               linestyle='--', lw=3, alpha=1.0)
    ax.set_title('Daily Trading Volume')
    df_cum_rets = timeseries.cum_returns(returns, starting_value=1)
    ax.set_xlim((df_cum_rets.index[0], df_cum_rets.index[-1]))
    ax.set_ylabel('Amount of shares traded')
    ax.set_xlabel('')
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
    drawdown_df['net drawdown in %'] = list(
        map(utils.round_two_dec_places, drawdown_df['net drawdown in %']))
    print(drawdown_df.sort('net drawdown in %', ascending=False))


def plot_monthly_returns_timeseries(returns, ax=None, **kwargs):
    """
    Plots monthly returns as a timeseries.

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

    def cumulate_returns(x):
        return timeseries.cum_returns(x)[-1]

    if ax is None:
        ax = plt.gca()

    monthly_rets = returns.resample('M', how=cumulate_returns).to_period()

    sns.barplot(x=monthly_rets.index,
                y=monthly_rets.values,
                color='steelblue')

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    # only show x-labels on year boundary
    xticks_coord = []
    xticks_label = []
    count = 0
    for i in monthly_rets.index:
        if i.month == 1:
            xticks_label.append(i)
            xticks_coord.append(count)
            # plot yearly boundary line
            ax.axvline(count, color='gray', ls='--', alpha=0.3)

        count += 1

    ax.axhline(0.0, color='darkgray', ls='-')
    ax.set_xticks(xticks_coord)
    ax.set_xticklabels(xticks_label)

    return ax


def plot_round_trip_life_times(round_trips, ax=None):
    """
    Plots timespans and directions of round trip trades.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        ax = plt.subplot()

    symbols = round_trips.symbol.unique()
    symbol_idx = pd.Series(np.arange(len(symbols)), index=symbols)

    for symbol, sym_round_trips in round_trips.groupby('symbol'):
        for _, row in sym_round_trips.iterrows():
            c = 'b' if row.long else 'r'
            y_ix = symbol_idx[symbol]
            ax.plot([row['open_dt'], row['close_dt']],
                    [y_ix, y_ix], color=c)

    ax.set_yticklabels(symbols)

    red_line = mlines.Line2D([], [], color='r', label='Short')
    blue_line = mlines.Line2D([], [], color='b', label='Long')
    ax.legend(handles=[red_line, blue_line], loc=0)

    return ax


def show_profit_attribution(round_trips):
    """
    Prints the share of total PnL contributed by each
    traded name.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    total_pnl = round_trips['pnl'].sum()
    pct_profit_attribution = round_trips.groupby(
        'symbol')['pnl'].sum() / total_pnl

    print('\nProfitability (PnL / PnL total) per name:')
    print(pct_profit_attribution.sort(inplace=False, ascending=False))


def plot_prob_profit_trade(round_trips, ax=None):
    """
    Plots a probability distribution for the event of making
    a profitable trade.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    x = np.linspace(0, 1., 500)

    round_trips['profitable'] = round_trips.pnl > 0

    dist = sp.stats.beta(round_trips.profitable.sum(),
                         (~round_trips.profitable).sum())
    y = dist.pdf(x)
    lower_perc = dist.ppf(.025)
    upper_perc = dist.ppf(.975)

    lower_plot = dist.ppf(.001)
    upper_plot = dist.ppf(.999)

    if ax is None:
        ax = plt.subplot()

    ax.plot(x, y)
    ax.axvline(lower_perc, color='0.5')
    ax.axvline(upper_perc, color='0.5')

    ax.set(xlabel='Probability making a profitable decision', ylabel='Belief',
           xlim=(lower_plot, upper_plot), ylim=(0, y.max() + 1.))

    return ax
