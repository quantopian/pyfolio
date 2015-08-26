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

from collections import OrderedDict

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from sklearn import preprocessing

import statsmodels.api as sm

from . import utils


def var_cov_var_normal(P, c, mu=0, sigma=1):
    """Variance-covariance calculation of daily Value-at-Risk in a
    portfolio.

    Parameters
    ----------
    P : float
        Portfolio value.
    c : float
        Confidence level.
    mu : float, optional
        Mean.

    Returns
    -------
    float
        Variance-covariance.

    """

    alpha = sp.stats.norm.ppf(1 - c, mu, sigma)
    return P - P * (alpha + 1)


def normalize(returns, starting_value=1):
    """
    Normalizes a returns timeseries based on the first value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    starting_value : float, optional
       The starting returns (default 1).

    Returns
    -------
    pd.Series
        Normalized returns.
    """

    return starting_value * (returns / returns.iloc[0])


def cum_returns(returns, starting_value=None):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    starting_value : float, optional
       The starting returns (default 1).

    Returns
    -------
    pandas.Series
        Series of cumulative returns.

    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    """

    # df_price.pct_change() adds a nan in first position, we can use
    # that to have cum_returns start at the origin so that
    # df_cum.iloc[0] == starting_value
    # Note that we can't add that ourselves as we don't know which dt
    # to use.
    if pd.isnull(returns.iloc[0]):
        returns.iloc[0] = 0.

    df_cum = np.exp(np.log(1 + returns).cumsum())

    if starting_value is None:
        return df_cum - 1
    else:
        return df_cum * starting_value


def aggregate_returns(df_daily_rets, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    df_daily_rets : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in tears.create_full_tear_sheet (returns).
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    def cumulate_returns(x):
        return cum_returns(x)[-1]

    if convert_to == 'weekly':
        return df_daily_rets.groupby(
            [lambda x: x.year,
             lambda x: x.month,
             lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == 'monthly':
        return df_daily_rets.groupby(
            [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == 'yearly':
        return df_daily_rets.groupby(
            [lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be weekly, monthly or yearly')


def max_drawdown(returns):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        Maximum drawdown.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    if returns.size < 1:
        return np.nan

    df_cum_rets = cum_returns(returns, starting_value=100)

    MDD = 0
    DD = 0
    peak = -99999
    for value in df_cum_rets:
        if (value > peak):
            peak = value
        else:
            DD = (peak - value) / peak
        if (DD > MDD):
            MDD = DD
    return -1 * MDD


def annual_return(returns, style='compound'):
    """Determines the annual returns of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    style : str, optional
        - If 'compound', then return will be calculated in geometric
          terms: (1+mean(all_daily_returns))^252 - 1.
        - If 'calendar', then return will be calculated as
          ((last_value - start_value)/start_value)/num_of_years.
        - Otherwise, return is simply mean(all_daily_returns)*252.

    Returns
    -------
    float
        Annual returns.

    """

    if returns.size < 1:
        return np.nan

    if style == 'calendar':
        num_years = len(returns) / 252.0
        df_cum_rets = cum_returns(returns, starting_value=100)
        start_value = df_cum_rets[0]
        end_value = df_cum_rets[-1]
        return ((end_value - start_value) / start_value) / num_years
    if style == 'compound':
        return pow((1 + returns.mean()), 252) - 1
    else:
        return returns.mean() * 252


def annual_volatility(returns):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        Annual volatility.
    """

    if returns.size < 2:
        return np.nan

    return returns.std() * np.sqrt(252)


def calmar_ratio(returns, returns_style='calendar'):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    returns_style : str, optional
        See annual_returns' style

    Returns
    -------
    float
        Calmar ratio (drawdown ratio).

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    temp_max_dd = max_drawdown(returns=returns)
    if temp_max_dd < 0:
        temp = annual_return(
            returns=returns,
            style=returns_style) / abs(max_drawdown(returns=returns))
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, annual_return_threshhold=0.0):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    annual_return_threshold : float, optional
        Threshold over which to consider positive vs negative
        returns. For the ratio, it will be converted to a daily return
        and compared to returns.

    Returns
    -------
    float
        Omega ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

"""

    daily_return_thresh = pow(1 + annual_return_threshhold, 1 / 252) - 1

    returns_less_thresh = returns - daily_return_thresh

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sortino_ratio(returns, returns_style='compound'):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        Sortino ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sortino_ratio for more details.
    """
    numer = annual_return(returns, style=returns_style)
    denom = annual_volatility(returns[returns < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns, returns_style='compound'):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    returns_style : str, optional
        See annual_returns' style

    Returns
    -------
    float
        Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    numer = annual_return(returns, style=returns_style)
    denom = annual_volatility(returns)

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the returns.

    Computes an ordinary least squares linear fit, and returns
    R-squared.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        R-squared.

    """

    if returns.size < 2:
        return np.nan

    df_cum_rets = cum_returns(returns, starting_value=100)
    df_cum_rets_log = np.log10(df_cum_rets.values)
    len_returns = df_cum_rets.size

    X = list(range(0, len_returns))
    X = sm.add_constant(X)

    model = sm.OLS(df_cum_rets_log, X).fit()

    return model.rsquared


def out_of_sample_vs_in_sample_returns_kde(
        bt_ts, oos_ts, transform_style='scale', return_zero_if_exception=True):
    """Determines similarity between two returns timeseries.

    Typically a backtest frame (in-sample) and live frame
    (out-of-sample).

    Parameters
    ----------
    bt_ts : pd.Series
       In-sample (backtest) returns of the strategy, noncumulative.
        - See full explanation in tears.create_full_tear_sheet (returns).
    oos_ts : pd.Series
       Out-of-sample (live trading) returns of the strategy,
       noncumulative.
        - See full explanation in tears.create_full_tear_sheet (returns).
    transform_style : float, optional
        'raw', 'scale', 'Normalize_L1', 'Normalize_L2' (default
        'scale')
    return_zero_if_exception : bool, optional
        If there is an exception, return zero instead of NaN.

    Returns
    -------
    float
        Similarity between returns.

    """

    bt_ts_pct = bt_ts.dropna()
    oos_ts_pct = oos_ts.dropna()

    bt_ts_r = bt_ts_pct.reshape(len(bt_ts_pct), 1)
    oos_ts_r = oos_ts_pct.reshape(len(oos_ts_pct), 1)

    if transform_style == 'raw':
        bt_scaled = bt_ts_r
        oos_scaled = oos_ts_r
    if transform_style == 'scale':
        bt_scaled = preprocessing.scale(bt_ts_r, axis=0)
        oos_scaled = preprocessing.scale(oos_ts_r, axis=0)
    if transform_style == 'normalize_L2':
        bt_scaled = preprocessing.normalize(bt_ts_r, axis=1)
        oos_scaled = preprocessing.normalize(oos_ts_r, axis=1)
    if transform_style == 'normalize_L1':
        bt_scaled = preprocessing.normalize(bt_ts_r, axis=1, norm='l1')
        oos_scaled = preprocessing.normalize(oos_ts_r, axis=1, norm='l1')

    X_train = bt_scaled
    X_test = oos_scaled

    X_train = X_train.reshape(len(X_train))
    X_test = X_test.reshape(len(X_test))

    x_axis_dim = np.linspace(-4, 4, 100)
    kernal_method = 'scott'

    try:
        scipy_kde_train = stats.gaussian_kde(
            X_train,
            bw_method=kernal_method)(x_axis_dim)
        scipy_kde_test = stats.gaussian_kde(
            X_test,
            bw_method=kernal_method)(x_axis_dim)
    except:
        if return_zero_if_exception:
            return 0.0
        else:
            return np.nan

    kde_diff = sum(abs(scipy_kde_test - scipy_kde_train)) / \
        (sum(scipy_kde_train) + sum(scipy_kde_test))

    return kde_diff


def calc_multifactor(returns, factors):
    """Computes multiple ordinary least squares linear fits, and returns
    fit parameters.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factors : pd.Series
        Secondary sets to fit.

    Returns
    -------
    pd.DataFrame
        Fit parameters.

    """

    import statsmodels.api as sm
    factors = factors.loc[returns.index]
    factors = sm.add_constant(factors)
    factors = factors.dropna(axis=0)
    results = sm.OLS(returns[factors.index], factors).fit()

    return results.params


def rolling_beta(returns, benchmark_rets, rolling_window=63):
    """Determines the rolling beta of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 63 days).

    Returns
    -------
    pd.Series
        Rolling beta.

    Note
    -----
    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.

    """

    out = pd.Series(index=returns.index)
    for beg, end in zip(returns.index[0:-rolling_window],
                        returns.index[rolling_window:]):
        out.loc[end] = calc_alpha_beta(returns.loc[beg:end],
                                       benchmark_rets.loc[beg:end])[1]

    return out


def rolling_multifactor_beta(returns, df_multi_factor, rolling_window=63):
    """Determines the rolling beta of multiple factors.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    df_multi_factor : pd.DataFrame
        Other factors over which to compute beta.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 63 days).

    Returns
    -------
    pd.DataFrame
        Rolling betas.

    Note
    -----

    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.

    """

    out = pd.DataFrame(columns=['const'] + list(df_multi_factor.columns),
                       index=returns.index)

    for beg, end in zip(returns.index[0:-rolling_window],
                        returns.index[rolling_window:]):
        out.loc[end] = calc_multifactor(returns.loc[beg:end],
                                        df_multi_factor.loc[beg:end])

    return out


def rolling_risk_factors(returns, risk_factors=None,
                         rolling_beta_window=63 * 2):
    """Computes rolling Fama-French single factor betas.

    Specifically, returns SMB, HML, and UMD.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    risk_factors : pd.DataFrame, optional
        data set containing the risk factors. See
        utils.load_portfolio_risk_factors.
    rolling_beta_window : int, optional
        The days window over which to compute the beta.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing rolling beta coefficients for SMB, HML
        and UMD
    """
    if risk_factors is None:
        risk_factors = utils.load_portfolio_risk_factors(
            start=returns.index[0],
            end=returns.index[-1])

    rolling_beta_SMB = rolling_beta(
        returns,
        risk_factors['SMB'],
        rolling_window=rolling_beta_window)
    rolling_beta_HML = rolling_beta(
        returns,
        risk_factors['HML'],
        rolling_window=rolling_beta_window)
    rolling_beta_UMD = rolling_beta(
        returns,
        risk_factors['UMD'],
        rolling_window=rolling_beta_window)

    rolling_factors = pd.concat([rolling_beta_SMB, rolling_beta_HML,
                                 rolling_beta_UMD])

    rolling_factors.columns = ['SMB', 'HML', 'UMD']

    return rolling_factors


def calc_alpha_beta(returns, benchmark_rets):
    """
    Calculates both alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """

    ret_index = returns.index
    beta, alpha = sp.stats.linregress(benchmark_rets.loc[ret_index].values,
                                      returns.values)[:2]

    return alpha * 252, beta


def perf_stats(
        returns,
        returns_style='compound',
        return_as_dict=False):
    """Calculates various performance metrics of a strategy, for use in
    plotting.show_perf_stats.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    returns_style : str, optional
       See annual_returns' style
    return_as_dict : boolean, optional
       If True, returns the computed metrics in a dictionary.

    Returns
    -------
    dict / pd.DataFrame
        Performance metrics.

    """

    all_stats = OrderedDict()
    all_stats['annual_return'] = annual_return(
        returns,
        style=returns_style)
    all_stats['annual_volatility'] = annual_volatility(returns)
    all_stats['sharpe_ratio'] = sharpe_ratio(
        returns,
        returns_style=returns_style)
    all_stats['calmar_ratio'] = calmar_ratio(
        returns,
        returns_style=returns_style)
    all_stats['stability'] = stability_of_timeseries(returns)
    all_stats['max_drawdown'] = max_drawdown(returns)
    all_stats['omega_ratio'] = omega_ratio(returns)
    all_stats['sortino_ratio'] = sortino_ratio(returns)
    all_stats['skewness'] = stats.skew(returns)
    all_stats['kurtosis'] = stats.kurtosis(returns)

    if return_as_dict:
        return all_stats
    else:
        all_stats_df = pd.DataFrame(
            index=list(all_stats.keys()),
            data=list(all_stats.values()))
        all_stats_df.columns = ['perf_stats']
        return all_stats_df


def get_max_drawdown_underwater(underwater):
    """Determines peak, valley, and recovery dates given and 'underwater'
    DataFrame.

    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.

    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.

    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.

    """

    valley = np.argmax(underwater)  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


def get_max_drawdown(returns):
    """
    Finds maximum drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = (running_max - df_cum) / running_max
    return get_max_drawdown_underwater(underwater)


def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater = pd.concat(
                [underwater.loc[:peak].iloc[:-1],
                 underwater.loc[recovery:].iloc[1:]])
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns


def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).

    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['net drawdown in %',
                                         'peak date',
                                         'valley date',
                                         'recovery date',
                                         'duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'peak date'] = peak
        df_drawdowns.loc[i, 'valley date'] = valley
        df_drawdowns.loc[i, 'recovery date'] = recovery
        df_drawdowns.loc[i, 'net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['peak date'] = pd.to_datetime(
        df_drawdowns['peak date'],
        unit='D')
    df_drawdowns['valley date'] = pd.to_datetime(
        df_drawdowns['valley date'],
        unit='D')
    df_drawdowns['recovery date'] = pd.to_datetime(
        df_drawdowns['recovery date'],
        unit='D')

    return df_drawdowns


def rolling_sharpe(returns, rolling_sharpe_window):
    """
    Determines the rolling Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_sharpe_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    return pd.rolling_mean(returns, rolling_sharpe_window) \
        / pd.rolling_std(returns, rolling_sharpe_window) * np.sqrt(252)


def cone_rolling(
        input_rets,
        num_stdev=1.0,
        warm_up_days_pct=0.5,
        std_scale_factor=252,
        update_std_oos_rolling=False,
        cone_fit_end_date=None,
        extend_fit_trend=True,
        create_future_cone=True):
    """Computes a rolling cone to place in the cumulative returns
    plot. See plotting.plot_rolling_returns.
    """

    # if specifying 'cone_fit_end_date' please use a pandas compatible format,
    # e.g. '2015-8-4', 'YYYY-MM-DD'

    warm_up_days = int(warm_up_days_pct * input_rets.size)

    # create initial linear fit from beginning of timeseries thru warm_up_days
    # or the specified 'cone_fit_end_date'
    if cone_fit_end_date is None:
        returns = input_rets[:warm_up_days]
    else:
        returns = input_rets[input_rets.index < cone_fit_end_date]

    perf_ts = cum_returns(returns, 1)

    X = list(range(0, perf_ts.size))
    X = sm.add_constant(X)
    sm.OLS(perf_ts, list(range(0, len(perf_ts))))
    line_ols = sm.OLS(perf_ts.values, X).fit()
    fit_line_ols_coef = line_ols.params[1]
    fit_line_ols_inter = line_ols.params[0]

    x_points = list(range(0, perf_ts.size))
    x_points = np.array(x_points) * fit_line_ols_coef + fit_line_ols_inter

    perf_ts_r = pd.DataFrame(perf_ts)
    perf_ts_r.columns = ['perf']

    warm_up_std_pct = np.std(perf_ts.pct_change().dropna())
    std_pct = warm_up_std_pct * np.sqrt(std_scale_factor)

    perf_ts_r['line'] = x_points
    perf_ts_r['sd_up'] = perf_ts_r['line'] * (1 + num_stdev * std_pct)
    perf_ts_r['sd_down'] = perf_ts_r['line'] * (1 - num_stdev * std_pct)

    std_pct = warm_up_std_pct * np.sqrt(std_scale_factor)

    last_backtest_day_index = returns.index[-1]
    cone_end_rets = input_rets[input_rets.index > last_backtest_day_index]
    new_cone_day_scale_factor = int(1)
    oos_intercept_shift = perf_ts_r.perf[-1] - perf_ts_r.line[-1]

    # make the cone for the out-of-sample/live papertrading period
    for i in cone_end_rets.index:
        returns = input_rets[:i]
        perf_ts = cum_returns(returns, 1)

        if extend_fit_trend:
            line_ols_coef = fit_line_ols_coef
            line_ols_inter = fit_line_ols_inter
        else:
            X = list(range(0, perf_ts.size))
            X = sm.add_constant(X)
            sm.OLS(perf_ts, list(range(0, len(perf_ts))))
            line_ols = sm.OLS(perf_ts.values, X).fit()
            line_ols_coef = line_ols.params[1]
            line_ols_inter = line_ols.params[0]

        x_points = list(range(0, perf_ts.size))
        x_points = np.array(x_points) * line_ols_coef + \
            line_ols_inter + oos_intercept_shift

        temp_line = x_points
        if update_std_oos_rolling:
            std_pct = np.sqrt(new_cone_day_scale_factor) * \
                np.std(perf_ts.pct_change().dropna())
        else:
            std_pct = np.sqrt(new_cone_day_scale_factor) * warm_up_std_pct

        temp_sd_up = temp_line * (1 + num_stdev * std_pct)
        temp_sd_down = temp_line * (1 - num_stdev * std_pct)

        new_daily_cone = pd.DataFrame(index=[i],
                                      data={'perf': perf_ts[i],
                                            'line': temp_line[-1],
                                            'sd_up': temp_sd_up[-1],
                                            'sd_down': temp_sd_down[-1]})

        perf_ts_r = perf_ts_r.append(new_daily_cone)
        new_cone_day_scale_factor += 1

    if create_future_cone:
        extend_ahead_days = 252
        future_cone_dates = pd.date_range(
            cone_end_rets.index[-1], periods=extend_ahead_days, freq='B')

        future_cone_intercept_shift = perf_ts_r.perf[-1] - perf_ts_r.line[-1]

        future_days_scale_factor = np.linspace(
            1,
            extend_ahead_days,
            extend_ahead_days)
        std_pct = np.sqrt(future_days_scale_factor) * warm_up_std_pct

        x_points = list(range(perf_ts.size, perf_ts.size + extend_ahead_days))
        x_points = np.array(x_points) * line_ols_coef + line_ols_inter + \
            oos_intercept_shift + future_cone_intercept_shift
        temp_line = x_points
        temp_sd_up = temp_line * (1 + num_stdev * std_pct)
        temp_sd_down = temp_line * (1 - num_stdev * std_pct)

        future_cone = pd.DataFrame(index=list(map(np.datetime64,
                                                  future_cone_dates)),
                                   data={'perf': temp_line,
                                         'line': temp_line,
                                         'sd_up': temp_sd_up,
                                         'sd_down': temp_sd_down})

        perf_ts_r = perf_ts_r.append(future_cone)

    return perf_ts_r


def gen_date_ranges_interesting():
    """Generates a list of historical event dates that may have had
    significant impact on markets.  See
    extract_interesting_date_ranges.

    Returns
    -------
    periods : OrderedDict
        Significant events.

    """

    periods = OrderedDict()
    # Dotcom bubble
    periods['Dotcom'] = (pd.Timestamp('20000310'), pd.Timestamp('20000910'))

    # Lehmann Brothers
    periods['Lehmann'] = (pd.Timestamp('20080801'), pd.Timestamp('20081001'))

    # 9/11
    periods['9/11'] = (pd.Timestamp('20010911'), pd.Timestamp('20011011'))

    # 05/08/11  US down grade and European Debt Crisis 2011
    periods[
        'US downgrade/European Debt Crisis'] = (pd.Timestamp('20110805'),
                                                pd.Timestamp('20110905'))

    # 16/03/11  Fukushima melt down 2011
    periods['Fukushima'] = (pd.Timestamp('20110316'), pd.Timestamp('20110416'))

    # 01/08/03  US Housing Bubble 2003
    periods['US Housing'] = (
        pd.Timestamp('20030108'), pd.Timestamp('20030208'))

    # 06/09/12  EZB IR Event 2012
    periods['EZB IR Event'] = (
        pd.Timestamp('20120910'), pd.Timestamp('20121010'))

    # August 2007, March and September of 2008, Q1 & Q2 2009,
    periods['Aug07'] = (pd.Timestamp('20070801'), pd.Timestamp('20070901'))
    periods['Mar08'] = (pd.Timestamp('20080301'), pd.Timestamp('20070401'))
    periods['Sept08'] = (pd.Timestamp('20080901'), pd.Timestamp('20081001'))
    periods['2009Q1'] = (pd.Timestamp('20090101'), pd.Timestamp('20090301'))
    periods['2009Q2'] = (pd.Timestamp('20090301'), pd.Timestamp('20090601'))

    # Flash Crash (May 6, 2010 + 1 week post),
    periods['Flash Crash'] = (
        pd.Timestamp('20100505'), pd.Timestamp('20100510'))

    # April and October 2014).
    periods['Apr14'] = (pd.Timestamp('20140401'), pd.Timestamp('20140501'))
    periods['Oct14'] = (pd.Timestamp('20141001'), pd.Timestamp('20141101'))

    return periods


def extract_interesting_date_ranges(returns):
    """Extracts returns based on interesting events. See
    gen_date_range_interesting.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    ranges : OrderedDict
        Date ranges, with returns, of all valid events.

    """

    periods = gen_date_ranges_interesting()
    returns_dupe = returns.copy()
    returns_dupe.index = returns_dupe.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in periods.items():
        try:
            period = returns_dupe.loc[start:end]
            if len(period) == 0:
                continue
            ranges[name] = period
        except:
            continue

    return ranges


def portfolio_returns(holdings_returns, exclude_non_overlapping=True):
    """Generates an equal-weight portfolio.

    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.

    exclude_non_overlapping : boolean, optional
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data

    Returns
    -------
    pd.Series
        Equal-weight returns timeseries.
    """
    port = holdings_returns[0]
    for i in range(1, len(holdings_returns)):
        port = port + holdings_returns[i]

    if exclude_non_overlapping:
        port = port.dropna()
    else:
        port = port.fillna(0)

    return port / len(holdings_returns)
