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
from functools import partial

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from sklearn import preprocessing

import statsmodels.api as sm

from . import utils
from .utils import APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR
from .utils import DAILY, WEEKLY, MONTHLY, YEARLY, ANNUALIZATION_FACTORS
from .interesting_periods import PERIODS


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

    if convert_to == WEEKLY:
        return df_daily_rets.groupby(
            [lambda x: x.year,
             lambda x: x.month,
             lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == MONTHLY:
        return df_daily_rets.groupby(
            [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == YEARLY:
        return df_daily_rets.groupby(
            [lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )


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


def annual_return(returns, period=DAILY):
    """Determines the annual returns of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.

    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate)

    """

    if returns.size < 1:
        return np.nan

    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be '{}'. "
            "Must be '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY
            )
        )

    num_years = float(len(returns)) / ann_factor
    df_cum_rets = cum_returns(returns, starting_value=100)
    start_value = 100
    end_value = df_cum_rets[-1]

    total_return = (end_value - start_value) / start_value
    annual_return = (1. + total_return) ** (1 / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing volatility. Can be 'monthly' or 'weekly' or 'daily'.
        - defaults to 'daily'

    Returns
    -------
    float
        Annual volatility.
    """

    if returns.size < 2:
        return np.nan

    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be: '{}'."
            " Must be '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY
            )
        )

    return returns.std() * np.sqrt(ann_factor)


def calmar_ratio(returns, period=DAILY):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.


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
            period=period
        ) / abs(max_drawdown(returns=returns))
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

    daily_return_thresh = pow(1 + annual_return_threshhold, 1 /
                              APPROX_BDAYS_PER_YEAR) - 1

    returns_less_thresh = returns - daily_return_thresh

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sortino_ratio(returns, required_return=0, period=DAILY):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized Sortino ratio.

    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be: '{}'."
            " Must be '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY
            )
        )

    mu = np.nanmean(returns - required_return, axis=0)
    sortino = mu / downside_risk(returns, required_return)
    if len(returns.shape) == 2:
        sortino = pd.Series(sortino, index=returns.columns)
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    required_return: float / series
        minimum acceptable return
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized downside deviation

    """
    try:
        ann_factor = ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            "period cannot be: '{}'."
            " Must be '{}', '{}', or '{}'".format(
                period, DAILY, WEEKLY, MONTHLY
            )
        )

    downside_diff = returns - required_return
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = np.nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def sharpe_ratio(returns, risk_free=0, period=DAILY):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.

    Returns
    -------
    float
        Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    returns_risk_adj = returns - risk_free

    return np.mean(returns_risk_adj) / \
        np.std(returns_risk_adj) * \
        np.sqrt(ANNUALIZATION_FACTORS[period])


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


def rolling_beta(returns, factor_returns,
                 rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """Determines the rolling beta of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If DataFrame is passed, computes rolling beta for each column.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 6 months).

    Returns
    -------
    pd.Series
        Rolling beta.

    Note
    -----
    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.

    """
    if factor_returns.ndim > 1:
        # Apply column-wise
        return factor_returns.apply(partial(rolling_beta, returns),
                                    rolling_window=rolling_window)
    else:
        out = pd.Series(index=returns.index)
        for beg, end in zip(returns.index[0:-rolling_window],
                            returns.index[rolling_window:]):
            out.loc[end] = calc_alpha_beta(
                returns.loc[beg:end],
                factor_returns.loc[beg:end])[1]

        return out


def rolling_fama_french(returns, factor_returns=None,
                        rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """Computes rolling Fama-French single factor betas.

    Specifically, returns SMB, HML, and UMD.

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
        Default is 6 months.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing rolling beta coefficients for SMB, HML
        and UMD
    """
    if factor_returns is None:
        factor_returns = utils.load_portfolio_risk_factors(
            start=returns.index[0], end=returns.index[-1])
        factor_returns = factor_returns.drop(['Mkt-RF', 'RF'],
                                             axis='columns')

    return rolling_beta(returns, factor_returns,
                        rolling_window=rolling_window)


def calc_alpha_beta(returns, factor_returns):
    """Calculates both alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float
        Alpha.
    float
        Beta.

"""

    ret_index = returns.index
    beta, alpha = sp.stats.linregress(factor_returns.loc[ret_index].values,
                                      returns.values)[:2]

    return alpha * APPROX_BDAYS_PER_YEAR, beta


def perf_stats(
        returns,
        return_as_dict=False,
        factor_returns=None,
        period=DAILY):
    """Calculates various performance metrics of a strategy, for use in
    plotting.show_perf_stats.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    return_as_dict : boolean, optional
       If True, returns the computed metrics in a dictionary.
    period : str, optional
        - defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'
        - defaults to 'daily'.
    factor_returns : pd.Series (optional)
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If None, do not compute alpha, beta, and information ratio.

    Returns
    -------
    dict / pd.DataFrame
        Performance metrics.

    """

    all_stats = OrderedDict()
    all_stats['annual_return'] = annual_return(returns, period=period)
    all_stats['annual_volatility'] = annual_volatility(returns, period=period)
    all_stats['sharpe_ratio'] = sharpe_ratio(
        returns)
    all_stats['calmar_ratio'] = calmar_ratio(returns, period=period)
    all_stats['stability'] = stability_of_timeseries(returns)
    all_stats['max_drawdown'] = max_drawdown(returns)
    all_stats['omega_ratio'] = omega_ratio(returns)
    all_stats['sortino_ratio'] = sortino_ratio(returns)
    all_stats['skewness'] = stats.skew(returns)
    all_stats['kurtosis'] = stats.kurtosis(returns)
    if factor_returns is not None:
        all_stats['information_ratio'] = information_ratio(returns,
                                                           factor_returns)
        alpha, beta = calc_alpha_beta(returns, factor_returns)
        all_stats['alpha'] = alpha
        all_stats['beta'] = beta

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
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
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
        df_drawdowns.loc[i, 'peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
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
        / pd.rolling_std(returns, rolling_sharpe_window) \
        * np.sqrt(APPROX_BDAYS_PER_YEAR)


def forecast_cone_bootstrap(is_returns, num_days, cone_std=(1., 1.5, 2.),
                            starting_value=1, num_samples=1000,
                            random_seed=None):
    """
    Determines the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Future cumulative mean and
    standard devation are computed by repeatedly sampling from the
    in-sample daily returns (i.e. bootstrap). This cone is non-parametric,
    meaning it does not assume that returns are normally distributed.

    Parameters
    ----------
    is_returns : pd.Series
        In-sample daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    num_days : int
        Number of days to project the probability cone forward.
    cone_std : int, float, or list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    starting_value : int or float
        Starting value of the out of sample period.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.

    Returns
    -------
    pd.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    """

    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for i in range(num_samples):
        samples[i, :] = is_returns.sample(num_days, replace=True,
                                          random_state=seed)

    cum_samples = np.cumprod(1 + samples, axis=1) * starting_value

    cum_mean = cum_samples.mean(axis=0)
    cum_std = cum_samples.std(axis=0)

    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]

    cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
    for num_std in cone_std:
        cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
        cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std

    return cone_bounds


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
    returns_dupe = returns.copy()
    returns_dupe.index = returns_dupe.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in PERIODS.items():
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


def portfolio_returns_metric_weighted(holdings_returns,
                                      exclude_non_overlapping=True,
                                      weight_function=None,
                                      weight_function_window=None,
                                      inverse_weight=False,
                                      portfolio_rebalance_rule='q',
                                      weight_func_transform=None):
    """
    Generates an equal-weight portfolio, or portfolio weighted by
    weight_function

    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.

    exclude_non_overlapping : boolean, optional
       (Only applicable if equal-weight portfolio, e.g. weight_function=None)
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data

    weight_function : function, optional
       Function to be applied to holdings_returns timeseries

    weight_function_window : int, optional
       Rolling window over which weight_function will use as its input values

    inverse_weight : boolean, optional
       If True, high values returned from weight_function will result in lower
       weight for that holding

    portfolio_rebalance_rule : string, optional
       A pandas.resample valid rule. Specifies how frequently to compute
       the weighting criteria

    weight_func_transform : function, optional
       Function applied to value returned from weight_function

    Returns
    -------
    (pd.Series, pd.DataFrame)
        pd.Series : Portfolio returns timeseries.
        pd.DataFrame : All the raw data used in the portfolio returns
           calculations
    """

    if weight_function is None:
        if exclude_non_overlapping:
            holdings_df = pd.DataFrame(holdings_returns).T.dropna()
        else:
            holdings_df = pd.DataFrame(holdings_returns).T.fillna(0)

        holdings_df['port_ret'] = holdings_df.sum(
            axis=1) / len(holdings_returns)
    else:
        holdings_df_na = pd.DataFrame(holdings_returns).T
        holdings_cols = holdings_df_na.columns
        holdings_df = holdings_df_na.dropna()
        holdings_func = pd.rolling_apply(holdings_df,
                                         window=weight_function_window,
                                         func=weight_function).dropna()
        holdings_func_rebal = holdings_func.resample(
            rule=portfolio_rebalance_rule,
            how='last')
        holdings_df = holdings_df.join(
            holdings_func_rebal, rsuffix='_f').fillna(method='ffill').dropna()
        if weight_func_transform is None:
            holdings_func_rebal_t = holdings_func_rebal
            holdings_df = holdings_df.join(
                holdings_func_rebal_t,
                rsuffix='_t').fillna(method='ffill').dropna()
        else:
            holdings_func_rebal_t = holdings_func_rebal.applymap(
                weight_func_transform)
            holdings_df = holdings_df.join(
                holdings_func_rebal_t,
                rsuffix='_t').fillna(method='ffill').dropna()
        transform_columns = list(map(lambda x: x + "_t", holdings_cols))
        if inverse_weight:
            inv_func = 1.0 / holdings_df[transform_columns]
            holdings_df_weights = inv_func.div(inv_func.sum(axis=1),
                                               axis='index')
        else:
            holdings_df_weights = holdings_df[transform_columns] \
                .div(holdings_df[transform_columns].sum(axis=1), axis='index')

        holdings_df_weights.columns = holdings_cols
        holdings_df = holdings_df.join(holdings_df_weights, rsuffix='_w')
        holdings_df_weighted_rets = np.multiply(
            holdings_df[holdings_cols], holdings_df_weights)
        holdings_df_weighted_rets['port_ret'] = holdings_df_weighted_rets.sum(
            axis=1)
        holdings_df = holdings_df.join(holdings_df_weighted_rets,
                                       rsuffix='_wret')

    return holdings_df['port_ret'], holdings_df


def bucket_std(value, bins=[0.12, 0.15, 0.18, 0.21], max_default=0.24):
    """
    Simple quantizing function. For use in binning stdevs into a "buckets"

    Parameters
    ----------
    value : float
       Value corresponding to the the stdev to be bucketed

    bins : list, optional
       Floats used to describe the buckets which the value can be placed

    max_default : float, optional
       If value is greater than all the bins, max_default will be returned

    Returns
    -------
    float
        bin which the value falls into
    """

    annual_vol = value * np.sqrt(252)

    for i in bins:
        if annual_vol <= i:
            return i

    return max_default


def min_max_vol_bounds(value, lower_bound=0.12, upper_bound=0.24):
    """
    Restrict volatility weighting of the lowest volatility asset versus the
    highest volatility asset to a certain limit.
    E.g. Never allocate more than 2x to the lowest volatility asset.
    round up all the asset volatilities that fall below a certain bound
    to a specified "lower bound" and round down all of the asset
    volatilites that fall above a certain bound to a specified "upper bound"

    Parameters
    ----------
    value : float
       Value corresponding to a daily volatility

    lower_bound : float, optional
       Lower bound for the volatility

    upper_bound : float, optional
       Upper bound for the volatility

    Returns
    -------
    float
        The value input, annualized, or the lower_bound or upper_bound
    """

    annual_vol = value * np.sqrt(252)

    if annual_vol < lower_bound:
        return lower_bound

    if annual_vol > upper_bound:
        return upper_bound

    return annual_vol


def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns: float / series

    Returns
    -------
    float
        The information ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/information_ratio for more details.

    """
    active_return = returns - factor_returns
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    return np.mean(active_return) / tracking_error
