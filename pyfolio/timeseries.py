#
# Copyright 2016 Quantopian, Inc.
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

from . import utils
from .utils import APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR
from .utils import DAILY
from .interesting_periods import PERIODS
from .deprecate import deprecated

import empyrical

DEPRECATION_WARNING = ("Risk functions in pyfolio.timeseries are deprecated "
                       "and will be removed in a future release. Please "
                       "install the empyrical package instead.")


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


@deprecated(msg=DEPRECATION_WARNING)
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

    return empyrical.max_drawdown(returns)


@deprecated(msg=DEPRECATION_WARNING)
def annual_return(returns, period=DAILY):
    """Determines the mean annual growth rate of returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """

    return empyrical.annual_return(returns, period=period)


@deprecated(msg=DEPRECATION_WARNING)
def annual_volatility(returns, period=DAILY):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing volatility. Can be 'monthly' or 'weekly' or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Annual volatility.
    """

    return empyrical.annual_volatility(returns, period=period)


@deprecated(msg=DEPRECATION_WARNING)
def calmar_ratio(returns, period=DAILY):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    return empyrical.calmar_ratio(returns, period=period)


@deprecated(msg=DEPRECATION_WARNING)
def omega_ratio(returns, annual_return_threshhold=0.0):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    annual_return_threshold : float, optional
        Minimum acceptable return of the investor. Annual threshold over which
        returns are considered positive or negative. It is converted to a
        value appropriate for the period of the returns for this ratio.
        E.g. An annual minimum acceptable return of 100 translates to a daily
        minimum acceptable return of 0.01848.
            (1 + 100) ** (1. / 252) - 1 = 0.01848
        Daily returns must exceed this value to be considered positive. The
        daily return yields the desired annual return when compounded over
        the average number of business days in a year.
            (1 + 0.01848) ** 252 - 1 = 99.93
        - Defaults to 0.0


    Returns
    -------
    float
        Omega ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """

    return empyrical.omega_ratio(returns,
                                 required_return=annual_return_threshhold)


@deprecated(msg=DEPRECATION_WARNING)
def sortino_ratio(returns, required_return=0, period=DAILY):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized Sortino ratio.

    """

    return empyrical.sortino_ratio(returns, required_return=required_return)


@deprecated(msg=DEPRECATION_WARNING)
def downside_risk(returns, required_return=0, period=DAILY):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized downside deviation

    """

    return empyrical.downside_risk(returns,
                                   required_return=required_return,
                                   period=period)


@deprecated(msg=DEPRECATION_WARNING)
def sharpe_ratio(returns, risk_free=0, period=DAILY):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Sharpe ratio.
    np.nan
        If insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """

    return empyrical.sharpe_ratio(returns, risk_free=risk_free, period=period)


@deprecated(msg=DEPRECATION_WARNING)
def information_ratio(returns, factor_returns):
    """
    Determines the Information ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns: float / series
        Benchmark return to compare returns against.

    Returns
    -------
    float
        The information ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/information_ratio for more details.

    """

    return empyrical.information_ratio(returns, factor_returns)


@deprecated(msg=DEPRECATION_WARNING)
def alpha_beta(returns, factor_returns):
    """Calculates both alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
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

    return empyrical.alpha_beta(returns, factor_returns=factor_returns)


@deprecated(msg=DEPRECATION_WARNING)
def alpha(returns, factor_returns):
    """Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float
        Alpha.
    """

    return empyrical.alpha(returns, factor_returns=factor_returns)


@deprecated(msg=DEPRECATION_WARNING)
def beta(returns, factor_returns):
    """Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.

    Returns
    -------
    float
        Beta.
    """

    return empyrical.beta(returns, factor_returns)


@deprecated(msg=DEPRECATION_WARNING)
def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """

    return empyrical.stability_of_timeseries(returns)


@deprecated(msg=DEPRECATION_WARNING)
def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.

    Returns
    -------
    float
        tail ratio

    """

    return empyrical.tail_ratio(returns)


def common_sense_ratio(returns):
    """Common sense ratio is the multiplication of the tail ratio and the
    Gain-to-Pain-Ratio -- sum(profits) / sum(losses).

    See http://bit.ly/1ORzGBk for more information on motivation of
    this metric.


    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        common sense ratio

    """
    return empyrical.tail_ratio(returns) * \
        (1 + empyrical.annual_return(returns))


SIMPLE_STAT_FUNCS = [
    empyrical.cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    common_sense_ratio,
]

FACTOR_STAT_FUNCS = [
    information_ratio,
    alpha,
    beta,
]


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


@deprecated(msg=DEPRECATION_WARNING)
def cum_returns(returns, starting_value=0):
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

    return empyrical.cum_returns(returns, starting_value=starting_value)


@deprecated(msg=DEPRECATION_WARNING)
def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    return empyrical.aggregate_returns(returns, convert_to=convert_to)


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
            out.loc[end] = empyrical.beta(
                returns.loc[beg:end],
                factor_returns.loc[beg:end])

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


def perf_stats(returns, factor_returns=None):
    """Calculates various performance metrics of a strategy, for use in
    plotting.show_perf_stats.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series (optional)
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If None, do not compute alpha, beta, and information ratio.

    Returns
    -------
    pd.Series
        Performance metrics.

    """

    stats = pd.Series()

    for stat_func in SIMPLE_STAT_FUNCS:
        stats[stat_func.__name__] = stat_func(returns)

    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            stats[stat_func.__name__] = stat_func(returns,
                                                  factor_returns)

    return stats


def perf_stats_bootstrap(returns, factor_returns=None, return_stats=True):
    """Calculates various bootstrapped performance metrics of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series (optional)
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
        If None, do not compute alpha, beta, and information ratio.
    return_stats : boolean (optional)
        If True, returns a DataFrame of mean, median, 5 and 95 percentiles
        for each perf metric.
        If False, returns a DataFrame with the bootstrap samples for
        each perf metric.

    Returns
    -------
    pd.DataFrame
        if return_stats is True:
        - Distributional statistics of bootstrapped sampling
        distribution of performance metrics.
        if return_stats is False:
        - Bootstrap samples for each performance metric.
    """
    bootstrap_values = OrderedDict()

    for stat_func in SIMPLE_STAT_FUNCS:
        stat_name = stat_func.__name__
        bootstrap_values[stat_name] = calc_bootstrap(stat_func,
                                                     returns)

    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            stat_name = stat_func.__name__
            bootstrap_values[stat_name] = calc_bootstrap(
                stat_func,
                returns,
                factor_returns=factor_returns)

    bootstrap_values = pd.DataFrame(bootstrap_values)

    if return_stats:
        stats = bootstrap_values.apply(calc_distribution_stats)
        return stats.T[['mean', 'median', '5%', '95%']]
    else:
        return bootstrap_values


def calc_bootstrap(func, returns, *args, **kwargs):
    """Performs a bootstrap analysis on a user-defined function returning
    a summary statistic.

    Parameters
    ----------
    func : function
        Function that either takes a single array (commonly returns)
        or two arrays (commonly returns and factor returns) and
        returns a single value (commonly a summary
        statistic). Additional args and kwargs are passed as well.
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series (optional)
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    n_samples : int (optional)
        Number of bootstrap samples to draw. Default is 1000.
        Increasing this will lead to more stable / accurate estimates.

    Returns
    -------
    numpy.ndarray
        Bootstrapped sampling distribution of passed in func.
    """

    n_samples = kwargs.pop('n_samples', 1000)
    out = np.empty(n_samples)

    factor_returns = kwargs.pop('factor_returns', None)

    for i in range(n_samples):
        idx = np.random.randint(len(returns), size=len(returns))
        returns_i = returns.iloc[idx].reset_index(drop=True)
        if factor_returns is not None:
            factor_returns_i = factor_returns.iloc[idx].reset_index(drop=True)
            out[i] = func(returns_i, factor_returns_i,
                          *args, **kwargs)
        else:
            out[i] = func(returns_i,
                          *args, **kwargs)

    return out


def calc_distribution_stats(x):
    """Calculate various summary statistics of data.

    Parameters
    ----------
    x : numpy.ndarray or pandas.Series
        Array to compute summary statistics for.

    Returns
    -------
    pandas.Series
        Series containing mean, median, std, as well as 5, 25, 75 and
        95 percentiles of passed in values.

    """
    return pd.Series({'mean': np.mean(x),
                      'median': np.median(x),
                      'std': np.std(x),
                      '5%': np.percentile(x, 5),
                      '25%': np.percentile(x, 25),
                      '75%': np.percentile(x, 75),
                      '95%': np.percentile(x, 95),
                      'IQR': np.subtract.reduce(
                          np.percentile(x, [75, 25])),
                      })


def get_max_drawdown_underwater(underwater):
    """Determines peak, valley, and recovery dates given an 'underwater'
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

    valley = np.argmin(underwater)  # end of the period
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
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.

    Returns
    -------
    float
        Maximum drawdown.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    returns = returns.copy()
    df_cum = cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1
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
    underwater = df_cum / running_max - 1

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

    df_drawdowns['peak date'] = pd.to_datetime(df_drawdowns['peak date'])
    df_drawdowns['valley date'] = pd.to_datetime(df_drawdowns['valley date'])
    df_drawdowns['recovery date'] = pd.to_datetime(
        df_drawdowns['recovery date'])

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

    return returns.rolling(rolling_sharpe_window).mean() \
        / returns.rolling(rolling_sharpe_window).std() \
        * np.sqrt(APPROX_BDAYS_PER_YEAR)


def simulate_paths(is_returns, num_days,
                   starting_value=1, num_samples=1000, random_seed=None):
    """
    Gnerate alternate paths using available values from in-sample returns.

    Parameters
    ----------
    is_returns : pandas.core.frame.DataFrame
        Non-cumulative in-sample returns.
    num_days : int
        Number of days to project the probability cone forward.
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

    samples : numpy.ndarray

    """
    samples = np.empty((num_samples, num_days))
    seed = np.random.RandomState(seed=random_seed)
    for i in range(num_samples):
        samples[i, :] = is_returns.sample(num_days, replace=True,
                                          random_state=seed)

    return samples


def summarize_paths(samples, cone_std=(1., 1.5, 2.)):
    """
    Gnerate the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns.

    Parameters
    ----------
    samples : numpy.ndarray
        Alternative paths, or series of possible outcomes.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.

    Returns
    -------

    samples : pandas.core.frame.DataFrame

    """
    cum_samples = np.cumprod(1 + samples, axis=1) * 1.
    cum_mean = cum_samples.mean(axis=0)
    cum_std = cum_samples.std(axis=0)

    if isinstance(cone_std, (float, int)):
        cone_std = [cone_std]

    cone_bounds = pd.DataFrame(columns=pd.Float64Index([]))
    for num_std in cone_std:
        cone_bounds.loc[:, float(num_std)] = cum_mean + cum_std * num_std
        cone_bounds.loc[:, float(-num_std)] = cum_mean - cum_std * num_std

    return cone_bounds


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

    samples = simulate_paths(
        is_returns=is_returns,
        num_days=num_days,
        starting_value=starting_value,
        num_samples=num_samples,
        random_seed=random_seed
    )

    cone_bounds = summarize_paths(
        samples=samples,
        cone_std=cone_std
    )

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
