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
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt

import pymc3 as pm

from .timeseries import cum_returns


def model_returns_t_alpha_beta(data, bmark, samples=2000):
    """Run Bayesian alpha-beta-model with T distributed returns.

    This model estimates intercept (alpha) and slope (beta) of two
    return sets. Usually, these will be algorithm returns and
    benchmark returns (e.g. S&P500). The data is assumed to be T
    distributed and thus is robust to outliers and takes tail events
    into account.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    bmark : pandas.Series
        Series of simple returns of a benchmark like the S&P500.
        If bmark has more recent returns than returns_train, these dates
        will be treated as missing values and predictions will be
        generated for them taking market correlations into account.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    if len(data) != len(bmark):
        # pad missing data
        data = pd.Series(data, index=bmark.index)

    data_no_missing = data.dropna()

    with pm.Model():
        sigma = pm.HalfCauchy(
            'sigma',
            beta=1,
            testval=data_no_missing.values.std())
        nu = pm.Exponential('nu_minus_two', 1. / 10., testval=.3)

        # alpha and beta
        beta_init, alpha_init = sp.stats.linregress(
            bmark.loc[data_no_missing.index],
            data_no_missing)[:2]

        alpha_reg = pm.Normal('alpha', mu=0, sd=.1, testval=alpha_init)
        beta_reg = pm.Normal('beta', mu=0, sd=1, testval=beta_init)

        pm.T('returns',
             nu=nu + 2,
             mu=alpha_reg + beta_reg * bmark,
             sd=sigma,
             observed=data)
        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)

    return trace


def model_returns_normal(data, samples=500):
    """Run Bayesian model assuming returns are Student-T distributed.

    Compared with the normal model, this model assumes returns be
    T-distributed and thus has a 3rd parameter (nu) that controls the
    mass in the tails.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.

    """
    with pm.Model():
        mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
        sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
        returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
        pm.Deterministic(
            'annual volatility',
            returns.distribution.variance**.5 *
            np.sqrt(252))
        pm.Deterministic(
            'sharpe',
            returns.distribution.mean /
            returns.distribution.variance**.5 *
            np.sqrt(252))

        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)
    return trace


def model_returns_t(data, samples=500):
    """Run Bayesian model assuming returns are normally distributed.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.

    """

    with pm.Model():
        mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
        sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
        nu = pm.Exponential('nu_minus_two', 1. / 10., testval=3.)

        returns = pm.T('returns', nu=nu + 2, mu=mu, sd=sigma, observed=data)
        pm.Deterministic('annual volatility',
                         returns.distribution.variance**.5 * np.sqrt(252))

        pm.Deterministic('sharpe', returns.distribution.mean /
                         returns.distribution.variance**.5 *
                         np.sqrt(252))

        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)
    return trace


def compute_bayes_cone(preds, starting_value=1.):
    """Compute 5, 25, 75 and 95 percentiles of cumulative returns, used
    for the Bayesian cone.

    Parameters
    ----------
    preds : numpy.array
        Multiple (simulated) cumulative returns.
    starting_value : int (optional)
        Have cumulative returns start around this value.
        Default = 1.

    Returns
    -------
    dict of percentiles over time
        Dictionary mapping percentiles (5, 25, 75, 95) to a
        timeseries.

    """

    cum_preds = np.cumprod(preds + 1, 1) * starting_value
    scoreatpercentile = lambda cum_preds, p: [
        stats.scoreatpercentile(
            c, p) for c in cum_preds.T]
    perc = {p: scoreatpercentile(cum_preds, p) for p in (5, 25, 75, 95)}

    return perc


def compute_consistency_score(returns_test, preds):
    """Compute Bayesian consistency score.

    Parameters
    ----------
    returns_test : pd.Series
        Observed cumulative returns.
    preds : numpy.array
        Multiple (simulated) cumulative returns.

    Returns
    -------
    Consistency score
        Score from 100 (returns_test perfectly on the median line of the
        Bayesian cone spanned by preds) to 0 (returns_test completely
        outside of Bayesian cone.)
    """
    returns_test_cum = cum_returns(returns_test, starting_value=1.)
    cum_preds = np.cumprod(preds + 1, 1)

    q = [sp.stats.percentileofscore(cum_preds[:, i],
                                    returns_test_cum.iloc[i],
                                    kind='weak')
         for i in range(len(returns_test_cum))]
    # normalize to be from 100 (perfect median line) to 0 (completely outside
    # of cone)
    return 100 - np.abs(50 - np.mean(q)) / .5


def _plot_bayes_cone(returns_train, returns_test,
                     preds, plot_train_len=None, ax=None):
    if ax is None:
        ax = plt.gca()

    returns_train_cum = cum_returns(returns_train, starting_value=1.)
    returns_test_cum = cum_returns(returns_test,
                                   starting_value=returns_train_cum.iloc[-1])

    perc = compute_bayes_cone(preds, starting_value=returns_train_cum.iloc[-1])
    # Add indices
    perc = {k: pd.Series(v, index=returns_test.index) for k, v in perc.items()}

    returns_test_cum_rel = returns_test_cum
    # Stitch together train and test
    returns_train_cum.loc[returns_test_cum_rel.index[0]] = \
        returns_test_cum_rel.iloc[0]

    # Plotting
    if plot_train_len is not None:
        returns_train_cum = returns_train_cum.iloc[-plot_train_len:]

    returns_train_cum.plot(ax=ax, color='g', label='in-sample')
    returns_test_cum_rel.plot(ax=ax, color='r', label='out-of-sample')

    ax.fill_between(returns_test.index, perc[5], perc[95], alpha=.3)
    ax.fill_between(returns_test.index, perc[25], perc[75], alpha=.6)
    ax.legend(loc='best')
    ax.set_title('Bayesian Cone')
    ax.set_xlabel('')
    ax.set_ylabel('Cumulative Returns')

    return ax


def run_model(model, returns_train, returns_test=None,
              bmark=None, samples=500):
    """Run one of the Bayesian models.

    Parameters
    ----------
    model : {'alpha_beta', 't', 'normal'}
        Which model to run
    returns_train : pd.Series
        Timeseries of simple returns
    returns_test : pd.Series (optional)
        Out-of-sample returns. Datetimes in returns_test will be added to
        returns_train as missing values and predictions will be generated
        for them.
    bmark : pd.Series (optional)
        Only used for alpha_beta to estimate regression coefficients.
        If bmark has more recent returns than returns_train, these dates
        will be treated as missing values and predictions will be
        generated for them taking market correlations into account.

    Returns
    -------
    pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """
    if returns_test is not None:
        period = returns_train.index.append(returns_test.index)
        rets = pd.Series(returns_train, period)
    else:
        rets = returns_train

    if model == 'alpha_beta':
        trace = model_returns_t_alpha_beta(returns_train, bmark, samples)
    elif model == 't':
        trace = model_returns_t(rets, samples)
    elif model == 'normal':
        trace = model_returns_normal(rets, samples)

    return trace


def plot_bayes_cone(returns_train, returns_test, bmark=None, model='t',
                    trace=None, plot_train_len=50, ax=None,
                    samples=500):
    """Generate cumulative returns plot with Bayesian cone.

    Parameters
    ----------
    returns_train : pd.Series
        Timeseries of simple returns
    returns_test : pd.Series
        Out-of-sample returns. Datetimes in returns_test will be added to
        returns_train as missing values and predictions will be generated
        for them.
    bmark : pd.Series (optional)
        Only used for alpha_beta to estimate regression coefficients.
        If bmark has more recent returns than returns_train, these dates
        will be treated as missing values and predictions will be
        generated for them taking market correlations into account.
    model : {None, 'alpha_beta', 't', 'normal'} (optional)
        Which model to run. If none, assume trace is being passed in.
    trace : pymc3.sampling.BaseTrace (optional)
        Trace of a previously run model.
    plot_train_len : int (optional)
        How many data points to plot of returns_train. Useful to zoom in on
        the prediction if there is a long backtest period.
    ax : matplotlib.Axis (optional)
        Axes upon which to plot.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    score : float
        Consistency score (see compute_consistency_score)
    trace : pymc3.sampling.BaseTrace
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    # generate cone
    if trace is None:
        trace = run_model(model, returns_train, returns_test=returns_test,
                          bmark=bmark, samples=samples)

    score = compute_consistency_score(returns_test, trace['returns_missing'])

    ax = _plot_bayes_cone(
        returns_train,
        returns_test,
        trace['returns_missing'],
        plot_train_len=plot_train_len,
        ax=ax)
    ax.text(
        0.40,
        0.90,
        'Consistency score: %.1f' %
        score,
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes,
    )

    ax.set_ylabel('Cumulative returns', fontsize=14)
    return score, trace
