#
# Copyright 2017 Quantopian, Inc.
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

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt

import pymc3 as pm

from . import _seaborn as sns
from empyrical import cum_returns


def model_returns_t_alpha_beta(data, bmark, samples=2000, progressbar=True):
    """
    Run Bayesian alpha-beta-model with T distributed returns.

    This model estimates intercept (alpha) and slope (beta) of two
    return sets. Usually, these will be algorithm returns and
    benchmark returns (e.g. S&P500). The data is assumed to be T
    distributed and thus is robust to outliers and takes tail events
    into account.  If a pandas.DataFrame is passed as a benchmark, then
    multiple linear regression is used to estimate alpha and beta.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    bmark : pandas.DataFrame
        DataFrame of benchmark returns (e.g., S&P500) or risk factors (e.g.,
        Fama-French SMB, HML, and UMD).
        If bmark has more recent returns than returns_train, these dates
        will be treated as missing values and predictions will be
        generated for them taking market correlations into account.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    model : pymc.Model object
        PyMC3 model containing all random variables.
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    data_bmark = pd.concat([data, bmark], axis=1).dropna()

    with pm.Model() as model:
        sigma = pm.HalfCauchy(
            'sigma',
            beta=1)
        nu = pm.Exponential('nu_minus_two', 1. / 10.)

        # alpha and beta
        X = data_bmark.iloc[:, 1]
        y = data_bmark.iloc[:, 0]

        alpha_reg = pm.Normal('alpha', mu=0, sd=.1)
        beta_reg = pm.Normal('beta', mu=0, sd=1)

        mu_reg = alpha_reg + beta_reg * X
        pm.StudentT('returns',
                    nu=nu + 2,
                    mu=mu_reg,
                    sd=sigma,
                    observed=y)
        trace = pm.sample(samples, progressbar=progressbar)

    return model, trace


def model_returns_normal(data, samples=500, progressbar=True):
    """
    Run Bayesian model assuming returns are normally distributed.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    samples : int (optional)
        Number of posterior samples to draw.

    Returns
    -------
    model : pymc.Model object
        PyMC3 model containing all random variables.
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    with pm.Model() as model:
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

        trace = pm.sample(samples, progressbar=progressbar)
    return model, trace


def model_returns_t(data, samples=500, progressbar=True):
    """
    Run Bayesian model assuming returns are Student-T distributed.

    Compared with the normal model, this model assumes returns are
    T-distributed and thus have a 3rd parameter (nu) that controls the
    mass in the tails.

    Parameters
    ----------
    returns : pandas.Series
        Series of simple returns of an algorithm or stock.
    samples : int, optional
        Number of posterior samples to draw.

    Returns
    -------
    model : pymc.Model object
        PyMC3 model containing all random variables.
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    with pm.Model() as model:
        mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
        sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
        nu = pm.Exponential('nu_minus_two', 1. / 10., testval=3.)

        returns = pm.StudentT('returns', nu=nu + 2, mu=mu, sd=sigma,
                              observed=data)
        pm.Deterministic('annual volatility',
                         returns.distribution.variance**.5 * np.sqrt(252))

        pm.Deterministic('sharpe', returns.distribution.mean /
                         returns.distribution.variance**.5 *
                         np.sqrt(252))

        trace = pm.sample(samples, progressbar=progressbar)
    return model, trace


def model_best(y1, y2, samples=1000, progressbar=True):
    """
    Bayesian Estimation Supersedes the T-Test

    This model runs a Bayesian hypothesis comparing if y1 and y2 come
    from the same distribution. Returns are assumed to be T-distributed.

    In addition, computes annual volatility and Sharpe of in and
    out-of-sample periods.

    This model replicates the example used in:
    Kruschke, John. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.

    Parameters
    ----------
    y1 : array-like
        Array of returns (e.g. in-sample)
    y2 : array-like
        Array of returns (e.g. out-of-sample)
    samples : int, optional
        Number of posterior samples to draw.

    Returns
    -------
    model : pymc.Model object
        PyMC3 model containing all random variables.
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.

    See Also
    --------
    plot_stoch_vol : plotting of tochastic volatility model
    """

    y = np.concatenate((y1, y2))

    mu_m = np.mean(y)
    mu_p = 0.000001 * 1 / np.std(y)**2

    sigma_low = np.std(y) / 1000
    sigma_high = np.std(y) * 1000
    with pm.Model() as model:
        group1_mean = pm.Normal('group1_mean', mu=mu_m, tau=mu_p,
                                testval=y1.mean())
        group2_mean = pm.Normal('group2_mean', mu=mu_m, tau=mu_p,
                                testval=y2.mean())
        group1_std = pm.Uniform('group1_std', lower=sigma_low,
                                upper=sigma_high, testval=y1.std())
        group2_std = pm.Uniform('group2_std', lower=sigma_low,
                                upper=sigma_high, testval=y2.std())
        nu = pm.Exponential('nu_minus_two', 1 / 29., testval=4.) + 2.

        returns_group1 = pm.StudentT('group1', nu=nu, mu=group1_mean,
                                     lam=group1_std**-2, observed=y1)
        returns_group2 = pm.StudentT('group2', nu=nu, mu=group2_mean,
                                     lam=group2_std**-2, observed=y2)

        diff_of_means = pm.Deterministic('difference of means',
                                         group2_mean - group1_mean)
        pm.Deterministic('difference of stds',
                         group2_std - group1_std)
        pm.Deterministic('effect size', diff_of_means /
                         pm.math.sqrt((group1_std**2 +
                                       group2_std**2) / 2))

        pm.Deterministic('group1_annual_volatility',
                         returns_group1.distribution.variance**.5 *
                         np.sqrt(252))
        pm.Deterministic('group2_annual_volatility',
                         returns_group2.distribution.variance**.5 *
                         np.sqrt(252))

        pm.Deterministic('group1_sharpe', returns_group1.distribution.mean /
                         returns_group1.distribution.variance**.5 *
                         np.sqrt(252))
        pm.Deterministic('group2_sharpe', returns_group2.distribution.mean /
                         returns_group2.distribution.variance**.5 *
                         np.sqrt(252))

        trace = pm.sample(samples, progressbar=progressbar)
    return model, trace


def plot_best(trace=None, data_train=None, data_test=None,
              samples=1000, burn=200, axs=None):
    """
    Plot BEST significance analysis.

    Parameters
    ----------
    trace : pymc3.sampling.BaseTrace, optional
        trace object as returned by model_best()
        If not passed, will run model_best(), for which
        data_train and data_test are required.
    data_train : pandas.Series, optional
        Returns of in-sample period.
        Required if trace=None.
    data_test : pandas.Series, optional
        Returns of out-of-sample period.
        Required if trace=None.
    samples : int, optional
        Posterior samples to draw.
    burn : int
        Posterior sampels to discard as burn-in.
    axs : array of matplotlib.axes objects, optional
        Plot into passed axes objects. Needs 6 axes.

    Returns
    -------
    None

    See Also
    --------
    model_best : Estimation of BEST model.
    """

    if trace is None:
        if (data_train is not None) or (data_test is not None):
            raise ValueError('Either pass trace or data_train and data_test')
        trace = model_best(data_train, data_test, samples=samples)

    trace = trace[burn:]
    if axs is None:
        fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 4))

    def distplot_w_perc(trace, ax):
        sns.distplot(trace, ax=ax)
        ax.axvline(
            stats.scoreatpercentile(trace, 2.5),
            color='0.5', label='2.5 and 97.5 percentiles')
        ax.axvline(
            stats.scoreatpercentile(trace, 97.5),
            color='0.5')

    sns.distplot(trace['group1_mean'], ax=axs[0], label='Backtest')
    sns.distplot(trace['group2_mean'], ax=axs[0], label='Forward')
    axs[0].legend(loc=0, frameon=True, framealpha=0.5)
    axs[1].legend(loc=0, frameon=True, framealpha=0.5)

    distplot_w_perc(trace['difference of means'], axs[1])

    axs[0].set(xlabel='Mean', ylabel='Belief', yticklabels=[])
    axs[1].set(xlabel='Difference of means', yticklabels=[])

    sns.distplot(trace['group1_annual_volatility'], ax=axs[2],
                 label='Backtest')
    sns.distplot(trace['group2_annual_volatility'], ax=axs[2],
                 label='Forward')
    distplot_w_perc(trace['group2_annual_volatility'] -
                    trace['group1_annual_volatility'], axs[3])
    axs[2].set(xlabel='Annual volatility', ylabel='Belief',
               yticklabels=[])
    axs[2].legend(loc=0, frameon=True, framealpha=0.5)
    axs[3].set(xlabel='Difference of volatility', yticklabels=[])

    sns.distplot(trace['group1_sharpe'], ax=axs[4], label='Backtest')
    sns.distplot(trace['group2_sharpe'], ax=axs[4], label='Forward')
    distplot_w_perc(trace['group2_sharpe'] - trace['group1_sharpe'],
                    axs[5])
    axs[4].set(xlabel='Sharpe', ylabel='Belief', yticklabels=[])
    axs[4].legend(loc=0, frameon=True, framealpha=0.5)
    axs[5].set(xlabel='Difference of Sharpes', yticklabels=[])

    sns.distplot(trace['effect size'], ax=axs[6])
    axs[6].axvline(
        stats.scoreatpercentile(trace['effect size'], 2.5),
        color='0.5')
    axs[6].axvline(
        stats.scoreatpercentile(trace['effect size'], 97.5),
        color='0.5')
    axs[6].set(xlabel='Difference of means normalized by volatility',
               ylabel='Belief', yticklabels=[])


def model_stoch_vol(data, samples=2000, progressbar=True):
    """
    Run stochastic volatility model.

    This model estimates the volatility of a returns series over time.
    Returns are assumed to be T-distributed. lambda (width of
    T-distributed) is assumed to follow a random-walk.

    Parameters
    ----------
    data : pandas.Series
        Return series to model.
    samples : int, optional
        Posterior samples to draw.

    Returns
    -------
    model : pymc.Model object
        PyMC3 model containing all random variables.
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.

    See Also
    --------
    plot_stoch_vol : plotting of tochastic volatility model
    """

    from pymc3.distributions.timeseries import GaussianRandomWalk

    with pm.Model() as model:
        nu = pm.Exponential('nu', 1. / 10, testval=5.)
        sigma = pm.Exponential('sigma', 1. / .02, testval=.1)
        s = GaussianRandomWalk('s', sigma**-2, shape=len(data))
        volatility_process = pm.Deterministic('volatility_process',
                                              pm.math.exp(-2 * s))
        pm.StudentT('r', nu, lam=volatility_process, observed=data)

        trace = pm.sample(samples, progressbar=progressbar)

    return model, trace


def plot_stoch_vol(data, trace=None, ax=None):
    """
    Generate plot for stochastic volatility model.

    Parameters
    ----------
    data : pandas.Series
        Returns to model.
    trace : pymc3.sampling.BaseTrace object, optional
        trace as returned by model_stoch_vol
        If not passed, sample from model.
    ax : matplotlib.axes object, optional
        Plot into axes object

    Returns
    -------
    ax object

    See Also
    --------
    model_stoch_vol : run stochastic volatility model
    """

    if trace is None:
        trace = model_stoch_vol(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))

    data.abs().plot(ax=ax)
    ax.plot(data.index, np.exp(trace['s', ::30].T), 'r', alpha=.03)
    ax.set(title='Stochastic volatility', xlabel='Time', ylabel='Volatility')
    ax.legend(['Abs returns', 'Stochastic volatility process'],
              frameon=True, framealpha=0.5)

    return ax


def compute_bayes_cone(preds, starting_value=1.):
    """
    Compute 5, 25, 75 and 95 percentiles of cumulative returns, used
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

    def scoreatpercentile(cum_preds, p):
        return [stats.scoreatpercentile(
            c, p) for c in cum_preds.T]

    cum_preds = np.cumprod(preds + 1, 1) * starting_value
    perc = {p: scoreatpercentile(cum_preds, p) for p in (5, 25, 75, 95)}

    return perc


def compute_consistency_score(returns_test, preds):
    """
    Compute Bayesian consistency score.

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

    returns_train_cum.plot(ax=ax, color='g', label='In-sample')
    returns_test_cum_rel.plot(ax=ax, color='r', label='Out-of-sample')

    ax.fill_between(returns_test.index, perc[5], perc[95], alpha=.3)
    ax.fill_between(returns_test.index, perc[25], perc[75], alpha=.6)
    ax.legend(loc='best', frameon=True, framealpha=0.5)
    ax.set_title('Bayesian cone')
    ax.set_xlabel('')
    ax.set_ylabel('Cumulative returns')

    return ax


def run_model(model, returns_train, returns_test=None,
              bmark=None, samples=500, ppc=False, progressbar=True):
    """
    Run one of the Bayesian models.

    Parameters
    ----------
    model : {'alpha_beta', 't', 'normal', 'best'}
        Which model to run
    returns_train : pd.Series
        Timeseries of simple returns
    returns_test : pd.Series (optional)
        Out-of-sample returns. Datetimes in returns_test will be added to
        returns_train as missing values and predictions will be generated
        for them.
    bmark : pd.Series or pd.DataFrame (optional)
        Only used for alpha_beta to estimate regression coefficients.
        If bmark has more recent returns than returns_train, these dates
        will be treated as missing values and predictions will be
        generated for them taking market correlations into account.
    samples : int (optional)
        Number of posterior samples to draw.
    ppc : boolean (optional)
        Whether to run a posterior predictive check. Will generate
        samples of length returns_test.  Returns a second argument
        that contains the PPC of shape samples x len(returns_test).

    Returns
    -------
    trace : pymc3.sampling.BaseTrace object
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.

    ppc : numpy.array (if ppc==True)
       PPC of shape samples x len(returns_test).
    """

    if model == 'alpha_beta':
        model, trace = model_returns_t_alpha_beta(returns_train,
                                                  bmark, samples,
                                                  progressbar=progressbar)
    elif model == 't':
        model, trace = model_returns_t(returns_train, samples,
                                       progressbar=progressbar)
    elif model == 'normal':
        model, trace = model_returns_normal(returns_train, samples,
                                            progressbar=progressbar)
    elif model == 'best':
        model, trace = model_best(returns_train, returns_test,
                                  samples=samples,
                                  progressbar=progressbar)
    else:
        raise NotImplementedError(
            'Model {} not found.'
            'Use alpha_beta, t, normal, or best.'.format(model))

    if ppc:
        ppc_samples = pm.sample_ppc(trace, samples=samples,
                                    model=model, size=len(returns_test),
                                    progressbar=progressbar)
        return trace, ppc_samples['returns']

    return trace


def plot_bayes_cone(returns_train, returns_test, ppc,
                    plot_train_len=50, ax=None):
    """
    Generate cumulative returns plot with Bayesian cone.

    Parameters
    ----------
    returns_train : pd.Series
        Timeseries of simple returns
    returns_test : pd.Series
        Out-of-sample returns. Datetimes in returns_test will be added to
        returns_train as missing values and predictions will be generated
        for them.
    ppc : np.array
        Posterior predictive samples of shape samples x
        len(returns_test).
    plot_train_len : int (optional)
        How many data points to plot of returns_train. Useful to zoom in on
        the prediction if there is a long backtest period.
    ax : matplotlib.Axis (optional)
        Axes upon which to plot.

    Returns
    -------
    score : float
        Consistency score (see compute_consistency_score)
    trace : pymc3.sampling.BaseTrace
        A PyMC3 trace object that contains samples for each parameter
        of the posterior.
    """

    score = compute_consistency_score(returns_test,
                                      ppc)

    ax = _plot_bayes_cone(
        returns_train,
        returns_test,
        ppc,
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

    ax.set_ylabel('Cumulative returns')
    return score
