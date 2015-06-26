import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

import theano.tensor as T

import pymc3 as pm

from .timeseries import cum_returns

def model_returns_t_alpha_beta(data, bmark, samples=2000):
    if len(data) != len(bmark):
        # pad missing data
        data = pd.Series(data, index=bmark.index)

    data_no_missing = data.dropna()

    with pm.Model() as model:
        sigma = pm.HalfCauchy('sigma', beta=1, testval=data_no_missing.values.std())
        nu = pm.Exponential('nu_minus_two', 1./10., testval=.3)

        # alpha and beta
        beta_init, alpha_init = sp.stats.linregress(bmark.loc[data_no_missing.index],
                                                    data_no_missing)[:2]

        alpha_reg = pm.Normal('alpha', mu=0, sd=.1, testval=alpha_init)
        beta_reg = pm.Normal('beta', mu=0, sd=1, testval=beta_init)

        data_missing = pd.DataFrame()
        returns = pm.T('returns',
                       nu=nu+2,
                       mu=alpha_reg + beta_reg * bmark,
                       sd=sigma,
                       observed=data)
        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)

    return trace

def model_returns_normal(data, samples=500):
    with pm.Model() as model:
        mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
        sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
        returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
        ann_vol = pm.Deterministic('annual volatility', returns.distribution.variance**.5 * np.sqrt(252))
        sharpe = pm.Deterministic('sharpe',
                                  returns.distribution.mean / returns.distribution.variance**.5 * np.sqrt(252))


        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)
    return trace

def model_returns_t(data, samples=500):
    with pm.Model() as model:
        mu = pm.Normal('mean returns', mu=0, sd=.01, testval=data.mean())
        sigma = pm.HalfCauchy('volatility', beta=1, testval=data.std())
        nu = pm.Exponential('nu_minus_two', 1./10., testval=3.)

        returns = pm.T('returns', nu=nu+2, mu=mu, sd=sigma, observed=data)
        ann_vol = pm.Deterministic('annual volatility',
                                   returns.distribution.variance**.5 * np.sqrt(252))

        sharpe = pm.Deterministic('sharpe',
                                  returns.distribution.mean / returns.distribution.variance**.5 * np.sqrt(252))

        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(samples, step, start=start)
    return trace


def compute_bayes_cone(preds):
    cum_preds = np.cumprod(preds + 1, 1)
    scoreatpercentile = lambda cum_preds, p: [stats.scoreatpercentile(c, p) for c in cum_preds.T]
    perc = {p: scoreatpercentile(cum_preds, p) for p in (5, 25, 75, 95)}

    return perc

def compute_consistency_score(df_test, preds):
    df_test_cum = cum_returns(df_test)
    cum_preds = np.cumprod(preds + 1, 1)

    q = [sp.stats.percentileofscore(cum_preds[i, :], df_test_cum.iloc[i]) for i in range(len(df_test_cum))]
    # normalize to be from 100 (perfect median line) to 0 (completely outside of cone)
    return 100 - np.abs(50 - np.mean(q)) / .5

def _plot_bayes_cone(df_train, df_test, preds, ax=None):
    if ax is None:
        ax = plt.gca()

    df_train_cum = cum_returns(df_train)
    df_test_cum = cum_returns(df_test)
    index = np.concatenate([df_train.index, df_test.index])
    offset = df_train_cum.iloc[-1] - df_test_cum.iloc[0]

    perc = compute_bayes_cone(preds)
    # Add indices
    perc = {k: pd.Series(v, index=df_test.index) for k, v in perc.iteritems()}

    df_test_cum_rel = df_test_cum + offset
    # Stitch together train and test
    df_train_cum.loc[df_test_cum_rel.index[0]] = df_test_cum_rel.iloc[0]

    # Plotting
    df_train_cum.plot(ax=ax, color='g', label='in-sample')
    df_test_cum_rel.plot(ax=ax, color='r', label='out-of-sample')

    ax.fill_between(df_test.index, perc[5] + offset, perc[95] + offset, alpha=.3)
    ax.fill_between(df_test.index, perc[25] + offset, perc[75] + offset, alpha=.6)

    return ax

def plot_bayes_cone(df_train, df_test, bmark, plot_train_len=50, ax=None):
    # generate cone
    trace = model_returns_t_alpha_beta(df_train, bmark)
    score = compute_consistency_score(df_test, trace['returns_missing'])
    ax = _plot_bayes_cone(df_train.iloc[-plot_train_len:], df_test,
                     trace['returns_missing'], ax=ax)
    ax.text(0.20, 0.95, 'Consistency score: %.1f' % score, fontsize=15,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,)

    return ax
