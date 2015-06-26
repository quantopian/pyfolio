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
    data_no_missing = data.dropna()
    with pm.Model() as model:
        sigma = pm.HalfCauchy('sigma', beta=1, testval=data_no_missing.values.std())
        nu = Exponential('nu_minus_two', 1./10., testval=.3)

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


def compute_bayes_cone(df_train, df_test, preds):
    x = np.arange(len(df_train) + len(df_test))
    x_train = x[:len(df_train)]
    x_test = x[(len(df_train)-1):]

    df_train_cum = cum_returns(df_train)
    offset = df_train_cum.iloc[-1] - 1
    df_test_cum = cum_returns(df_test)
    df_test_cum -= df_test_cum.iloc[0]

    df_test_cum += offset + 1

    cum_preds = +1, 0) + offset
    scoreatpercentile = lambda cum_preds, p: [stats.scoreatpercentile(c, p) for c in cum_preds]
    perc5 = scoreatpercentile(cum_preds, 5)
    perc25 = scoreatpercentile(cum_preds, 25)
    perc75 = scoreatpercentile(cum_preds, 75)
    perc95 = scoreatpercentile(cum_preds, 95)


def compute_consistency_score():
    q = [sp.stats.percentileofscore(cum_preds[i, :], df_test_cum[i]) for i in range(len(cum_preds))]
    # normalize to be from 100 (perfect median line) to 0 (completely outside of cone)
    return 100 - np.abs(50 - np.mean(q)) / .5

def plot_bayes_cone(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(x_train, df_train_cum, color='g', label='in-sample')
    ax.plot(x_test,  df_test_cum, color='r', label='out-of-sample')

    ax.plot(x_test, perc5, '0.5', alpha=.5);
    ax.plot(x_test, perc95, '0.5', alpha=.5);
    ax.plot(x_test, perc25, '0.7', alpha=.5);
    ax.plot(x_test, perc75, '0.7', alpha=.5);
    ax.fill_between(x_test, perc5, perc95, alpha=.3)
    ax.fill_between(x_test, perc25, perc75, alpha=.6)
