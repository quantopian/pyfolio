from __future__ import division
from collections import OrderedDict

from operator import *
from collections import *

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
from sklearn import preprocessing

import statsmodels.api as sm

import datetime


def var_cov_var_normal(P, c, mu=0, sigma=1, **kwargs):
    """
    Variance-Covariance calculation of daily Value-at-Risk
    using confidence level c, with mean of returns mu
    and standard deviation of returns sigma, on a portfolio
    of value P.
    """
    alpha = sp.stats.norm.ppf(1 - c, mu, sigma)
    return P - P * (alpha + 1)


def normalize(df_rets, starting_value=1):
    if starting_value > 1:
        return starting_value * (df_rets / df_rets.iloc[0])
    else:
        return df_rets / df_rets.iloc[0]


def cum_returns(df_rets, starting_value=1.):
    """Compute cumulative returns from simple returns

    Parameters
    ----------
    df_rets : pandas.Series
        Series of simple returns. First element may be nan
        and will be filled with 0. to have returned Series
        start in origin.
    starting_value : float, optional
        Have cumulative returns start around this value.
        Default = 1.

    Returns
    -------
    pandas.Series
        Series of cumulative returns.

    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying

    """
    # df_price.pct_change() adds a nan in first position, we can use
    # that to have cum_returns start at the origin so that
    # df_cum.iloc[0] == starting_value
    # Note that we can't add that ourselves as we don't know which dt
    # to use.
    if pd.isnull(df_rets.iloc[0]):
        df_rets.iloc[0] = 0.

    df_cum = np.exp(np.log(1 + df_rets).cumsum())

    return df_cum * starting_value


def aggregate_returns(df_daily_rets, convert_to):
    cumulate_returns = lambda x: cum_returns(x)[-1]
    if convert_to == 'weekly':
        return df_daily_rets.groupby(
            [lambda x: x.year, lambda x: x.month, lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == 'monthly':
        return df_daily_rets.groupby(
            [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == 'yearly':
        return df_daily_rets.groupby(
            [lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be daily, weekly, monthly or yearly')

# Strategy Performance statistics & timeseries analysis functions


def max_drawdown(df_rets, inputIsNAV=True):
    if df_rets.size < 1:
        return np.nan

    if inputIsNAV:
        temp_ts = df_rets
    else:
        temp_ts = cum_returns(df_rets, starting_value=100)

    MDD = 0
    DD = 0
    peak = -99999
    for value in temp_ts:
        if (value > peak):
            peak = value
        else:
            DD = (peak - value) / peak
        if (DD > MDD):
            MDD = DD
    return -1 * MDD


def annual_return(df_rets, inputIsNAV=True, style='calendar'):
    # if style == 'compound' then return will be calculated in geometric terms: (1+mean(all_daily_returns))^252 - 1
    # if style == 'calendar' then return will be calculated as ((last_value - start_value)/start_value)/num_of_years
    # if style == 'arithmetic' then return is simply
    # mean(all_daily_returns)*252
    if df_rets.size < 1:
        return np.nan

    if inputIsNAV:
        tempReturns = df_rets.pct_change().dropna()
        if style == 'calendar':
            num_years = len(tempReturns) / 252
            start_value = df_rets[0]
            end_value = df_rets[-1]
            return ((end_value - start_value) / start_value) / num_years
        if style == 'compound':
            return pow((1 + tempReturns.mean()), 252) - 1
        else:
            return tempReturns.mean() * 252
    else:
        if style == 'calendar':
            num_years = len(df_rets) / 252
            temp_NAV = cum_returns(df_rets, starting_value=100)
            start_value = temp_NAV[0]
            end_value = temp_NAV[-1]
            return ((end_value - start_value) / start_value) / num_years
        if style == 'compound':
            return pow((1 + df_rets.mean()), 252) - 1
        else:
            return df_rets.mean() * 252


def annual_volatility(df_rets, inputIsNAV=True):
    if df_rets.size < 2:
        return np.nan
    if inputIsNAV:
        tempReturns = df_rets.pct_change().dropna()
        return tempReturns.std() * np.sqrt(252)
    else:
        return df_rets.std() * np.sqrt(252)


def calmer_ratio(df_rets, inputIsNAV=True, returns_style='calendar'):
    temp_max_dd = max_drawdown(df_rets=df_rets, inputIsNAV=inputIsNAV)
    # print(temp_max_dd)
    if temp_max_dd < 0:
        if inputIsNAV:
            temp = annual_return(df_rets=df_rets,
                                 inputIsNAV=True,
                                 style=returns_style) / abs(max_drawdown(df_rets=df_rets,
                                                                         inputIsNAV=True))
        else:
            tempNAV = cum_returns(df_rets, starting_value=100)
            temp = annual_return(df_rets=tempNAV,
                                 inputIsNAV=True,
                                 style=returns_style) / abs(max_drawdown(df_rets=tempNAV,
                                                                         inputIsNAV=True))
        # print(temp)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan
    else:
        return temp


def sharpe_ratio(df_rets, inputIsNAV=True, returns_style='calendar'):
    return annual_return(df_rets,
                         inputIsNAV=inputIsNAV,
                         style=returns_style) / annual_volatility(df_rets,
                                                                  inputIsNAV=inputIsNAV)


def stability_of_timeseries(df_rets, logValue=True, inputIsNAV=True):
    if df_rets.size < 2:
        return np.nan

    if logValue:
        if inputIsNAV:
            tempValues = np.log10(df_rets.values)
            leng_df_rets = df_rets.size
        else:
            temp_ts = cum_returns(df_rets, starting_value=100)
            tempValues = np.log10(temp_ts.values)
            leng_df_rets = temp_ts.size
    else:
        if inputIsNAV:
            tempValues = df_rets.values
            leng_df_rets = df_rets.size
        else:
            temp_ts = cum_returns(df_rets, starting_value=100)
            tempValues = temp_ts.values
            leng_df_rets = temp_ts.size

    X = range(0, leng_df_rets)
    X = sm.add_constant(X)

    model = sm.OLS(tempValues, X).fit()

    return model.rsquared

def out_of_sample_vs_in_sample_returns_kde(bt_ts, oos_ts, transform_style='scale', return_zero_if_exception=True):
    
    bt_ts_pct = bt_ts.pct_change().dropna()
    oos_ts_pct = oos_ts.pct_change().dropna()
    
    bt_ts_r = bt_ts_pct.reshape(len(bt_ts_pct),1)
    oos_ts_r = oos_ts_pct.reshape(len(oos_ts_pct),1)
    
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
        scipy_kde_train = stats.gaussian_kde(X_train, bw_method=kernal_method)(x_axis_dim)
        scipy_kde_test = stats.gaussian_kde(X_test, bw_method=kernal_method)(x_axis_dim)
    except:
        if return_zero_if_exception:
            return 0.0
        else:
            return np.nan
    
    kde_diff = sum(abs(scipy_kde_test - scipy_kde_train)) / (sum(scipy_kde_train) + sum(scipy_kde_test))

    return kde_diff

def calc_multifactor(df_rets, factors):
    import statsmodels.api as sm
    factors = factors.loc[df_rets.index]
    factors = sm.add_constant(factors)
    factors = factors.dropna(axis=0)
    results = sm.OLS(df_rets[factors.index], factors).fit()

    return results.params


def rolling_multifactor_beta(ser, multi_factor_df, rolling_window=63):
    results = [calc_multifactor(ser[beg:end], multi_factor_df) for beg, end in zip(
        ser.index[0:-rolling_window], ser.index[rolling_window:])]

    return pd.DataFrame(index=ser.index[rolling_window:], data=results)


def multi_factor_alpha(
        factors_ts_list,
        single_ts,
        factor_names_list,
        input_is_returns=False,
        annualized=False,
        annualize_factor=252,
        show_output=False):

    factors_ts = [i.asfreq(freq='D', normalize=True) for i in factors_ts_list]
    dep_var = single_ts.asfreq(freq='D', normalize=True)

    if not input_is_returns:
        factors_ts = [i.pct_change().dropna() for i in factors_ts]
        dep_var = dep_var.pct_change().dropna()

    factors_align = pd.DataFrame(factors_ts).T.dropna()
    factors_align.columns = factor_names_list

    if show_output:
        print factors_align.head(5)
        print dep_var.head(5)

    if dep_var.shape[0] < 2:
        return np.nan
    if factors_align.shape[0] < 2:
        return np.nan

    factor_regress = pd.ols(y=dep_var, x=factors_align, intercept=True)

    factor_alpha = factor_regress.summary_as_matrix.intercept.beta

    if show_output:
        print factor_regress.resid
        print factor_regress.summary_as_matrix

    if annualized:
        return factor_alpha * annualize_factor
    else:
        return factor_alpha


def calc_alpha_beta(df_rets, benchmark_rets):
    ret_index = df_rets.index
    beta, alpha = sp.stats.linregress(benchmark_rets.loc[ret_index].values,
                                      df_rets.values)[:2]

    return alpha * 252, beta


def rolling_beta(ser, benchmark_rets, rolling_window=63):
    results = [calc_alpha_beta(ser[beg:end],
                               benchmark_rets)[1] for beg,
               end in zip(ser.index[0:-rolling_window],
                          ser.index[rolling_window:])]

    return pd.Series(index=ser.index[rolling_window:], data=results)


def perf_stats(
        df_rets,
        inputIsNAV=True,
        returns_style='compound',
        return_as_dict=False):
    all_stats = {}
    all_stats['annual_return'] = annual_return(
        df_rets,
        inputIsNAV=inputIsNAV,
        style=returns_style)
    all_stats['annual_volatility'] = annual_volatility(
        df_rets,
        inputIsNAV=inputIsNAV)
    all_stats['sharpe_ratio'] = sharpe_ratio(
        df_rets,
        inputIsNAV=inputIsNAV,
        returns_style=returns_style)
    all_stats['calmar_ratio'] = calmer_ratio(
        df_rets,
        inputIsNAV=inputIsNAV,
        returns_style=returns_style)
    all_stats['stability'] = stability_of_timeseries(
        df_rets, inputIsNAV=inputIsNAV)
    all_stats['max_drawdown'] = max_drawdown(df_rets, inputIsNAV=inputIsNAV)

    if return_as_dict:
        return all_stats
    else:
        all_stats_df = pd.DataFrame(
            index=all_stats.keys(),
            data=all_stats.values())
        all_stats_df.columns = ['perf_stats']
        return all_stats_df


def get_max_draw_down_underwater(underwater):
    valley = np.argmax(underwater)  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery


def get_max_draw_down(df_rets):
    df_rets = df_rets.copy()
    df_cum = cum_returns(df_rets, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = (running_max - df_cum) / running_max
    return get_max_draw_down_underwater(underwater)


def get_top_draw_downs(df_rets, top=10):
    df_rets = df_rets.copy()
    df_cum = cum_returns(df_rets, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_draw_down_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater = pd.concat(
                [underwater.loc[:peak].iloc[:-1], underwater.loc[recovery:].iloc[1:]])
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if len(df_rets) == 0:
            break

    return drawdowns


def gen_drawdown_table(df_rets, top=10):
    df_cum = cum_returns(df_rets, 1.0)
    drawdown_periods = get_top_draw_downs(df_rets, top=top)
    df_drawdowns = pd.DataFrame(index=range(top), columns=['net drawdown in %',
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


def rolling_sharpe(df_rets, rolling_sharpe_window):
    return pd.rolling_mean(df_rets, rolling_sharpe_window) \
        / pd.rolling_std(df_rets, rolling_sharpe_window) * np.sqrt(252)


def cone_rolling(
                input_rets, 
                num_stdev=1.0, 
                warm_up_days_pct=0.5, 
                std_scale_factor=252, 
                update_std_oos_rolling=False, 
                cone_fit_end_date=None, 
                extend_fit_trend=True, 
                create_future_cone=True):

    # if specifying 'cone_fit_end_date' please use a pandas compatible format, e.g. '2015-8-4', 'YYYY-MM-DD'

    warm_up_days = int(warm_up_days_pct*input_rets.size)

    # create initial linear fit from beginning of timeseries thru warm_up_days or the specified 'cone_fit_end_date'
    if cone_fit_end_date is None:
        df_rets = input_rets[:warm_up_days]
    else:
        df_rets = input_rets[ input_rets.index < cone_fit_end_date]
    
    perf_ts = cum_returns(df_rets, 1)
        
    X = range(0, perf_ts.size)
    X = sm.add_constant(X)
    sm.OLS(perf_ts , range(0,len(perf_ts)))
    line_ols = sm.OLS(perf_ts.values , X).fit()
    fit_line_ols_coef = line_ols.params[1]
    fit_line_ols_inter = line_ols.params[0]

    x_points = range(0, perf_ts.size)
    x_points = np.array(x_points) * fit_line_ols_coef + fit_line_ols_inter
    
    perf_ts_r = pd.DataFrame(perf_ts)
    perf_ts_r.columns = ['perf']
    
    warm_up_std_pct = np.std(perf_ts.pct_change().dropna())
    std_pct = warm_up_std_pct * np.sqrt(std_scale_factor) 

    perf_ts_r['line'] = x_points
    perf_ts_r['sd_up'] = perf_ts_r['line'] * ( 1 + num_stdev * std_pct )
    perf_ts_r['sd_down'] = perf_ts_r['line'] * ( 1 - num_stdev * std_pct )
    
    std_pct = warm_up_std_pct * np.sqrt(std_scale_factor) 
    
    last_backtest_day_index = df_rets.index[-1]
    cone_end_rets = input_rets[ input_rets.index > last_backtest_day_index ]
    new_cone_day_scale_factor = int(1)
    oos_intercept_shift = perf_ts_r.perf[-1] - perf_ts_r.line[-1]

    # make the cone for the out-of-sample/live papertrading period
    for i in cone_end_rets.index:
        df_rets = input_rets[:i]
        perf_ts = cum_returns(df_rets, 1)
        
        if extend_fit_trend:
            line_ols_coef = fit_line_ols_coef
            line_ols_inter = fit_line_ols_inter
        else:
            X = range(0, perf_ts.size)
            X = sm.add_constant(X)
            sm.OLS(perf_ts , range(0,len(perf_ts)))
            line_ols = sm.OLS(perf_ts.values , X).fit()
            line_ols_coef = line_ols.params[1]
            line_ols_inter = line_ols.params[0]
            
        x_points = range(0, perf_ts.size)
        x_points = np.array(x_points) * line_ols_coef + line_ols_inter + oos_intercept_shift
        
        temp_line = x_points   
        if update_std_oos_rolling:
            #std_pct = np.sqrt(std_scale_factor) * np.std(perf_ts.pct_change().dropna())
            std_pct = np.sqrt(new_cone_day_scale_factor) * np.std(perf_ts.pct_change().dropna())
        else:
            std_pct = np.sqrt(new_cone_day_scale_factor) * warm_up_std_pct
        
        temp_sd_up = temp_line * ( 1 + num_stdev * std_pct )
        temp_sd_down = temp_line * ( 1 - num_stdev * std_pct )
        
        new_daily_cone = pd.DataFrame(index=[i], data={'perf':perf_ts[i], 
                                                       'line':temp_line[-1], 
                                                       'sd_up':temp_sd_up[-1], 
                                                       'sd_down':temp_sd_down[-1] } )
        
        perf_ts_r = perf_ts_r.append(new_daily_cone)
        new_cone_day_scale_factor+=1


    if create_future_cone:
        extend_ahead_days = 252
        future_cone_dates = pd.date_range(cone_end_rets.index[-1], periods=extend_ahead_days, freq='B')
        
        future_cone_intercept_shift = perf_ts_r.perf[-1] - perf_ts_r.line[-1]
        
        future_days_scale_factor = np.linspace(1,extend_ahead_days,extend_ahead_days)
        std_pct = np.sqrt(future_days_scale_factor) * warm_up_std_pct
        
        x_points = range(perf_ts.size, perf_ts.size + extend_ahead_days)
        x_points = np.array(x_points) * line_ols_coef + line_ols_inter + oos_intercept_shift + future_cone_intercept_shift
        temp_line = x_points   
        temp_sd_up = temp_line * ( 1 + num_stdev * std_pct )
        temp_sd_down = temp_line * ( 1 - num_stdev * std_pct )

        future_cone = pd.DataFrame(index=map( np.datetime64, future_cone_dates ), data={'perf':temp_line, 
                                                                                        'line':temp_line, 
                                                                                        'sd_up':temp_sd_up, 
                                                                                        'sd_down':temp_sd_down } )
    
        perf_ts_r = perf_ts_r.append(future_cone)

    return perf_ts_r


def gen_date_ranges_interesting():
    periods = OrderedDict()
    # Dotcom bubble
    periods['Dotcom'] = (pd.Timestamp('20000310'), pd.Timestamp('20000910'))

    # Lehmann Brothers
    periods['Lehmann'] = (pd.Timestamp('20080801'), pd.Timestamp('20081001'))

    # 9/11
    periods['9/11'] = (pd.Timestamp('20010911'), pd.Timestamp('20011011'))

    # 05/08/11	US down grade and European Debt Crisis 2011
    periods[
        'US downgrade/European Debt Crisis'] = (pd.Timestamp('20110805'), pd.Timestamp('20110905'))

    # 16/03/11	Fukushima melt down 2011
    periods['Fukushima'] = (pd.Timestamp('20110316'), pd.Timestamp('20110416'))

    # 01/08/03	US Housing Bubble 2003
    periods['US Housing'] = (
        pd.Timestamp('20030108'), pd.Timestamp('20030208'))

    # 06/09/12	EZB IR Event 2012
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


def extract_interesting_date_ranges(df_rets):
    periods = gen_date_ranges_interesting()
    df_rets_dupe = df_rets.copy()
    df_rets_dupe.index = df_rets_dupe.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in periods.iteritems():
        period = df_rets_dupe.loc[start:end]
        if len(period) == 0:
            continue
        ranges[name] = period

    return ranges
