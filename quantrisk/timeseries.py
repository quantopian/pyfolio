from __future__ import division

from quant_utils.utils import *

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.signal as signal
import seaborn as sns
from collections import *
from operator import *
import string

import statsmodels.api as sm

import zipline
from zipline.finance.risk import RiskMetricsCumulative, RiskMetricsPeriod
from zipline.utils.factory import create_simulation_parameters
from zipline.utils import tradingcalendar

import datetime
from datetime import datetime
from datetime import timedelta
import pytz
import time

from bson import ObjectId

import os
import zlib

utc=pytz.UTC
indexTradingCal = pd.DatetimeIndex(tradingcalendar.trading_days)
indexTradingCal = indexTradingCal.normalize()


##### timeseries manipulation functions
def rolling_metric_stat(ret_ts, metric
                        , stat_func=np.mean
                        , window=63, sample_freq=21
                        , return_all_values=False):
    roll_results = pd.rolling_apply(ret_ts, window, metric).dropna()
    roll_results_sample = roll_results[ np.sort(range(len(roll_results)-1, 0, -sample_freq)) ]

    if return_all_values:
        return roll_results_sample
    else:
        temp_f = lambda x: stat_func(x)
        return temp_f(roll_results_sample)

def normalize(df, withStartingValue=1):
    if withStartingValue > 1:
        return withStartingValue * ( df / df.iloc[0] )
    else:
        return df / df.iloc[0]

def cum_returns(df, withStartingValue=None):
    if withStartingValue is None:
        return np.exp(np.log(1 + df).cumsum()) - 1
    else:
        return np.exp(np.log(1 + df).cumsum()) * withStartingValue

def aggregate_returns(df_daily_rets, convert_to):
    cumulate_returns = lambda x: cum_returns(x)[-1]
    if convert_to == 'daily':
        return df_daily_rets
    elif convert_to == 'weekly':
        return df_daily_rets.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == 'monthly':
        return df_daily_rets.groupby([lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == 'yearly':
        return df_daily_rets.groupby([lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be daily, weekly, monthly or yearly')

def detrendTS(theTS):
    return pd.Series(data=signal.detrend(theTS.values),index=theTS.index.values)

def appendSeries( ts1, ts2 ):
    '''
    @params
      ts1: <pd.Series>
          The 1st Series
      ts2: <pd.Series>
          The 2nd Series to be appended below 1st timeseries
    '''
    return pd.Series( data=np.concatenate([ts1,ts2]) , index=np.concatenate([ts1.index,ts2.index]) )

def dfTS(df, dateColumnLabel='Date'):
    # e.g.: takes a dataframe from a yahoo finance price csv and returns a df with datetime64 index
    colNames = df.columns
    tempDF = df.copy()
    indexDates = map(np.datetime64, df.ix[:,dateColumnLabel].values)
    # tempDF = pd.DataFrame(data=df.values , index=map(pd.to_datetime, indexDates))
    tempDF = pd.DataFrame(data=df.values , index=indexDates)
    tempDF.columns = colNames

    return tempDF.drop(axis=1,labels=dateColumnLabel).sort_index()

def multiTimeseriesToDF_fromDict( tsDictInput, asPctChange=False, startDate=None, endDate=None,
                        dropNA=False, ffillNA=False, fillNAvalue=None ):
    tempDict = {}

    for i in tsDictInput.keys():
        if asPctChange:
            # print(i)
            # print(tsDictInput.get(i).head())
            tempDict[ i ] = sliceTS(tsDictInput.get(i).pct_change().dropna(), startDate=startDate, endDate=endDate)
        else:
            tempDict[ i ] = sliceTS(tsDictInput.get(i), startDate=startDate, endDate=endDate)
    tempDF = pd.DataFrame(tempDict)
    if dropNA:
        tempDF = tempDF.dropna()
    elif ffillNA:
        tempDF = tempDF.fillna(method="ffill")
    elif fillNAvalue is not None:
        tempDF = tempDF.fillna(fillNAvalue)

    return tempDF

def multiTimeseriesToDF( tsSeriesList, tsSeriesNamesArr, asPctChange=False, startDate=None, endDate=None,
                        dropNA=False, ffillNA=False, fillNAvalue=None ):
    tempDict = {}

    for i in range(0,len(tsSeriesNamesArr)):
        if asPctChange:
            tempDict[ tsSeriesNamesArr[i] ] = sliceTS(tsSeriesList[i].pct_change().dropna(), startDate=startDate, endDate=endDate)
        else:
            tempDict[ tsSeriesNamesArr[i] ] = tsSeriesList[i]
    tempDF = pd.DataFrame(tempDict)
    if dropNA:
        tempDF = tempDF.dropna()
    elif ffillNA:
        tempDF = tempDF.fillna(method="ffill")
    elif fillNAvalue is not None:
        tempDF = tempDF.fillna(fillNAvalue)

    return tempDF

def sliceTS(theTS, startDate=None, endDate=None, inclusive_start=False, inclusive_end=False ):

    if (startDate is None) and (endDate is None):
        return theTS

    if startDate is None:
        if inclusive_end:
            return theTS[ :endDate ]
        else:
            return theTS[ :endDate ][:-1]

    if endDate is None:
        if inclusive_start:
            return theTS[startDate:]
        else:
            return theTS[startDate:][1:]

    if inclusive_start & inclusive_end:
        return theTS[ startDate:endDate ]

    if inclusive_start & ~inclusive_end:
        return theTS[ startDate:endDate ][:-1]
    elif ~inclusive_start & inclusive_end:
        return theTS[ startDate:endDate ][1:]
    elif ~inclusive_start & ~inclusive_end:
        return theTS[ startDate:endDate ][1:-1]

    return pd.Series()





##### timeseries manipulation functions


##### Strategy Performance statistics & timeseries analysis functions

def maxDrawdown(ts, inputIsNAV=True):
    if ts.size < 1:
        return np.nan

    if inputIsNAV:
        temp_ts = ts
    else:
        temp_ts = cum_returns(ts, withStartingValue=100)

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
    return -1*MDD

def annualReturn(ts, inputIsNAV=True, style='calendar'):
    # if style == 'compound' then return will be calculated in geometric terms: (1+mean(all_daily_returns))^252 - 1
    # if style == 'calendar' then return will be calculated as ((last_value - start_value)/start_value)/num_of_years
    # if style == 'arithmetic' then return is simply mean(all_daily_returns)*252
    if ts.size < 1:
        return np.nan

    if inputIsNAV:
        tempReturns = ts.pct_change().dropna()
        if style == 'calendar':
            num_years = len(tempReturns) / 252
            start_value = ts[0]
            end_value = ts[-1]
            return ((end_value - start_value)/start_value) / num_years
        if style == 'compound':
            return pow( (1 + tempReturns.mean()), 252 ) - 1
        else:
            return tempReturns.mean() * 252
    else:
        if style == 'calendar':
            num_years = len(ts) / 252
            temp_NAV = cum_returns(ts, withStartingValue=100)
            start_value = temp_NAV[0]
            end_value = temp_NAV[-1]
            return ((end_value - start_value)/start_value) / num_years
        if style == 'compound':
            return pow( (1 + ts.mean()), 252 ) - 1
        else:
            return ts.mean() * 252


def annualVolatility(ts, inputIsNAV=True):
    if ts.size < 2:
        return np.nan
    if inputIsNAV:
        tempReturns = ts.pct_change().dropna()
        return tempReturns.std() * np.sqrt(252)
    else:
        return ts.std() * np.sqrt(252)


def calmerRatio(ts, inputIsNAV=True, returns_style='calendar'):
    temp_max_dd = maxDrawdown(ts=ts, inputIsNAV=inputIsNAV)
    # print(temp_max_dd)
    if temp_max_dd < 0:
        if inputIsNAV:
            temp = annualReturn(ts=ts, inputIsNAV=True, style=returns_style) / abs(maxDrawdown(ts=ts,inputIsNAV=True))
        else:
            tempNAV = cum_returns(ts,withStartingValue=100)
            temp = annualReturn(ts=tempNAV, inputIsNAV=True, style=returns_style) / abs(maxDrawdown(ts=tempNAV,inputIsNAV=True))
        # print(temp)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan
    else:
        return temp

def sharpeRatio(ts, inputIsNAV=True, returns_style='calendar'):
    return annualReturn(ts, inputIsNAV=inputIsNAV, style=returns_style) / annualVolatility(ts, inputIsNAV=inputIsNAV)

def stabilityOfTimeseries( ts, logValue=True, inputIsNAV=True ):
    if ts.size < 2:
        return np.nan

    if logValue:
        if inputIsNAV:
            tempValues = np.log10(ts.values)
            tsLen = ts.size
        else:
            temp_ts = cum_returns(ts, withStartingValue=100)
            tempValues = np.log10(temp_ts.values)
            tsLen = temp_ts.size
    else:
        if inputIsNAV:
            tempValues = ts.values
            tsLen = ts.size
        else:
            temp_ts = cum_returns(ts, withStartingValue=100)
            tempValues = temp_ts.values
            tsLen = temp_ts.size

    X = range(0, tsLen)
    X = sm.add_constant(X)

    model = sm.OLS( tempValues, X ).fit()

    return model.rsquared

def calc_multifactor(df_rets, factors):
    import statsmodels.api as sm
    factors = factors.loc[df_rets.index]
    factors = sm.add_constant(factors)
    factors = factors.dropna(axis=0)
    results = sm.OLS(df_rets[factors.index], factors).fit()

    return results.params

def rolling_multifactor_beta(ser, multi_factor_df, rolling_window=63):
    results = [ calc_multifactor( ser[beg:end], multi_factor_df)
               for beg,end in zip(ser.index[0:-rolling_window],ser.index[rolling_window:]) ]

    return pd.DataFrame(index=ser.index[rolling_window:], data=results)

def multi_factor_alpha( factors_ts_list, single_ts, factor_names_list, input_is_returns=False
                        , annualized=False, annualize_factor=252, show_output=False):

    factors_ts = [ i.asfreq(freq='D',normalize=True) for i in factors_ts_list ]
    dep_var = single_ts.asfreq(freq='D',normalize=True)

    if not input_is_returns:
        factors_ts = [ i.pct_change().dropna() for i in factors_ts ]
        dep_var = dep_var.pct_change().dropna()

    factors_align = pd.DataFrame( factors_ts ).T.dropna()
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


def calc_alpha_beta(df_rets, benchmark_rets, startDate=None, endDate=None, return_beta_only=False, inputs_are_returns=True, normalize=False, remove_zeros=False):
    if not inputs_are_returns:
        df_rets = df_rets.pct_change().dropna()
        benchmark_rets = benchmark_rets.pct_change().dropna()

    if startDate != None:
        df_rets = df_rets[startDate:]

    if endDate != None:
        df_rets = df_rets[:endDate]

    if df_rets.ndim == 1:
        if remove_zeros:
            df_rets = df_rets[df_rets != 0]

        if normalize:
            ret_index = df_rets.index.normalize()
        else:
            ret_index = df_rets.index

        beta, alpha = sp.stats.linregress(benchmark_rets.loc[ret_index].values,
                                          df_rets.values)[:2]

    if df_rets.ndim == 2:
        beta = pd.Series(index=df_rets.columns)
        alpha = pd.Series(index=df_rets.columns)
        for algo_id in df_rets:
            df = df_rets[algo_id]
            if remove_zeros:
                df = df[df != 0]
            if normalize:
                ret_index = df.index.normalize()
            else:
                ret_index = df.index
            beta[algo_id], alpha[algo_id] = sp.stats.linregress(benchmark_rets.loc[ret_index].values,
                                                                df.values)[:2]
        alpha.name = 'alpha'
        beta.name = 'beta'

    if return_beta_only:
        return beta
    else:
        return alpha * 252, beta




def rolling_beta(ser, benchmark_rets, rolling_window=63):
    results = [ calc_alpha_beta( ser[beg:end], benchmark_rets, return_beta_only=True, normalize=True)
               for beg,end in zip(ser.index[0:-rolling_window],ser.index[rolling_window:]) ]

    return pd.Series(index=ser.index[rolling_window:], data=results)


def rolling_beta_TS(theTS, benchmarkTS, startDate=None, endDate=None, rolling_window=63):


    if startDate != None:
        theTS = theTS[(startDate -  timedelta(days=rolling_window)):]

    if endDate != None:
        theTS = theTS[:endDate]


    results = [ betaTimeseries( theTS[beg:end], benchmarkTS)
               for beg,end in zip(theTS.index[0:-rolling_window], theTS.index[rolling_window:]) ]


    return pd.Series(index=theTS.index[rolling_window:], data=results)



def betaTimeseries( theTS, benchmarkTS, startDate=None, endDate=None, inputIsReturns=False):
    tempTS = theTS.copy()
    tempBench = benchmarkTS.copy()

    tempTS = tempTS.asfreq(freq='D',normalize=True)
    tempBench = tempBench.asfreq(freq='D',normalize=True)

    if not inputIsReturns:
        tempTS = tempTS.pct_change().dropna()
        tempBench = tempBench.pct_change().dropna()

    if startDate != None:
        tempTS = tempTS[startdate:]
        tempBench = tempBench[startdate:]

    if endDate != None:
        tempTS = tempTS[:enddate]
        tempBench = tempBench[:enddate]

    tempTS = tempTS[ np.isfinite(tempTS) ]
    tempBench = tempBench[ np.isfinite(tempBench) ]

    # remove intraday timestamps by normalizing since only working with daily data right now
    # tempTS.reindex(tempTS.index.normalize())
    # tempBench.reindex(tempBench.index.normalize())

    # tempTS.reindex(indexTradingCal)
    # tempBench.reindex(indexTradingCal)

    tempAlign = tempBench.align(tempTS,join='inner')
    alignBench = tempAlign[0]
    alignTS = tempAlign[1]
    # print( alignBench.head() )
    # print( alignTS.head() )

    if alignBench.shape[0] < 2:
        return np.nan
    if alignTS.shape[0] < 2:
        return np.nan

    regX = np.array( alignBench.values )
    regY = np.array( alignTS.values )

    regX = np.reshape(regX,len(regX))
    regY = np.reshape(regY,len(regY))

    m, b = np.polyfit(regX, regY, 1)


    return m

def rolling_beta_TS(ts=None, benchmark=None, startDate=None, endDate=None, rolling_window=63, sampling_days=None ):
#     ts = ts.normalize()
    if startDate != None:
        #set start date rolling_window trading days before startDate
        tsp = ts[startDate:].index[0]
        true_start = ts.index.get_loc(tsp) - rolling_window
        ts = ts[true_start:]

    if endDate != None:
        ts = ts[:endDate]


    results = [ betaTimeseries( ts[beg:end], benchmark)
               for beg,end in zip(ts.index[0:-rolling_window], ts.index[rolling_window:]) ]

    results = pd.Series(index=ts.index[rolling_window:], data=results)

    if sampling_days is not None:
        results = results[::sampling_days]

    return results

def calc_beta_like_zipline(df_rets, benchmark_rets, startDate=None, endDate=None, inputs_are_returns=True):
    if not inputs_are_returns:
        df_rets = df_rets.pct_change().dropna()
        benchmark_rets = benchmark_rets.pct_change().dropna()

    if startDate != None:
        df_rets = df_rets[startDate:]

    if endDate != None:
        df_rets = df_rets[:endDate]

    benchmark_rets = benchmark_rets.loc[df_rets.index.normalize()]

    returns_matrix = np.vstack([df_rets.values, benchmark_rets.values])

    C = np.cov(returns_matrix, ddof=1)
    algorithm_covariance = C[0][1]
    benchmark_variance = C[1][1]
    beta = algorithm_covariance / benchmark_variance

    return beta

def calc_beta_like_zipline_rolling(df_rets, benchmark_rets, startDate=None, endDate=None, sampling_days=None, rolling_window=252, inputs_are_returns=True):

    if startDate != None:
        #set start date rolling_window trading days before startDate
        tsp = df_rets[startDate:].index[0]
        true_start = df_rets.index.get_loc(tsp) - rolling_window
        df_rets = df_rets[true_start:]

    if endDate != None:
        df_rets = df_rets[:endDate]

    results = [ calc_beta_like_zipline( df_rets[beg:end], benchmark_rets, inputs_are_returns=inputs_are_returns)
               for beg,end in zip(df_rets.index[0:-rolling_window], df_rets.index[rolling_window:]) ]

    results = pd.Series(index=df_rets.index[rolling_window:], data=results)

    if sampling_days is not None:
        results = results[::sampling_days]

    return results


def hurst(ts, lagsToTest=20):
    tau = []
    lagvec = []
    #  Step through the different lags
    for lag in range(2,lagsToTest):
        #  produce price difference with lag
        pp = np.subtract(ts[lag:],ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the differnce vector
        tau.append(np.sqrt(np.std(pp)))
    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)
    # calculate hurst
    hurst = m[0]*2
    # plot lag vs variance
    #py.plot(lagvec,tau,'o'); show()
    return hurst

def halfLife(ts):

    price = pd.Series(ts)
    lagged_price = price.shift(1).fillna(method="bfill")
    delta = price - lagged_price
    beta = np.polyfit(lagged_price, delta, 1)[0]
    half_life = (-1*np.log(2)/beta)

    return half_life

def perf_stats(ts, inputIsNAV=True, returns_style='compound', return_as_dict=False):
    all_stats = {}
    all_stats['annual_return'] = annualReturn(ts, inputIsNAV=inputIsNAV, style=returns_style)
    all_stats['annual_volatility'] = annualVolatility(ts, inputIsNAV=inputIsNAV)
    all_stats['sharpe_ratio'] = sharpeRatio(ts, inputIsNAV=inputIsNAV, returns_style=returns_style)
    all_stats['calmar_ratio'] = calmerRatio(ts, inputIsNAV=inputIsNAV, returns_style=returns_style)
    all_stats['stability'] = stabilityOfTimeseries(ts, inputIsNAV=inputIsNAV)
    all_stats['max_drawdown'] = maxDrawdown(ts, inputIsNAV=inputIsNAV)
    # print(all_stats)

    if return_as_dict:
        return all_stats
    else:
        all_stats_df = pd.DataFrame(index=all_stats.keys(), data=all_stats.values())
        all_stats_df.columns = ['perf_stats']
        return all_stats_df


##### Strategy Performance statistics & timeseries analysis functions
def calc_correl_matrix(tsDict, startDate=None, endDate=None, returnAsDict=False):
    # if 'returnAsDict' = False, then it will return in the form of the MultiIndex dataframe that corr() returns
    tempDF = multiTimeseriesToDF_fromDict(tsDict, asPctChange=True, startDate=startDate, endDate=endDate, dropNA=True)
    if returnAsDict:
        tempDF_unstack = tempDF.corr().unstack()
        tempDict = {}
        for i in tsDict.keys():
            tempDict[i] = tempDF_unstack[i]
        return tempDict
    else:
        return tempDF.corr()

def calc_avg_pairwise_correl_dict(tsDict, startDate=None, endDate=None, overlapping_periods_for_all=True):
    pairwise_dict = {}
    if overlapping_periods_for_all:
        # this path first joins all the timeseries into dataframe containing rows (days) that are valid in ALL timeseries
        tempcorrel = calc_correl_matrix(tsDict, startDate, endDate, returnAsDict=True)
        for i in tempcorrel.keys():
            temp_series = tempcorrel[i].dropna()
            # the below just removes the affect of the results containing the 1.0 correlation that each element has with itself
            if len(temp_series) > 0:
                temp_series_avg = np.append(temp_series,-1.0).sum() / (temp_series.size - 1)
                pairwise_dict[i] = temp_series_avg
            else:
                pairwise_dict[i] = np.nan
    else:
        # this path computes each pairwise correl for days valid across just those 2 inputs in each individual pair
        # so if 1 input only has 10 valid days, each of its pairwise correlations will be across those 10 days,
        # but for other pairs of timeseries that have more valid days, those correlations will include all the valid days
        for i in tsDict.keys():
            pair_correl_arr = np.array([],dtype='float')
            # pair_correl_arr = pd.Series()
            for j in ( set(tsDict.keys()) - {i} ):
                pair_dict = {}
                pair_dict[i] = tsDict[i]
                pair_dict[j] = tsDict[j]
                tempcorrel = calc_correl_matrix(pair_dict, startDate, endDate, returnAsDict=True)
                # pair_correl_arr = np.append( pair_correl_arr, tempcorrel[i][j] )
                if not np.isnan(tempcorrel[i][j]):
                    pair_correl_arr = np.append( pair_correl_arr, tempcorrel[i][j] )
                    # pair_correl_arr = pd.Series.append( pair_correl_arr, tempcorrel[i][j] )

            if len(pair_correl_arr) > 0:
                pairwise_dict[i] = pair_correl_arr.mean()
            else:
                pairwise_dict[i] = np.nan

    return pairwise_dict

def get_max_draw_down_underwater(underwater):
    valley = np.argmax(underwater) # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    return peak, valley, recovery

def get_max_draw_down(df_rets):
    df_rets = df_rets.copy()
    df_cum = cum_returns(df_rets)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum
    return get_max_draw_down_underwater(underwater)

def get_top_draw_downs(df_rets, top=10):
    df_rets = df_rets.copy()
    df_cum = cum_returns(df_rets)
    running_max = np.maximum.accumulate(df_cum)
    underwater = running_max - df_cum

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_draw_down_underwater(underwater)
        # Slice out draw-down period
        underwater = pd.concat([underwater.loc[:peak].iloc[:-1], underwater.loc[recovery:].iloc[1:]])
        drawdowns.append((peak, valley, recovery))
        if len(df_rets) == 0:
            break
    return drawdowns

def gen_drawdown_table(df_rets, top=10):
    df_cum = cum_returns(df_rets,1)
    drawdown_periods = get_top_draw_downs(df_rets, top=top)
    df_drawdowns = pd.DataFrame(index=range(top), columns=['net drawdown in %',
                                                           'peak date',
                                                           'valley date',
                                                           'recovery date',
                                                           'duration'])
    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        df_drawdowns.loc[i, 'duration'] = len(pd.date_range(peak, recovery, freq='B'))
        df_drawdowns.loc[i, 'peak date'] = peak
        df_drawdowns.loc[i, 'valley date'] = valley
        df_drawdowns.loc[i, 'recovery date'] = recovery
        # df_drawdowns.loc[i, 'net drawdown in %'] = (df_cum.loc[peak] - df_cum.loc[valley]) * 100
        df_drawdowns.loc[i, 'net drawdown in %'] = ( (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak] ) * 100

    df_drawdowns['peak date'] = pd.to_datetime(df_drawdowns['peak date'],unit='D')
    df_drawdowns['valley date'] = pd.to_datetime(df_drawdowns['valley date'],unit='D')
    df_drawdowns['recovery date'] = pd.to_datetime(df_drawdowns['recovery date'],unit='D')

    return df_drawdowns

def rolling_sharpe(df_rets, rolling_sharpe_window):
    return pd.rolling_mean(df_rets, rolling_sharpe_window) / pd.rolling_std(df_rets, rolling_sharpe_window) * np.sqrt(252)
