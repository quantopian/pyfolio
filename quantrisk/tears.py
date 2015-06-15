import timeseries
import utils

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def show_perf_stats(df_rets, algo_create_date, benchmark_rets):
    df_rets_backtest = df_rets[ df_rets.index < algo_create_date]
    df_rets_live = df_rets[ df_rets.index > algo_create_date]

    print 'Out-of-Sample Months: ' + str( int( len(df_rets_live) / 21) )
    print 'Backtest Months: ' + str( int( len(df_rets_backtest) / 21) )

    perf_stats_backtest = np.round(timeseries.perf_stats(df_rets_backtest, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_backtest_ab = np.round(timeseries.calc_alpha_beta(df_rets_backtest, benchmark_rets), 2)
    perf_stats_backtest.loc['alpha'] = perf_stats_backtest_ab[0]
    perf_stats_backtest.loc['beta'] = perf_stats_backtest_ab[1]
    perf_stats_backtest.columns = ['Backtest']

    perf_stats_live = np.round(timeseries.perf_stats(df_rets_live, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_live_ab = np.round(timeseries.calc_alpha_beta(df_rets_live, benchmark_rets), 2)
    perf_stats_live.loc['alpha'] = perf_stats_live_ab[0]
    perf_stats_live.loc['beta'] = perf_stats_live_ab[1]
    perf_stats_live.columns = ['Out_of_Sample']

    perf_stats_all = np.round(timeseries.perf_stats(df_rets, inputIsNAV=False, returns_style='arithmetic'), 2)
    perf_stats_all_ab = np.round(timeseries.calc_alpha_beta(df_rets, benchmark_rets), 2)
    perf_stats_all.loc['alpha'] = perf_stats_all_ab[0]
    perf_stats_all.loc['beta'] = perf_stats_all_ab[1]
    perf_stats_all.columns = ['All_History']

    perf_stats_both = perf_stats_backtest.join(perf_stats_live, how='inner')
    perf_stats_both = perf_stats_both.join(perf_stats_all, how='inner')

    print perf_stats_both
    
def show_cone_plot(algo_ts, df_rets, benchmark_rets, benchmark2_rets, algo_create_date, future_cone_stdev, timeseries_input_only, fig, ax, show_plot=True):
    
    if not timeseries_input_only and algo_ts.index[-1] <= algo_create_date and show_plot:
        algo_ts.plot(lw=3, color='forestgreen', label='', alpha=0.6)
        plt.legend(['S&P500', '7-10yr Bond', 'Algo backtest'], loc='upper left')
    else:
        if show_plot:
            algo_ts[:algo_create_date].plot(lw=3, color='forestgreen', label='', alpha=0.6)
            algo_ts[algo_create_date:].plot(lw=4, color='red', label='', alpha=0.6)

            cone_df = timeseries.cone_rolling(df_rets, num_stdev=future_cone_stdev, cone_fit_end_date=algo_create_date)

            cone_df_fit = cone_df[ cone_df.index < algo_create_date]
            cone_df_live = cone_df[ cone_df.index > algo_create_date]
            cone_df_live = cone_df_live[ cone_df_live.index < df_rets.index[-1] ]
            cone_df_future = cone_df[ cone_df.index > df_rets.index[-1] ]

            cone_df_fit['line'].plot(ls='--', lw=2, color='forestgreen', alpha=0.7)
            cone_df_live['line'].plot(ls='--', lw=2, color='coral', alpha=0.7)
            cone_df_future['line'].plot(ls='--', lw=2, color='navy', alpha=0.7)

            ax.fill_between(cone_df_live.index, 
                            cone_df_live.sd_down, 
                            cone_df_live.sd_up, 
                            color='coral', alpha=0.20)

            ax.fill_between(cone_df_future.index, 
                            cone_df_future.sd_down, 
                            cone_df_future.sd_up, 
                            color='navy', alpha=0.15)
            
            plt.axhline(1.0 , linestyle='--', color='black', lw=2)
            plt.ylabel('Cumulative returns', fontsize=14)
            plt.xlim((algo_ts.index[0], cone_df.index[-1]))
        
            if timeseries_input_only:
                plt.legend(['S&P500', '7-10yr Bond', 'Portfolio'], loc='upper left')
            else:
                plt.legend(['S&P500', '7-10yr Bond', 'Algo backtest','Algo LIVE'], loc='upper left')
        
        return cone_df, cone_df_fit, cone_df_live, cone_df_future