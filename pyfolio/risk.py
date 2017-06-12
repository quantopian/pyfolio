import numpy as np
import matplotlib.pyplot as plt

SECTORS = \
    {
        101: '101 Basic Materials',
        102: '102 Consumer Cyclical',
        103: '103 Financial Services',
        104: '104 Real Estate',
        205: '205 Consumer Defensive',
        206: '206 Healthcare',
        207: '207 Utilities',
        308: '308 Communication Services',
        309: '309 Energy',
        310: '310 Industrials',
        311: '311 Technology'
    }

CAP_CUTOFFS = [50000000, 300000000, 2000000000, 10000000000, 200000000000]
CAP_NAMES = ['Micro', 'Small', 'Mid', 'Large', 'Mega']


def compute_style_factor_exposures(positions, risk_factor):
    '''
    Returns style factor exposure of an algorithm's positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in create_risk_tear_sheet

    risk_factor : pd.DataFrame
        Daily risk factor per asset.
        - DataFrame with dates as index and equities as columns
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	  -0.51284     1.39173
        2017-04-04	  -0.73381     0.98149
        2017-04-05	  -0.90132	   1.13981
    '''

    positions_wo_cash = positions.drop('cash', axis=1)
    gross_exposure = positions_wo_cash.abs().sum(axis=1)

    sfe = positions_wo_cash.multiply(risk_factor, level=1) \
        .divide(gross_exposure, axis='index')
    tot_sfe = sfe.sum(axis=1, skipna=True)

    return tot_sfe


def plot_style_factor_exposures(tot_sfe, factor_name, ax=None):
    '''
    Plots DataFrame output of compute_style_factor_exposures as a line graph

    Parameters
    ----------
    tot_sfe : pd.Series
        Daily style factor exposures (output of compute_style_factor_exposures)
        - Time series with decimal style factor exposures
        - Example:
            2017-04-24    0.037820
            2017-04-25    0.016413
            2017-04-26   -0.021472
            2017-04-27   -0.024859

    factor_name : string
        Name of style factor, for use in graph title
    '''

    if ax is None:
        ax = plt.gca()

    ax.plot(tot_sfe.index, tot_sfe, label=factor_name)
    avg = tot_sfe.mean()
    ax.axhline(avg, linestyle='-.', label='Mean = {:.3}'.format(avg))
    ax.axhline(0, color='k', linestyle='-')
    ax.set_title('{} Weighted Exposure'.format(factor_name), fontsize='medium')
    ax.set_ylabel('{} Weighted Exposure'.format(factor_name))
    ax.legend()

    return ax


def compute_sector_exposures(positions, sectors):
    '''
    Returns arrays of long, short and gross sector exposures of an algorithm's
    positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in compute_style_factor_exposures.

    sector : pd.DataFrame
        Daily Morningstar sector code per asset
        - See full explanation in create_risk_tear_sheet
    '''

    sector_ids = SECTORS.keys()
    long_exposures = []
    short_exposures = []
    gross_exposures = []

    positions_wo_cash = positions.drop('cash', axis=1)
    long_exposure = positions_wo_cash[positions_wo_cash > 0].sum(axis=1)
    short_exposure = positions_wo_cash[positions_wo_cash < 0].abs().sum(axis=1)
    gross_exposure = positions_wo_cash.abs().sum(axis=1)

    for sector_id in sector_ids:
        in_sector = positions_wo_cash[sectors == sector_id]

        long_sector = in_sector[in_sector > 0] \
            .sum(axis=1).divide(long_exposure)

        short_sector = in_sector[in_sector < 0] \
            .sum(axis=1).divide(short_exposure)

        gross_sector = in_sector.abs().sum(axis=1).divide(gross_exposure)

        long_exposures.append(long_sector)
        short_exposures.append(short_sector)
        gross_exposures.append(gross_sector)

    return long_exposures, short_exposures, gross_exposures


def plot_sector_exposures_longshort(long_exposures, short_exposures, ax=None):
    '''
    Plots outputs of compute_sector_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures : arrays
        Arrays of long and short sector exposures (output of
        compute_sector_exposures).
    '''

    if ax is None:
        ax = plt.gca()

    sector_names = SECTORS.values()

    colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFCC',
              '#99FFFF', '#99CCFF', '#9999FF', '#CC99FF', '#FF99FF']

    ax.stackplot(long_exposures[0].index, long_exposures,
                 labels=sector_names, colors=colors, baseline='zero')
    ax.stackplot(long_exposures[0].index, short_exposures,
                 colors=colors, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_title('Sector Exposures: Long and Short', fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion of Long/Short Exposure in Sectors')
    ax.legend(loc=2, fontsize='medium')

    return ax


def plot_sector_exposures_gross(gross_exposures, ax=None):
    '''
    Plots outputs of compute_sector_exposures as area charts

    Parameters
    ----------
    gross_exposures : arrays
        Arrays of gross sector exposures (output of compute_sector_exposures).
    '''

    if ax is None:
        ax = plt.gca()

    sector_names = SECTORS.values()

    colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFCC',
              '#99FFFF', '#99CCFF', '#9999FF', '#CC99FF', '#FF99FF']

    ax.stackplot(gross_exposures[0].index, gross_exposures,
                 labels=sector_names, colors=colors, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_title('Sector Exposures: Gross', fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion of Gross Exposure in Sectors')
    ax.legend(loc=2, fontsize='medium')

    return ax


def compute_cap_exposures(positions, caps):
    '''
    Returns arrays of long, short and gross market cap exposures of an
    algorithm's positions

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - See full explanation in compute_style_factor_exposures.

    caps : pd.DataFrame
        Daily Morningstar sector code per asset
        - See full explanation in create_risk_tear_sheet
    '''

    long_exposures = []
    short_exposures = []
    gross_exposures = []

    positions_wo_cash = positions.drop('cash', axis=1)
    tot_gross_exposure = positions_wo_cash.abs().sum(axis=1)
    tot_long_exposure = positions_wo_cash[positions_wo_cash > 0].sum(axis=1)
    tot_short_exposure = positions_wo_cash[positions_wo_cash < 0] \
        .abs().sum(axis=1)

    for i in range(1, len(CAP_CUTOFFS)+1):
        if i == len(CAP_CUTOFFS):
            in_bucket = positions_wo_cash[caps >= CAP_CUTOFFS[-1]]
        else:
            in_bucket = positions_wo_cash[(caps <= CAP_CUTOFFS[i])
                                          & (caps >= CAP_CUTOFFS[i-1])]

        gross_bucket = in_bucket.abs().sum(axis=1).divide(tot_gross_exposure)
        long_bucket = in_bucket[in_bucket > 0] \
            .sum(axis=1).divide(tot_long_exposure)
        short_bucket = in_bucket[in_bucket < 0] \
            .sum(axis=1).divide(tot_short_exposure)

        gross_exposures.append(gross_bucket)
        long_exposures.append(long_bucket)
        short_exposures.append(short_bucket)

    return long_exposures, short_exposures, gross_exposures


def plot_cap_exposures_longshort(long_exposures, short_exposures, ax=None):
    '''
    Plots outputs of compute_cap_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures : arrays
        Arrays of long and short market cap exposures (output of
        compute_cap_exposures).
    '''

    if ax is None:
        ax = plt.gca()

    colors = ['#FF9999', '#FFCC99', '#99FF99', '#99CCFF', '#CC99FF']

    ax.stackplot(long_exposures[0].index, long_exposures,
                 labels=CAP_NAMES, colors=colors, baseline='zero')
    ax.stackplot(long_exposures[0].index, short_exposures,
                 colors=colors, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_title('Market Cap Exposures: Long and Short', fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion of Long/Short Exposure in Market Cap Buckets')
    ax.legend(loc=2, fontsize='medium')

    return ax


def plot_cap_exposures_gross(gross_exposures, ax=None):
    '''
    Plots outputs of compute_cap_exposures as area charts

    Parameters
    ----------
    gross_exposures : arrays
        Arrays of gross market cap exposures (output of compute_cap_exposures).
    '''

    if ax is None:
        ax = plt.gca()

    colors = ['#FF9999', '#FFCC99', '#99FF99', '#99CCFF', '#CC99FF']

    ax.stackplot(gross_exposures[0].index, gross_exposures,
                 labels=CAP_NAMES, colors=colors, baseline='zero')
    ax.axhline(0, color='k', linestyle='-')
    ax.set_title('Market Cap Exposures: Gross', fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion of Gross Exposure in Market Cap Buckets')
    ax.legend(loc=2, fontsize='medium')

    return ax


def compute_volume_exposures(shares_held, volumes, percentile):
    '''
    Returns arrays of pth percentile of long, short and gross volume exposures
    of an algorithm's held shares

    Parameters
    ----------
    shares_held : pd.DataFrame
        Daily number of shares held by an algorithm.
        - See full explanation in create_risk_tear_sheet

    volume : pd.DataFrame
        Daily volume per asset
        - See full explanation in create_risk_tear_sheet

    percentile : float
        Percentile to use when computing and plotting volume exposures
        - See full explanation in create_risk_tear_sheet
    '''

    shares_held = shares_held.replace(0, np.nan)

    shares_longed = shares_held[shares_held > 0]
    shares_shorted = shares_held[shares_held > 0]
    shares_grossed = shares_held.abs()

    longed_frac = shares_longed.divide(volumes)
    shorted_frac = shares_shorted.divide(volumes)
    grossed_frac = shares_grossed.divide(volumes)

    longed_threshold = 100*longed_frac.quantile(percentile, axis=1)
    shorted_threshold = 100*shorted_frac.quantile(percentile, axis=1)
    grossed_threshold = 100*grossed_frac.quantile(percentile, axis=1)

    return longed_threshold, shorted_threshold, grossed_threshold


def plot_volume_exposures_longshort(longed_threshold, shorted_threshold,
                                    percentile, ax=None):
    '''
    Plots outputs of compute_volume_exposures as line graphs

    Parameters
    ----------
    longed_threshold, shorted_threshold : pd.Series
        Series of longed and shorted volume exposures (output of
        compute_volume_exposures).

    percentile : float
        Percentile to use when computing and plotting volume exposures.
        - See full explanation in create_risk_tear_sheet
    '''

    if ax is None:
        ax = plt.gca()

    ax.plot(longed_threshold.index, longed_threshold, label='long')
    ax.plot(shorted_threshold.index, shorted_threshold, label='short')
    ax.axhline(0, color='k')
    ax.set_title('{}th Percentile of Proportion of Volume: Longs and Shorts'
                 .format(100*percentile), fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('{}th Percentile of Proportion of Volume (%)'
                  .format(100*percentile))
    ax.legend(fontsize='medium')

    return ax


def plot_volume_exposures_gross(grossed_threshold, percentile, ax=None):
    '''
    Plots outputs of compute_volume_exposures as line graphs

    Parameters
    ----------
    grossed_threshold : pd.Series
        Series of grossed volume exposures (output of
        compute_volume_exposures).

    percentile : float
        Percentile to use when computing and plotting volume exposures
        - See full explanation in create_risk_tear_sheet
    '''

    if ax is None:
        ax = plt.gca()

    ax.plot(grossed_threshold.index, grossed_threshold, label='gross')
    ax.axhline(0, color='k')
    ax.set_title('{}th Percentile of Proportion of Volume: Gross'
                 .format(100*percentile), fontsize='large')
    ax.set_xlabel('Date')
    ax.set_ylabel('{}th Percentile of Proportion of Volume (%)'
                  .format(100*percentile))
    ax.legend(fontsize='medium')

    return ax
