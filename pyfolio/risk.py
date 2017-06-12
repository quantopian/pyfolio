import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def create_risk_tear_sheet(positions,
                           style_factor_panel,
                           sectors=None,
                           caps=None,
                           shares_held=None,
                           volumes=None,
                           percentile=None):

    '''
    Creates risk tear sheet: computes and plots style factor exposures, sector
    exposures, market cap exposures and volume exposures.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily equity positions of algorithm, in dollars.
        - DataFrame with dates as index; equities as columns
        - Last column is cash held
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])             cash
        2017-04-03	-108062.40 	  4401.540     2.247757e+07
        2017-04-04	-108852.00	  4373.820     2.540999e+07
        2017-04-05	-119968.66	  4336.200     2.839812e+07

    style_factor_panel : pd.Panel
        Panel where each item is a DataFrame that tabulates style factor per
        equity per day.
        - Each item has dates as index; equities as columns
        - Example item:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	  -0.51284     1.39173
        2017-04-04	  -0.73381     0.98149
        2017-04-05	  -0.90132	   1.13981

    sector : pd.DataFrame
        Daily Morningstar sector code per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                     Equity(24   Equity(62
                       [AAPL])      [ABT])
        2017-04-03	     311.0       206.0
        2017-04-04	     311.0       206.0
        2017-04-05	     311.0	     206.0

    caps : pd.DataFrame
        Daily Morningstar sector code per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03     1.327160e+10     6.402460e+10
        2017-04-04	   1.329620e+10     6.403694e+10
        2017-04-05	   1.297464e+10	    6.397187e+10

    shares_held : pd.DataFrame
        Daily number of shares held by an algorithm.
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03             1915            -2595
        2017-04-04	           1968            -3272
        2017-04-05	           2104            -3917

    volumes : pd.DataFrame
        Daily volume per asset
        - DataFrame with dates as index and equities as columns
        - Example:
                          Equity(24        Equity(62
                            [AAPL])           [ABT])
        2017-04-03      34940859.00       4665573.80
        2017-04-04	    35603329.10       4818463.90
        2017-04-05	    41846731.75	      4129153.10

    percentile : float
        Percentile to use when computing and plotting volume exposures
    '''

    if percentile is None:
        percentile = 0.1

    for name, df in style_factor_panel.iteritems():
        sfe = compute_style_factor_exposures(positions, df)
        plot_style_factor_exposures(sfe, name)

    if sectors is not None:
        exposures = compute_sector_exposures(positions, sectors)
        plot_sector_exposures(exposures[0], exposures[1], exposures[2])

    if caps is not None:
        exposures = compute_cap_exposures(positions, caps)
        plot_cap_exposures(exposures[0], exposures[1], exposures[2])

    if volumes is not None:
        exposures = compute_volume_exposures(positions, volumes, percentile)
        plot_volume_exposures(exposures[0], exposures[1], exposures[2],
                              percentile)

    plt.show()


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


def plot_style_factor_exposures(tot_sfe, factor_name):
    '''
    Plots DataFrame output of compute_style_factor_exposures as a line graph

    Parameters
    ----------
    tot_sfe : pd.Series
        Daily style factor exposures; output of compute_style_factor_exposures
        - Time series with decimal style factor exposures
        - Example:
            2017-04-24    0.037820
            2017-04-25    0.016413
            2017-04-26   -0.021472
            2017-04-27   -0.024859

    factor_name : string
        Name of style factor, for use in graph title
    '''

    fig = plt.figure(figsize=(14, 6))
    plt.plot(tot_sfe.index, tot_sfe, label=factor_name)
    avg = tot_sfe.mean()
    plt.axhline(avg, linestyle='-.', label='Mean = {:.3}'.format(avg))
    plt.axhline(0, color='k', linestyle='-')
    plt.title('{} Weighted Exposure'.format(factor_name), fontsize='medium')
    plt.ylabel('{} Weighted Exposure'.format(factor_name))
    plt.legend()

    return fig


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


def plot_sector_exposures(long_exposures, short_exposures, gross_exposures):
    '''
    Plots outputs of compute_sector_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures, gross_exposures : arrays
        Arrays of long, short and gross sector exposures; output of
        compute_sector_exposures.
    '''

    sector_names = SECTORS.values()

    colors = ['#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99', '#99FFCC',
              '#99FFFF', '#99CCFF', '#9999FF', '#CC99FF', '#FF99FF']

    fig = plt.figure(figsize=[14, 6*3])
    gs = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.2)

    ax0 = plt.subplot(gs[:2, :])
    ax0.stackplot(long_exposures[0].index, long_exposures,
                  labels=sector_names, colors=colors, baseline='zero')
    ax0.stackplot(long_exposures[0].index, short_exposures,
                  colors=colors, baseline='zero')
    ax0.axhline(0, color='k', linestyle='-')
    ax0.set_title('Sector Exposures: Long and Short', fontsize='large')
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Proportion of Long/Short Exposure in Sectors')
    ax0.legend(loc=2, fontsize='medium')

    ax1 = plt.subplot(gs[2, :])
    ax1.stackplot(gross_exposures[0].index, gross_exposures,
                  labels=sector_names, colors=colors, baseline='zero')
    ax1.axhline(0, color='k', linestyle='-')
    ax1.set_title('Sector Exposures: Gross', fontsize='large')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Proportion of Gross Exposure in Sectors')
    ax1.legend(loc=2, fontsize='medium')

    return fig


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


def plot_cap_exposures(long_exposures, short_exposures, gross_exposures):
    '''
    Plots outputs of compute_cap_exposures as area charts

    Parameters
    ----------
    long_exposures, short_exposures, gross_exposures : arrays
        Arrays of long, short and gross market cap exposures; output of
        compute_cap_exposures.
    '''

    fig = plt.figure(figsize=(14, 6*3))
    gs = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)

    colors = ['#FF9999', '#FFCC99', '#99FF99', '#99CCFF', '#CC99FF']

    ax0 = plt.subplot(gs[:2, :])
    ax0.stackplot(long_exposures[0].index, long_exposures,
                  labels=CAP_NAMES, colors=colors, baseline='zero')
    ax0.stackplot(long_exposures[0].index, short_exposures,
                  colors=colors, baseline='zero')
    ax0.axhline(0, color='k', linestyle='-')
    ax0.set_title('Market Cap Exposures: Long and Short', fontsize='large')
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Proportion of Long/Short Exposure in Market Cap Buckets')
    ax0.legend(loc=2, fontsize='medium')

    ax1 = plt.subplot(gs[2, :])
    ax1.stackplot(gross_exposures[0].index, gross_exposures,
                  labels=CAP_NAMES, colors=colors, baseline='zero')
    ax1.axhline(0, color='k', linestyle='-')
    ax1.set_title('Market Cap Exposures: Gross', fontsize='large')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Proportion of Gross Exposure in Market Cap Buckets')
    ax1.legend(loc=2, fontsize='medium')

    return fig


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


def plot_volume_exposures(longed_threshold, shorted_threshold,
                          grossed_threshold, percentile):
    '''
    Plots outputs of compute_volume_exposures as line graphs

    Parameters
    ----------
    longed_threshold, shorted_threshold, grossed_threshold : pd.Series
        Series of longed, shorted and grossed volume exposures; output of
        compute_volume_exposures.

    percentile : float
        Percentile to use when computing and plotting volume exposures
        - See full explanation in create_risk_tear_sheet
    '''

    fig = plt.figure(figsize=(14, 3*6))
    gs = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.2)

    ax0 = plt.subplot(gs[:2, :])
    ax0.plot(longed_threshold.index, longed_threshold, label='long')
    ax0.plot(shorted_threshold.index, shorted_threshold, label='short')
    ax0.axhline(0, color='k')
    ax0.set_title('{}th Percentile of Proportion of Volume: Longs and Shorts'
                  .format(100*percentile), fontsize='large')
    ax0.set_xlabel('Date')
    ax0.set_ylabel('{}th Percentile of Proportion of Volume (%)'
                   .format(100*percentile))
    ax0.legend(fontsize='medium')

    ax1 = plt.subplot(gs[2, :])
    ax1.plot(grossed_threshold.index, grossed_threshold, label='gross')
    ax1.axhline(0, color='k')
    ax1.set_title('{}th Percentile of Proportion of Volume: Gross'
                  .format(100*percentile), fontsize='large')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('{}th Percentile of Proportion of Volume (%)'
                   .format(100*percentile))
    ax1.legend(fontsize='medium')

    return fig
