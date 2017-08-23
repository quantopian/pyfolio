import numpy as np
import pandas as pd
import unittest

from pyfolio.perf_attrib import perf_attrib


def generate_toy_risk_model_output(start_date='2017-01-01', periods=10):
    """
    Generate toy risk model output.

    Parameters
    ----------
    start_date : str
        date to start generating toy data
    periods : int
        number of days for which to generate toy data

    Returns
    -------
    tuple of (returns, factor_returns, positions, factor_loadings)
    returns : pd.DataFrame
    factor_returns : pd.DataFrame
    """
    dts = pd.date_range(start_date, periods=periods)
    np.random.seed(123)
    tickers = ['AAPL', 'TLT', 'XOM']
    styles = ['factor1', 'factor2']

    returns = pd.Series(index=dts,
                        data=np.random.randn(periods)) / 100

    factor_returns = pd.DataFrame(
        columns=styles, index=dts,
        data=np.random.randn(periods, len(styles))) / 100

    arrays = [dts, tickers]
    index = pd.MultiIndex.from_product(arrays, names=['dt', 'ticker'])

    positions = pd.DataFrame(
        columns=tickers, index=dts,
        data=np.random.randint(100, size=(periods, len(tickers)))
    )
    positions['cash'] = np.zeros(periods)

    factor_loadings = pd.DataFrame(
        columns=styles, index=index,
        data=np.random.randn(periods * len(tickers), len(styles))
    )

    return returns, positions, factor_returns, factor_loadings


class PerfAttribTestCase(unittest.TestCase):

    def test_perf_attrib_simple(self):

        start_date = '2017-01-01'
        periods = 2
        dts = pd.date_range(start_date, periods=periods)
        dts.name = 'dt'

        tickers = ['stock1', 'stock2']
        styles = ['risk_factor1', 'risk_factor2']

        returns = pd.Series(data=[0.1, 0.1], index=dts)

        factor_returns = pd.DataFrame(
            columns=styles,
            index=dts,
            data={'risk_factor1': [.1, .1],
                  'risk_factor2': [.1, .1]}
        )

        positions = pd.DataFrame(
            index=dts,
            data={'stock1': [20, 20],
                  'stock2': [50, 50],
                  'cash': [0, 0]}
        )

        index = pd.MultiIndex.from_product(
            [dts, tickers], names=['dt', 'ticker'])

        factor_loadings = pd.DataFrame(
            columns=styles,
            index=index,
            data={'risk_factor1': [0.25, 0.25, 0.25, 0.25],
                  'risk_factor2': [0.25, 0.25, 0.25, 0.25]}
        )

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2', 'common_returns',
                     'specific_returns', 'total_returns'],
            data={'risk_factor1': [0.025, 0.025],
                  'risk_factor2': [0.025, 0.025],
                  'common_returns': [0.05, 0.05],
                  'specific_returns': [0.05, 0.05],
                  'total_returns': returns}
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2'],
            data={'risk_factor1': [0.25, 0.25],
                  'risk_factor2': [0.25, 0.25]}
        )

        exposures_portfolio, perf_attrib_output = perf_attrib(returns,
                                                              positions,
                                                              factor_returns,
                                                              factor_loadings)

        pd.util.testing.assert_frame_equal(expected_perf_attrib_output,
                                           perf_attrib_output)

        pd.util.testing.assert_frame_equal(expected_exposures_portfolio,
                                           exposures_portfolio)

        # test long and short positions
        positions = pd.DataFrame(index=dts,
                                 data={'stock1': [20, 20],
                                       'stock2': [-20, -20],
                                       'cash': [20, 20]})

        exposures_portfolio, perf_attrib_output = perf_attrib(returns,
                                                              positions,
                                                              factor_returns,
                                                              factor_loadings)

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2', 'common_returns',
                     'specific_returns', 'total_returns'],
            data={'risk_factor1': [0.0, 0.0],
                  'risk_factor2': [0.0, 0.0],
                  'common_returns': [0.0, 0.0],
                  'specific_returns': [0.1, 0.1],
                  'total_returns': returns}
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2'],
            data={'risk_factor1': [0.0, 0.0],
                  'risk_factor2': [0.0, 0.0]}
        )

        pd.util.testing.assert_frame_equal(expected_perf_attrib_output,
                                           perf_attrib_output)

        pd.util.testing.assert_frame_equal(expected_exposures_portfolio,
                                           exposures_portfolio)
