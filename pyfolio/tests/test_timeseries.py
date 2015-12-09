from __future__ import division

from unittest import TestCase
from nose_parameterized import parameterized
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from .. import timeseries
from .. import utils

DECIMAL_PLACES = 8


class TestDrawdown(TestCase):
    px_list_1 = np.array(
        [100, 120, 100, 80, 70, 110, 180, 150]) / 100.  # Simple
    px_list_2 = np.array(
        [100, 120, 100, 80, 70, 80, 90, 90]) / 100.  # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=8, freq='D')

    @parameterized.expand([
        (pd.Series(px_list_1,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         pd.Timestamp('2000-1-9')),
        (pd.Series(px_list_2,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         None)
    ])
    def test_get_max_drawdown(
            self, px, expected_peak, expected_valley, expected_recovery):
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_drawdown(rets)
        # Need to use isnull because the result can be NaN, NaT, etc.
        self.assertTrue(
            pd.isnull(peak)) if expected_peak is None else self.assertEqual(
                peak,
                expected_peak)
        self.assertTrue(
            pd.isnull(valley)) if expected_valley is None else \
            self.assertEqual(
                valley,
                expected_valley)
        self.assertTrue(
            pd.isnull(recovery)) if expected_recovery is None else \
            self.assertEqual(
                recovery,
                expected_recovery)

    @parameterized.expand([
        (pd.Series(px_list_2,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         None,
         None),
        (pd.Series(px_list_1,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         pd.Timestamp('2000-1-9'),
         4)
    ])
    def test_gen_drawdown_table(self, px, expected_peak,
                                expected_valley, expected_recovery,
                                expected_duration):
        rets = px.pct_change().iloc[1:]

        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[
                    0,
                    'peak date'])) if expected_peak is None \
            else self.assertEqual(drawdowns.loc[0, 'peak date'],
                                  expected_peak)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[0, 'valley date'])) \
            if expected_valley is None else self.assertEqual(
                drawdowns.loc[0, 'valley date'],
                expected_valley)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[0, 'recovery date'])) \
            if expected_recovery is None else self.assertEqual(
                drawdowns.loc[0, 'recovery date'],
                expected_recovery)
        self.assertTrue(
            pd.isnull(drawdowns.loc[0, 'duration'])) \
            if expected_duration is None else self.assertEqual(
                drawdowns.loc[0, 'duration'], expected_duration)

    def test_drawdown_overlaps(self):
        # Add test to show that drawdowns don't overlap
        # Bug #145 observed for FB stock on the period 2014-10-24 - 2015-03-19
        # Reproduced on SPY data (cached) but need a large number of drawdowns
        spy_rets = utils.get_symbol_rets('SPY',
                                         start='1997-01-01',
                                         end='2004-12-31')
        spy_drawdowns = timeseries.gen_drawdown_table(spy_rets, top=20).sort(
            'peak date')
        # Compare the recovery date of each drawdown with the peak of the next
        # Last pair might contain a NaT if drawdown didn't finish, so ignore it
        pairs = list(zip(spy_drawdowns['recovery date'],
                         spy_drawdowns['peak date'].shift(-1)))[:-1]
        for recovery, peak in pairs:
            self.assertLessEqual(recovery, peak)

    @parameterized.expand([
        (pd.Series(px_list_1 - 1, index=dt), -0.44000000000000011)
    ])
    def test_max_drawdown(self, returns, expected):
        self.assertEqual(timeseries.max_drawdown(returns), expected)

    @parameterized.expand([
        (pd.Series(px_list_1 - 1, index=dt), -0.44000000000000011)
    ])
    def test_max_drawdown_underwater(self, underwater, expected):
        self.assertEqual(timeseries.max_drawdown(underwater), expected)

    @parameterized.expand([
        (pd.Series(px_list_1,
                   index=dt),
         1,
         [(pd.Timestamp('2000-01-03 00:00:00'),
           pd.Timestamp('2000-01-03 00:00:00'),
           pd.Timestamp('2000-01-03 00:00:00'))])
    ])
    def test_top_drawdowns(self, returns, top, expected):
        self.assertEqual(
            timeseries.get_top_drawdowns(
                returns,
                top=top),
            expected)


class TestCumReturns(TestCase):
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    @parameterized.expand([
        (pd.Series([.1, -.05, .1], index=dt),
         pd.Series([1.1, 1.1 * .95, 1.1 * .95 * 1.1], index=dt), 1.),
        (pd.Series([np.nan, -.05, .1], index=dt),
         pd.Series([1., 1. * .95, 1. * .95 * 1.1], index=dt), 1.),
    ])
    def test_expected_result(self, input, expected, starting_value):
        output = timeseries.cum_returns(input, starting_value=starting_value)
        pdt.assert_series_equal(output, expected)


class TestVariance(TestCase):

    @parameterized.expand([
        (1e7, 0.5, 1, 1, -10000000.0)
    ])
    def test_var_cov_var_normal(self, P, c, mu, sigma, expected):
        self.assertEqual(
            timeseries.var_cov_var_normal(
                P,
                c,
                mu,
                sigma),
            expected)


class TestNormalize(TestCase):
    dt = pd.date_range('2000-1-3', periods=8, freq='D')
    px_list = [1.0, 1.2, 1.0, 0.8, 0.7, 0.8, 0.8, 0.8]

    @parameterized.expand([
        (pd.Series(np.array(px_list) * 100, index=dt),
         pd.Series(px_list, index=dt))
    ])
    def test_normalize(self, returns, expected):
        self.assertTrue(timeseries.normalize(returns).equals(expected))


class TestAggregateReturns(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-3',
            periods=500,
            freq='D'))

    @parameterized.expand([
        (simple_rets, 'yearly', [0.3310000000000004, 0.0]),
        (simple_rets[:100], 'monthly', [0.3310000000000004, 0.0, 0.0, 0.0]),
        (simple_rets[:20], 'weekly', [0.3310000000000004, 0.0, 0.0])
    ])
    def test_aggregate_rets(self, returns, convert_to, expected):
        self.assertEqual(
            timeseries.aggregate_returns(
                returns,
                convert_to).values.tolist(),
            expected)


class TestStats(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-3',
            periods=500,
            freq='D'))

    simple_week_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-31',
            periods=500,
            freq='W'))

    simple_month_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-31',
            periods=500,
            freq='M'))

    simple_benchmark = pd.Series(
        [0.03] * 4 + [0] * 496,
        pd.date_range(
            '2000-1-1',
            periods=500,
            freq='D'))
    px_list = np.array(
        [10, -10, 10]) / 100.  # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    px_list_2 = [1.0, 1.2, 1.0, 0.8, 0.7, 0.8, 0.8, 0.8]
    dt_2 = pd.date_range('2000-1-3', periods=8, freq='D')

    @parameterized.expand([
        (simple_rets, utils.DAILY, 0.15500998835658075),
        (simple_week_rets, utils.WEEKLY, 0.030183329386562319),
        (simple_month_rets, utils.MONTHLY, 0.006885932704891129)
    ])
    def test_annual_ret(self, returns, period, expected):
        self.assertEqual(
            timeseries.annual_return(
                returns,
                period=period
            ),
            expected)

    @parameterized.expand([
        (simple_rets, utils.DAILY, 0.12271674212427248),
        (simple_rets, utils.DAILY, 0.12271674212427248),
        (simple_week_rets, utils.WEEKLY, 0.055744909991675112),
        (simple_week_rets, utils.WEEKLY, 0.055744909991675112),
        (simple_month_rets, utils.MONTHLY, 0.026778988562993072),
        (simple_month_rets, utils.MONTHLY, 0.026778988562993072)
    ])
    def test_annual_volatility(self, returns, period, expected):
        self.assertAlmostEqual(
            timeseries.annual_volatility(
                returns,
                period=period
            ),
            expected,
            DECIMAL_PLACES
        )

    @parameterized.expand([
        (simple_rets, 1.2333396776895436)
    ])
    def test_sharpe(self, returns, expected):
        self.assertAlmostEqual(
            timeseries.sharpe_ratio(
                returns),
            expected, DECIMAL_PLACES)

    @parameterized.expand([
        (simple_rets[:5], 2, '[nan, inf, inf, 11.224972160321828, inf]')
    ])
    def test_sharpe_2(self, returns, rolling_sharpe_window, expected):
        self.assertEqual(str(timeseries.rolling_sharpe(
            returns, rolling_sharpe_window).values.tolist()), expected)

    @parameterized.expand([
        (simple_rets, 0.010766923838471554)
    ])
    def test_stability_of_timeseries(self, returns, expected):
        self.assertAlmostEqual(
            timeseries.stability_of_timeseries(returns),
            expected, DECIMAL_PLACES)

    @parameterized.expand([
        (simple_rets[:5], simple_benchmark[:5], 2, 8.024708101613483e-32)
    ])
    def test_beta(self, returns, benchmark_rets, rolling_window, expected):
        self.assertEqual(
            timeseries.rolling_beta(
                returns,
                benchmark_rets,
                rolling_window=rolling_window).values.tolist()[2],
            expected)

    @parameterized.expand([
        (pd.Series(px_list_2,
                   index=dt_2).pct_change().dropna(), -2.3992211554712197)
    ])
    def test_calmar(self, returns, expected):
        self.assertEqual(
            timeseries.calmar_ratio(
                returns),
            expected)

    @parameterized.expand([
        (pd.Series(px_list,
                   index=dt), 0.0, 2.0)
    ])
    def test_omega(self, returns, annual_return_threshhold, expected):
        self.assertEqual(
            timeseries.omega_ratio(
                returns,
                annual_return_threshhold=annual_return_threshhold),
            expected)

    @parameterized.expand([
        (-simple_rets[:5], -12.29634091915152),
        (-simple_rets, -1.2296340919151518),
        (simple_rets, np.inf)
    ])
    def test_sortino(self, returns, expected):
        self.assertAlmostEqual(
            timeseries.sortino_ratio(returns),
            expected, DECIMAL_PLACES)


class TestMultifactor(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-1',
            periods=500,
            freq='D'))
    simple_benchmark_rets = pd.DataFrame(
        pd.Series(
            [0.03] * 4 + [0] * 496,
            pd.date_range(
                '2000-1-1',
                periods=500,
                freq='D')),
        columns=['bm'])

    @parameterized.expand([
        (simple_rets[:4], simple_benchmark_rets[:4], [2.5000000000000004])
    ])
    def test_calc_multifactor(self, returns, factors, expected):
        self.assertEqual(
            timeseries.calc_multifactor(
                returns,
                factors).values.tolist(),
            expected)


class TestCone(TestCase):
    def test_bootstrap_cone_against_linear_cone_normal_returns(self):
        random_seed = 100
        np.random.seed(random_seed)
        days_forward = 200
        cone_stdevs = (1., 1.5, 2.)
        mu = .005
        sigma = .002
        rets = pd.Series(np.random.normal(mu, sigma, 10000))

        midline = np.cumprod(1 + (rets.mean() * np.ones(days_forward)))
        stdev = rets.std() * midline * np.sqrt(np.arange(days_forward)+1)

        normal_cone = pd.DataFrame(columns=pd.Float64Index([]))
        for s in cone_stdevs:
            normal_cone[s] = midline + s * stdev
            normal_cone[-s] = midline - s * stdev

        bootstrap_cone = timeseries.forecast_cone_bootstrap(
            rets, days_forward, cone_stdevs, starting_value=1,
            random_seed=random_seed, num_samples=10000)

        for col, vals in bootstrap_cone.iteritems():
            expected = normal_cone[col].values
            assert_allclose(vals.values, expected, rtol=.005)
