from __future__ import division

from unittest import TestCase
from nose_parameterized import parameterized

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from .. import timeseries


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

    @parameterized.expand([
        (pd.Series(px_list_1 - 1, index=dt), -0.44000000000000011)
    ])
    def test_max_drawdown(self, df_rets, expected):
        self.assertEqual(timeseries.max_drawdown(df_rets), expected)

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
    def test_top_drawdowns(self, df_rets, top, expected):
        self.assertEqual(
            timeseries.get_top_drawdowns(
                df_rets,
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
    def test_normalize(self, df, expected):
        self.assertTrue(timeseries.normalize(df).equals(expected))


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
    def test_aggregate_rets(self, df_rets, convert_to, expected):
        self.assertEqual(
            timeseries.aggregate_returns(
                df_rets,
                convert_to).values.tolist(),
            expected)


class TestStats(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-3',
            periods=500,
            freq='D'))
    simple_benchmark = pd.Series(
        [0.03] * 4 + [0] * 496,
        pd.date_range(
            '2000-1-1',
            periods=500,
            freq='D'))
    px_list = np.array(
        [10, -10, 10]) / 100.  # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    @parameterized.expand([
        (simple_rets, 'calendar', 0.10584000000000014),
        (simple_rets, 'compound', 0.16317653888658334),
        (simple_rets, 'calendar', 0.10584000000000014),
        (simple_rets, 'compound', 0.16317653888658334)
    ])
    def test_annual_ret(self, df_rets, style, expected):
        self.assertEqual(
            timeseries.annual_return(
                df_rets,
                style=style),
            expected)

    @parameterized.expand([
        (simple_rets, 0.12271674212427248),
        (simple_rets, 0.12271674212427248)
    ])
    def test_annual_volatility(self, df_rets, expected):
        self.assertEqual(timeseries.annual_volatility(df_rets), expected)

    @parameterized.expand([
        (simple_rets, 'calendar', 0.8624740045072119),
        (simple_rets, 'compound', 1.3297007080039505)
    ])
    def test_sharpe(self, df_rets, returns_style, expected):
        self.assertEqual(
            timeseries.sharpe_ratio(
                df_rets,
                returns_style=returns_style),
            expected)

    @parameterized.expand([
        (simple_rets[:5], 2, '[nan, inf, inf, 11.224972160321828, inf]')
    ])
    def test_sharpe_2(self, df_rets, rolling_sharpe_window, expected):
        self.assertEqual(str(timeseries.rolling_sharpe(
            df_rets, rolling_sharpe_window).values.tolist()), expected)

    @parameterized.expand([
        (simple_rets, True, 0.010766923838471554)
    ])
    def test_stability_of_timeseries(self, df_rets, logValue, expected):
        self.assertEqual(
            timeseries.stability_of_timeseries(
                df_rets,
                logValue=logValue),
            expected)

    @parameterized.expand([
        (simple_rets[:5], simple_benchmark[:5], 2, 8.024708101613483e-32)
    ])
    def test_beta(self, df_rets, benchmark_rets, rolling_window, expected):
        self.assertEqual(
            timeseries.rolling_beta(
                df_rets,
                benchmark_rets,
                rolling_window=rolling_window).values.tolist()[2],
            expected)

    @parameterized.expand([
        (pd.Series(px_list,
                   index=dt), 'calendar', -8.3999999999999559),
        (pd.Series(px_list,
                   index=dt), 'arithmetic', 84.000000000000014)
    ])
    def test_calmar(self, df_rets, returns_style, expected):
        self.assertEqual(
            timeseries.calmar_ratio(
                df_rets,
                returns_style=returns_style),
            expected)

    @parameterized.expand([
        (pd.Series(px_list,
                   index=dt), 0.0, 2.0)
    ])
    def test_omega(self, df_rets, annual_return_threshhold, expected):
        self.assertEqual(
            timeseries.omega_ratio(
                df_rets,
                annual_return_threshhold=annual_return_threshhold),
            expected)

    @parameterized.expand([
        (-simple_rets[:5], 'calendar', -458003439.10738045),
        (-simple_rets[:5], 'arithmetic', -723163324.90639055)
    ])
    def test_sortino(self, df_rets, returns_style, expected):
        self.assertEqual(
            timeseries.sortino_ratio(
                df_rets,
                returns_style=returns_style),
            expected)


class TestMultifactor(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-1',
            periods=500,
            freq='D'))
    simple_benchmark_df = pd.DataFrame(
        pd.Series(
            [0.03] * 4 + [0] * 496,
            pd.date_range(
                '2000-1-1',
                periods=500,
                freq='D')),
        columns=['bm'])

    @parameterized.expand([
        (simple_rets[:4], simple_benchmark_df[:4], [2.5000000000000004])
    ])
    def test_calc_multifactor(self, df_rets, factors, expected):
        self.assertEqual(
            timeseries.calc_multifactor(
                df_rets,
                factors).values.tolist(),
            expected)

    @parameterized.expand([
        (simple_rets[:5],
         simple_benchmark_df[:5],
         2,
         [0.09991008092716558,
            0.002997302427814967])
    ])
    def test_multifactor_beta(
            self, df_rets, benchmark_df, rolling_window, expected):
        self.assertEqual(
            timeseries.rolling_multifactor_beta(
                df_rets,
                benchmark_df,
                rolling_window=rolling_window).values.tolist()[2],
            expected)
