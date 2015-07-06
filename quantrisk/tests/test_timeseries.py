from unittest import TestCase
from nose_parameterized import parameterized

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from .. import timeseries

class TestDrawdown(TestCase):
    px_list_1 = [100, 120, 100, 80, 70, 80, 120, 130] # Simple
    px_list_2 = [100, 120, 100, 80, 70, 80, 80, 80] # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=8, freq='D')

    @parameterized.expand([
        (pd.Series(px_list_1, index=dt), pd.Timestamp('2000-1-4'), pd.Timestamp('2000-1-7'), pd.Timestamp('2000-1-9')),
        (pd.Series(px_list_2, index=dt), pd.Timestamp('2000-1-4'), pd.Timestamp('2000-1-7'), None)
    ])
    def test_get_max_draw_down(self, px, expected_peak, expected_valley, expected_recovery):
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_draw_down(rets)
        # Need to use isnull because the result can be NaN, NaT, etc.
        self.assertTrue(pd.isnull(peak)) if expected_peak is None else self.assertEqual(peak, expected_peak)
        self.assertTrue(pd.isnull(valley)) if expected_valley is None else self.assertEqual(valley, expected_valley)
        self.assertTrue(pd.isnull(recovery)) if expected_recovery is None else self.assertEqual(recovery, expected_recovery)

    @parameterized.expand([
        (pd.Series(px_list_2, index=dt), pd.Timestamp('2000-1-4'), pd.Timestamp('2000-1-7'), None, None)
    ])
    def test_gen_drawdown_table_end_in_draw_down(self, px, expected_peak, expected_valley, expected_recovery, expected_duration):
        rets = px.pct_change().iloc[1:]

        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'peak date'])) if expected_peak is None else self.assertEqual(drawdowns.loc[0, 'peak date'], expected_peak)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'valley date'])) if expected_valley is None else self.assertEqual(drawdowns.loc[0, 'valley date'], expected_valley)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'recovery date'])) if expected_recovery is None else self.assertEqual(drawdowns.loc[0, 'recovery date'], expected_recovery)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'duration'])) if expected_duration is None else self.assertEqual(drawdowns.loc[0, 'duration'], expected_duration)

    @parameterized.expand([
        (pd.Series(px_list_1, index=dt), True, -0.41666666666666669)
    ])
    def test_max_drawdown(self, df_rets, input_is_NAV, expected):
        self.assertEqual(timeseries.max_drawdown(df_rets, input_is_NAV), expected)




class TestCumReturns(TestCase):
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    @parameterized.expand([
        (pd.Series([.1, -.05, .1], index=dt), pd.Series([1.1, 1.1*.95, 1.1*.95*1.1], index=dt), 1.),
        (pd.Series([np.nan, -.05, .1], index=dt), pd.Series([1., 1.*.95, 1.*.95*1.1], index=dt), 1.),
    ])
    def test_expected_result(self, input, expected, starting_value):
        output = timeseries.cum_returns(input, starting_value=starting_value)
        pdt.assert_series_equal(output, expected)

class TestVariance(TestCase):
    @parameterized.expand([
       (1e7, 0.5, 1, 1, -10000000.0)
    ])
    def test_var_cov_var_normal(self, P, c, mu, sigma, expected):
        self.assertEqual(timeseries.var_cov_var_normal(P, c, mu, sigma), expected)

class TestNormalize(TestCase):
    dt = pd.date_range('2000-1-3', periods=8, freq='D')
    px_list = [1.0, 1.2, 1.0, 0.8, 0.7, 0.8, 0.8, 0.8]

    @parameterized.expand([
       (pd.Series(np.array(px_list)*100, index=dt), pd.Series(px_list, index=dt))
    ])
    def test_normalize(self, df, expected):
        self.assertTrue(timeseries.normalize(df).equals(expected))

class TestAggregateReturns(TestCase):
    simple_rets = pd.Series([0.1]*3+[0]*497, pd.date_range('2000-1-3', periods=500, freq='D'))
    @parameterized.expand([
        (simple_rets, 'yearly', [0.3310000000000004, 0.0]),
        (simple_rets[:100], 'monthly', [0.3310000000000004, 0.0, 0.0, 0.0]),
        (simple_rets[:20], 'weekly', [0.3310000000000004, 0.0, 0.0])
    ])
    def test_aggregate_rets(self, df_rets, convert_to, expected):
        self.assertEqual(timeseries.aggregate_returns(df_rets, convert_to).values.tolist(), expected)


class TestStats(TestCase):
    simple_rets = pd.Series([0.1]*3+[0]*497, pd.date_range('2000-1-3', periods=500, freq='D'))

    @parameterized.expand([
        (simple_rets, True, 'calendar', -84.0),
        (simple_rets, True, 'compound', -1.0),
        (simple_rets, False, 'calendar', 0.10584000000000014),
        (simple_rets, False, 'compound', 0.16317653888658334)
    ])
    def test_annual_ret(self, df_rets, inputIsNAV, style, expected):
        self.assertEqual(timeseries.annual_return(df_rets, inputIsNAV=inputIsNAV, style=style), expected)

    @parameterized.expand([
        (simple_rets, True, 9.1651513899116814),
        (simple_rets, False, 0.12271674212427248)
    ])
    def test_annual_volatility(self, df_rets, inputIsNAV, expected):
        self.assertEqual(timeseries.annual_volatility(df_rets, inputIsNAV=inputIsNAV), expected)

    @parameterized.expand([
        (simple_rets, True, 'calendar', -84.0),
        #(simple_rets[:30], False, 'compound', x)
    ])
    def test_calmer(self, df_rets, inputIsNAV, returns_style, expected):
        self.assertEqual(timeseries.calmer_ratio(df_rets, inputIsNAV=inputIsNAV, returns_style=returns_style), expected)

    
    @parameterized.expand([
        (simple_rets, True, 'calendar', -9.1651513899116779),
        (simple_rets, True, 'compound', -0.10910894511799617)
    ])
    def test_sharpe(self, df_rets, inputIsNAV, returns_style, expected):
        self.assertEqual(timeseries.sharpe_ratio(df_rets, inputIsNAV=inputIsNAV, returns_style=returns_style), expected)

    @parameterized.expand([
        (simple_rets, False, True, 0.017892071568286205)
    ])
    def test_stability_of_timeseries(self, df_rets, logValue, inputIsNAV, expected):
        self.assertEqual(timeseries.stability_of_timeseries(df_rets, logValue=logValue, inputIsNAV=inputIsNAV), expected)

    """"
    @parameterized.expand([
        (pd.Series([0.997357, 1.006424, 0.993907], pd.date_range('2009-2-2', periods=3, freq='D')), pd.Series([0.992269, 0.995216, 0.994577], pd.date_range('2013-10-7', periods=3, freq='D')), 'scale', True, 0.017892071568286205)
    ])
    def test_kde(self, bt_ts, oos_ts, transform_style, return_zero_if_exception, expected):
        self.assertEqual(timeseries.out_of_sample_vs_in_sample_returns_kde(bt_ts, oos_ts, transform_style=transform_style, return_zero_if_exception=return_zero_if_exception), expected)
    """

class TestMultifactor(TestCase):
    simple_rets = pd.Series([0.1]*3+[0]*497, pd.date_range('2000-1-1', periods=500, freq='D'))
    """
    @parameterized.expand([
        (simple_rets, simple_rets, 0.5),
        (simple_rets[:100], simple_rets[:100], 0.5)
    ])
    def test_calc_multifactor(self, df_rets, factors, expected):
        self.assertEqual(timeseries.calc_multifactor(df_rets, factors), expected)
    """