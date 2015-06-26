from unittest import TestCase
from nose_parameterized import parameterized

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from .. import timeseries

class TestDrawdown(TestCase):

    px_list_1 = [100, 120, 100, 80, 70, 80, 120, 130] # Simple
    px_list_2 = [100, 120, 100, 80, 70, 80, 80, 80] # Ends in drawdown
    dt = pd.date_range('2000-1-1', periods=8, freq='D')

    @parameterized.expand([
        (pd.Series(px_list_1, index=dt), pd.Timestamp('2000-1-2'), pd.Timestamp('2000-1-5'), pd.Timestamp('2000-1-7')),
        (pd.Series(px_list_2, index=dt), pd.Timestamp('2000-1-2'), pd.Timestamp('2000-1-5'), None)
    ])
    def test_get_max_draw_down(self, px, expected_peak, expected_valley, expected_recovery):
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_draw_down(rets)
        # Need to use isnull because the result can be NaN, NaT, etc.
        self.assertTrue(pd.isnull(peak)) if expected_peak is None else self.assertEqual(peak, expected_peak)
        self.assertTrue(pd.isnull(valley)) if expected_valley is None else self.assertEqual(valley, expected_valley)
        self.assertTrue(pd.isnull(recovery)) if expected_recovery is None else self.assertEqual(recovery, expected_recovery)

    @parameterized.expand([
        (pd.Series(px_list_2, index=dt), pd.Timestamp('2000-1-2'), pd.Timestamp('2000-1-5'), None, None)
    ])
    def test_gen_drawdown_table_end_in_draw_down(self, px, expected_peak, expected_valley, expected_recovery, expected_duration):
        rets = px.pct_change().iloc[1:]

        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'peak date'])) if expected_peak is None else self.assertEqual(drawdowns.loc[0, 'peak date'], expected_peak)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'valley date'])) if expected_valley is None else self.assertEqual(drawdowns.loc[0, 'valley date'], expected_valley)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'recovery date'])) if expected_recovery is None else self.assertEqual(drawdowns.loc[0, 'recovery date'], expected_recovery)
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'duration'])) if expected_duration is None else self.assertEqual(drawdowns.loc[0, 'duration'], expected_duration)

class TestCumReturns(TestCase):
    dt = pd.date_range('2000-1-1', periods=3, freq='D')

    @parameterized.expand([
        (pd.Series([.1, -.05, .1], index=dt), pd.Series([1.1, 1.1*.95, 1.1*.95*1.1], index=dt)),
        (pd.Series([np.nan, -.05, .1], index=dt), pd.Series([1., 1.*.95, 1.*.95*1.1], index=dt)),
    ])
    def test_expected_result(self, input, expected):
        output = timeseries.cum_returns(input)
        pdt.assert_series_equal(output, expected)
