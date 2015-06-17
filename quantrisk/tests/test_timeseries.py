from unittest import TestCase

import numpy as np
import pandas as pd

from .. import timeseries

class TestDrawdown(TestCase):
    def test_get_max_draw_down_simple(self):
        px_list = [100, 120, 100, 80, 70, 80, 120, 130]
        dt = pd.date_range('2000-1-1', periods=len(px_list), freq='D')
        px = pd.Series(px_list, index=dt)
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_draw_down(rets)
        self.assertEqual(peak, pd.Timestamp('2000-1-2'))
        self.assertEqual(valley, pd.Timestamp('2000-1-5'))
        self.assertEqual(recovery, pd.Timestamp('2000-1-7'))

    def test_get_max_draw_down_ends_in_draw_down(self):
        px_list = [100, 120, 100, 80, 70, 80, 80, 80]
        dt = pd.date_range('2000-1-1', periods=len(px_list), freq='D')
        px = pd.Series(px_list, index=dt)
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_draw_down(rets)
        self.assertEqual(peak, pd.Timestamp('2000-1-2'))
        self.assertEqual(valley, pd.Timestamp('2000-1-5'))
        self.assertTrue(pd.isnull(recovery))

    def test_gen_drawdown_table_end_in_draw_down(self):
        px_list = [100, 120, 100, 80, 70, 80, 80, 80]
        dt = pd.date_range('2000-1-1', periods=len(px_list), freq='D')
        px = pd.Series(px_list, index=dt)
        rets = px.pct_change().iloc[1:]

        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertEqual(drawdowns.loc[0, 'peak date'], pd.Timestamp('2000-1-2'))
        self.assertEqual(drawdowns.loc[0, 'valley date'], pd.Timestamp('2000-1-5'))
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'recovery date']))
        self.assertTrue(pd.isnull(drawdowns.loc[0, 'duration']))
