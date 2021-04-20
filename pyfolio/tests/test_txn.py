from unittest import TestCase

import pandas as pd

from pandas.testing import assert_series_equal

from pyfolio.txn import get_turnover, adjust_returns_for_slippage


class TransactionsTestCase(TestCase):
    def test_get_turnover(self):
        """
        Tests turnover using a 20 day period.

        With no transactions, the turnover should be 0.

        with 200% of the AGB traded each day, the daily
        turnover rate should be 2.0.
        """
        dates = pd.date_range(start="2015-01-01", freq="D", periods=20)

        # In this test, there is one sid (0) and a cash column
        positions = pd.DataFrame(
            [[10.0, 10.0]] * len(dates), columns=[0, "cash"], index=dates
        )

        # Set every other non-cash position to 40
        positions[0][::2] = 40

        transactions = pd.DataFrame(
            data=[], columns=["sid", "amount", "price", "symbol"], index=dates
        )

        # Test with no transactions
        expected = pd.Series([0.0] * len(dates), index=dates)
        result = get_turnover(positions, transactions).asfreq("D")
        assert_series_equal(result, expected)

        transactions = pd.DataFrame(
            data=[[1, 1, 10, 0]] * len(dates) + [[2, -1, 10, 0]] * len(dates),
            columns=["sid", "amount", "price", "symbol"],
            index=dates.append(dates),
        ).sort_index()

        # Turnover is more on day 1, because the day 0 AGB is set to zero
        # in get_turnover. On most days, we get 0.8 because we have 20
        # transacted and mean(10, 40) = 25, so 20/25.
        expected = pd.Series([1.0] + [0.8] * (len(dates) - 1), index=dates)
        result = get_turnover(positions, transactions).asfreq("D")

        assert_series_equal(result, expected)

        # Test with denominator = 'portfolio_value'
        result = get_turnover(
            positions, transactions, denominator="portfolio_value"
        ).asfreq("D")

        # Our portfolio value alternates between $20 and $50 so turnover
        # should alternate between 20/20 = 1.0 and 20/50 = 0.4.
        expected = pd.Series(
            [0.4, 1.0] * (int((len(dates) - 1) / 2) + 1), index=dates
        )

        assert_series_equal(result, expected)

    def test_adjust_returns_for_slippage(self):
        dates = pd.date_range(start="2015-01-01", freq="D", periods=20)

        positions = pd.DataFrame(
            [[0.0, 10.0]] * len(dates), columns=[0, "cash"], index=dates
        )

        # 100% total, 50% average daily turnover
        transactions = pd.DataFrame(
            data=[[1, 1, 10, "A"]] * len(dates),
            columns=["sid", "amount", "price", "symbol"],
            index=dates,
        )

        returns = pd.Series([0.05] * len(dates), index=dates)
        # 0.001% slippage per dollar traded
        slippage_bps = 10
        expected = pd.Series([0.049] * len(dates), index=dates)

        result = adjust_returns_for_slippage(
            returns, positions, transactions, slippage_bps
        )

        assert_series_equal(result, expected)
