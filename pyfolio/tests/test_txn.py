from unittest import TestCase

from pandas import (
    Series,
    DataFrame,
    date_range
)
from pandas.util.testing import (assert_series_equal)

from pyfolio.txn import (get_turnover,
                         adjust_returns_for_slippage)


class TransactionsTestCase(TestCase):

    def test_get_turnover(self):
        """
        Tests turnover using a 20 day period.

        With no transactions the turnover should be 0.

        with 100% of the porfolio value traded each day
        the daily turnover rate should be 0.5.

        For monthly turnover it should be the sum
        of the daily turnovers because 20 days < 1 month.

        e.g (20 days) * (0.5 daily turn) = 10x monthly turnover rate.
        """
        dates = date_range(start='2015-01-01', freq='D', periods=20)

        positions = DataFrame([[0.0, 10.0]]*len(dates),
                              columns=[0, 'cash'], index=dates)

        transactions = DataFrame(data=[],
                                 columns=['sid', 'amount', 'price', 'symbol'],
                                 index=dates)

        # Test with no transactions
        expected = Series([0.0]*len(dates), index=dates)
        result = get_turnover(positions, transactions)
        assert_series_equal(result, expected)

        # Monthly freq
        index = date_range('01-01-2015', freq='M', periods=1)
        expected = Series([0.0], index=index)
        result = get_turnover(positions, transactions, period='M')
        assert_series_equal(result, expected)

        transactions = DataFrame(data=[[1, 1, 10, 'A']]*len(dates),
                                 columns=['sid', 'amount', 'price', 'symbol'],
                                 index=dates)

        expected = Series([0.5]*len(dates), index=dates)
        result = get_turnover(positions, transactions)
        assert_series_equal(result, expected)

        # Monthly freq: should be the sum of the daily freq
        result = get_turnover(positions, transactions, period='M')
        expected = Series([10.0], index=index)
        assert_series_equal(result, expected)

    def test_adjust_returns_for_slippage(self):
        dates = date_range(start='2015-01-01', freq='D', periods=20)

        positions = DataFrame([[0.0, 10.0]]*len(dates),
                              columns=[0, 'cash'], index=dates)

        # 100% total, 50% average daily turnover
        transactions = DataFrame(data=[[1, 1, 10, 'A']]*len(dates),
                                 columns=['sid', 'amount', 'price', 'symbol'],
                                 index=dates)

        returns = Series([0.05]*len(dates), index=dates)
        # 0.001% slippage per dollar traded
        slippage_bps = 10
        expected = Series([0.049]*len(dates), index=dates)

        turnover = get_turnover(positions, transactions, average=False)
        result = adjust_returns_for_slippage(returns, turnover, slippage_bps)

        assert_series_equal(result, expected)
