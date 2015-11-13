from nose_parameterized import parameterized

from unittest import TestCase

from pandas import (
    Series,
    DataFrame,
    date_range,
    Timedelta
)
from pandas.util.testing import (assert_series_equal,
                                 assert_frame_equal)

from pyfolio.txn import (get_turnover,
                         adjust_returns_for_slippage,
                         extract_round_trips,
                         add_closing_transactions)


class TransactionsTestCase(TestCase):
    dates = date_range(start='2015-01-01', freq='D', periods=20)

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
        result = get_turnover(transactions, positions)
        assert_series_equal(result, expected)

        # Monthly freq
        index = date_range('01-01-2015', freq='M', periods=1)
        expected = Series([0.0], index=index)
        result = get_turnover(transactions, positions, period='M')
        assert_series_equal(result, expected)

        transactions = DataFrame(data=[[1, 1, 10, 'A']]*len(dates),
                                 columns=['sid', 'amount', 'price', 'symbol'],
                                 index=dates)

        expected = Series([0.5]*len(dates), index=dates)
        result = get_turnover(transactions, positions)
        assert_series_equal(result, expected)

        # Monthly freq: should be the sum of the daily freq
        result = get_turnover(transactions, positions, period='M')
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

        turnover = get_turnover(transactions, positions, average=False)
        result = adjust_returns_for_slippage(returns, turnover, slippage_bps)

        assert_series_equal(result, expected)

    @parameterized.expand([
        (DataFrame(data=[[2, 10, 'A'],
                         [-2, 15, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:2]),
         DataFrame(data=[[dates[0], dates[1], Timedelta(days=1),
                          10, True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'long', 'symbol'],
                   index=[0])
         ),
        (DataFrame(data=[[2, 10, 'A'],
                         [2, 15, 'A'],
                         [-9, 10, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[2], Timedelta(days=2),
                          -10, True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'long', 'symbol'],
                   index=[0])
         ),
        (DataFrame(data=[[2, 10, 'A'],
                         [-4, 15, 'A'],
                         [3, 20, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[1], Timedelta(days=1),
                          10, True, 'A'],
                         [dates[1] + Timedelta(seconds=1), dates[2],
                          Timedelta(days=1) - Timedelta(seconds=1),
                          -10, False, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'long', 'symbol'],
                   index=[0, 1])
         )
    ])
    def test_extract_round_trips(self, transactions, expected):
        round_trips = extract_round_trips(transactions)
        assert_frame_equal(round_trips, expected)

    def test_add_closing_trades(self):
        dates = date_range(start='2015-01-01', freq='D', periods=20)
        transactions = DataFrame(data=[[2, 10, 'A'],
                                       [-5, 10, 'A']],
                                 columns=['amount', 'price', 'symbol'],
                                 index=[dates[:2]])
        positions = DataFrame(data=[[20, 0],
                                    [-30, 30],
                                    [-60, 30]],
                              columns=['A', 'cash'],
                              index=[dates[:3]])
        expected = DataFrame(data=[[2, 10, 'A'],
                                   [-5, 10, 'A'],
                                   [3, 20., 'A']],
                             columns=['amount', 'price', 'symbol'],
                             index=[dates[:3]])

        transactions_closed = add_closing_transactions(positions, transactions)
        assert_frame_equal(transactions_closed, expected)
