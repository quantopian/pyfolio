from unittest import TestCase
from collections import OrderedDict

from pandas import (
    Series,
    DataFrame,
    date_range,
    Timestamp
)
from pandas.util.testing import (assert_frame_equal,
                                 assert_series_equal)
from numpy import (
    absolute,
    arange,
    zeros_like,
)

from pyfolio.pos import (get_portfolio_alloc,
                         extract_pos,
                         get_turnover)


class PositionsTestCase(TestCase):

    def test_get_portfolio_alloc(self):
        raw_data = arange(15, dtype=float).reshape(5, 3)
        # Make the first column negative to test absolute magnitudes.
        raw_data[:, 0] *= -1

        frame = DataFrame(
            raw_data,
            index=date_range('01-01-2015', freq='D', periods=5),
            columns=['A', 'B', 'C']
        )

        result = get_portfolio_alloc(frame)
        expected_raw = zeros_like(raw_data)
        for idx, row in enumerate(raw_data):
            expected_raw[idx] = row / absolute(row).sum()

        expected = DataFrame(
            expected_raw,
            index=frame.index,
            columns=frame.columns,
        )

        assert_frame_equal(result, expected)

    def test_extract_pos(self):
        index_dup = [Timestamp('2015-06-08', tz='UTC'),
                     Timestamp('2015-06-08', tz='UTC'),
                     Timestamp('2015-06-09', tz='UTC'),
                     Timestamp('2015-06-09', tz='UTC')]
        index = [Timestamp('2015-06-08', tz='UTC'),
                 Timestamp('2015-06-09', tz='UTC')]

        positions = DataFrame(
            {'amount': [100., 200., 300., 400.],
             'last_sale_price': [10., 20., 30., 40.],
             'sid': [1, 2, 1, 2]},
            index=index_dup
        )
        cash = Series([100., 200.], index=index)

        result = extract_pos(positions, cash)

        expected = DataFrame(OrderedDict([
            (1, [100.*10., 300.*30.]),
            (2, [200.*20., 400.*40.]),
            ('cash', [100., 200.])]),
            index=index
        )
        expected.index.name = 'index'
        expected.columns.name = 'sid'

        assert_frame_equal(result, expected)

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
        transactions = DataFrame([[0, 0]]*len(dates),
                                 columns=['txn_volume', 'txn_shares'],
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

        # Test with 0.5 daily turnover
        transactions = DataFrame([[10.0, 0]]*len(dates),
                                 columns=['txn_volume', 'txn_shares'],
                                 index=dates)

        expected = Series([0.5]*len(dates), index=dates)
        result = get_turnover(transactions, positions)
        assert_series_equal(result, expected)

        # Monthly freq: should be the sum of the daily freq
        result = get_turnover(transactions, positions, period='M')
        expected = Series([10.0], index=index)
        assert_series_equal(result, expected)
