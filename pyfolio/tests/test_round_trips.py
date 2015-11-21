from nose_parameterized import parameterized

from unittest import TestCase

from pandas import (
    DataFrame,
    DatetimeIndex,
    date_range,
    Timedelta,
    read_csv
)
from pandas.util.testing import (assert_frame_equal)

import os
import gzip

from pyfolio.round_trips import (extract_round_trips,
                                 add_closing_transactions)


class RoundTripTestCase(TestCase):
    dates = date_range(start='2015-01-01', freq='D', periods=20)

    @parameterized.expand([
        (DataFrame(data=[[2, 10, 'A'],
                         [-2, 15, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:2]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10, .5,
                          True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'returns',
                            'long', 'symbol'],
                   index=[0])
         ),
        (DataFrame(data=[[2, 10, 'A'],
                         [2, 15, 'A'],
                         [-9, 10, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[2],
                          Timedelta(days=2), -10, -.2,
                          True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'returns',
                            'long', 'symbol'],
                   index=[0])
         ),
        (DataFrame(data=[[2, 10, 'A'],
                         [-4, 15, 'A'],
                         [3, 20, 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10, .5,
                          True, 'A'],
                         [dates[1] + Timedelta(seconds=1), dates[2],
                          Timedelta(days=1) - Timedelta(seconds=1),
                          -10, (-1. / 3),
                          False, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'returns',
                            'long', 'symbol'],
                   index=[0, 1])
         )
    ])
    def test_extract_round_trips(self, transactions, expected):
        round_trips = extract_round_trips(transactions)

        assert_frame_equal(round_trips, expected)

    def test_add_closing_trades(self):
        dates = date_range(start='2015-01-01', periods=20)
        transactions = DataFrame(data=[[2, 10, 'A'],
                                       [-5, 10, 'A'],
                                       [-1, 10, 'B']],
                                 columns=['amount', 'price', 'symbol'],
                                 index=[dates[:3]])
        positions = DataFrame(data=[[20, 10, 0],
                                    [-30, 10, 30],
                                    [-60, 0, 30]],
                              columns=['A', 'B', 'cash'],
                              index=[dates[:3]])

        expected_ix = dates[:3].append(DatetimeIndex([dates[2] +
                                       Timedelta(seconds=1)]))
        expected = DataFrame(data=[[2, 10, 'A'],
                                   [-5, 10, 'A'],
                                   [-1, 10., 'B'],
                                   [3, 20., 'A']],
                             columns=['amount', 'price', 'symbol'],
                             index=expected_ix)

        transactions_closed = add_closing_transactions(positions, transactions)
        assert_frame_equal(transactions_closed, expected)

    def test_txn_pnl_matches_round_trip_pnl(self):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        test_txn = read_csv(gzip.open(
                            __location__ + '/test_data/test_txn.csv.gz'),
                            index_col=0, parse_dates=0)
        test_pos = read_csv(gzip.open(
                            __location__ + '/test_data/test_pos.csv.gz'),
                            index_col=0, parse_dates=0)

        transactions_closed = add_closing_transactions(test_pos, test_txn)
        transactions_closed['txn_dollars'] = transactions_closed.amount * \
            -1. * transactions_closed.price
        round_trips = extract_round_trips(transactions_closed)

        self.assertAlmostEqual(round_trips.pnl.sum(),
                               transactions_closed.txn_dollars.sum())
