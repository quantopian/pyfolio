from nose_parameterized import parameterized

from unittest import TestCase

from pandas import (
    DataFrame,
    date_range,
    Timedelta
)
from pyfolio.round_trips import (extract_round_trips,
                                 add_closing_transactions)

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