from unittest import TestCase
from parameterized import parameterized
from datetime import datetime
import pandas as pd

from pandas.testing import assert_frame_equal, assert_series_equal

from pyfolio.capacity import (
    days_to_liquidate_positions,
    get_max_days_to_liquidate_by_ticker,
    get_low_liquidity_transactions,
    daily_txns_with_bar_data,
    apply_slippage_penalty,
)


class CapacityTestCase(TestCase):
    dates = pd.date_range(start="2015-01-01", freq="D", periods=3)

    positions = pd.DataFrame(
        [[1.0, 3.0, 0.0], [0.0, 1.0, 1.0], [3.0, 0.0, 1.0]],
        columns=["A", "B", "cash"],
        index=dates,
    )

    columns = ["sid", "amount", "price", "symbol"]
    transactions = pd.DataFrame(
        data=[[1, 100000, 10, "A"]] * len(dates), columns=columns, index=dates
    )

    volume = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], columns=["A", "B"], index=dates
    )

    volume.index.name = "dt"
    volume *= 1000000
    volume["market_data"] = "volume"

    price = pd.DataFrame(
        [[1.0, 1.0]] * len(dates), columns=["A", "B"], index=dates
    )
    price.index.name = "dt"
    price["market_data"] = "price"

    market_data = (
        pd.concat([volume, price])
        .reset_index()
        .set_index(["dt", "market_data"])
    )

    def test_days_to_liquidate_positions(self):
        dtlp = days_to_liquidate_positions(
            self.positions,
            self.market_data,
            max_bar_consumption=1,
            capital_base=1e6,
            mean_volume_window=1,
        )

        expected = pd.DataFrame(
            [[0.0, 0.5 / 3], [0.75 / 2, 0.0]],
            columns=["A", "B"],
            index=self.dates[1:],
        )
        assert_frame_equal(dtlp, expected)

    def test_get_max_days_to_liquidate_by_ticker(self):
        mdtl = get_max_days_to_liquidate_by_ticker(
            self.positions,
            self.market_data,
            max_bar_consumption=1,
            capital_base=1e6,
            mean_volume_window=1,
        )

        expected = pd.DataFrame(
            [
                [datetime(2015, 1, 3), 0.75 / 2, 75.0],
                [datetime(2015, 1, 2), 0.5 / 3, 50.0],
            ],
            columns=["date", "days_to_liquidate", "pos_alloc_pct"],
            index=["A", "B"],
        )
        expected.index.name = "symbol"

        assert_frame_equal(mdtl, expected)

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    [
                        [datetime(2015, 1, 1), 100.0],
                        [datetime(2015, 1, 2), 100.0],
                    ],
                    columns=["date", "max_pct_bar_consumed"],
                    index=["A", "B"],
                ),
                None,
            ),
            (
                pd.DataFrame(
                    [[datetime(2015, 1, 3), 100 / 3]],
                    columns=["date", "max_pct_bar_consumed"],
                    index=["A"],
                ),
                1,
            ),
        ]
    )
    def test_get_low_liquidity_transactions(self, expected, last_n_days):
        txn_daily = pd.DataFrame(
            data=[
                [1, 1000000, 1, "A"],
                [2, 2000000, 1, "B"],
                [1, 1000000, 1, "A"],
            ],
            columns=["sid", "amount", "price", "symbol"],
            index=self.dates,
        )

        llt = get_low_liquidity_transactions(
            txn_daily, self.market_data, last_n_days=last_n_days
        )
        expected.index.name = "symbol"
        assert_frame_equal(llt, expected)

    def test_daily_txns_with_bar_data(self):
        daily_txn = daily_txns_with_bar_data(
            self.transactions, self.market_data
        )

        columns = ["symbol", "amount", "price", "volume"]
        expected = pd.DataFrame(
            data=[
                ["A", 100000, 1.0, 1000000.0],
                ["A", 100000, 1.0, 2000000.0],
                ["A", 100000, 1.0, 3000000.0],
            ],
            columns=columns,
            index=self.dates,
        )

        assert_frame_equal(daily_txn, expected)

    @parameterized.expand(
        [
            (1000000, 1, [0.9995, 0.9999375, 0.99998611]),
            (10000000, 1, [0.95, 0.99375, 0.998611]),
            (100000, 1, [0.999995, 0.999999375, 0.9999998611]),
            (1000000, 0.1, [0.99995, 0.99999375, 0.999998611]),
        ]
    )
    def test_apply_slippage_penalty(
        self, starting_base, impact, expected_adj_returns
    ):
        returns = pd.Series([1.0, 1.0, 1.0], index=self.dates)
        daily_txn = daily_txns_with_bar_data(
            self.transactions, self.market_data
        )

        adj_returns = apply_slippage_penalty(
            returns, daily_txn, starting_base, 1000000, impact=impact
        )
        expected_adj_returns = pd.Series(
            expected_adj_returns, index=self.dates
        )

        assert_series_equal(adj_returns, expected_adj_returns)
