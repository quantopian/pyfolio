from unittest import TestCase

from pandas import (
    DataFrame,
    date_range,
)
from pandas.util.testing import assert_frame_equal
from numpy import (
    absolute,
    arange,
    zeros_like,
)

from pyfolio.pos import get_portfolio_alloc


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
