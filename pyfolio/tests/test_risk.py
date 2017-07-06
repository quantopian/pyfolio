from unittest import TestCase
from nose_parameterized import parameterized
import os
import gzip

import pandas as pd
from pandas import read_csv
from pyfolio.utils import to_utc

from pandas.util.testing import assert_frame_equal, assert_series_equal

from pyfolio.risk import (compute_style_factor_exposures,
                          compute_sector_exposures,
                          compute_cap_exposures,
                          compute_volume_exposures)


class RiskTestCase(TestCase):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    test_pos = to_utc(read_csv(
        gzip.open(__location__ + '/test_data/test_pos.csv.gz'),
        index_col=0, parse_dates=True))
    test_pos.columns = [351, 1419, 1787, 25317, 3321, 3951, 4922, 'cash']

    test_txn = to_utc(read_csv(
        gzip.open(
            __location__ + '/test_data/test_txn.csv.gz'),
        index_col=0, parse_dates=True))
    test_sectors = to_utc(read_csv(
        __location__ + '/test_data/test_sectors.csv',
        index_col=0, parse_dates=True))
    expected_sectors_longed = to_utc(read_csv(
        __location__ + '/test_data/expected_sectors_longed.csv',
        index_col=0, parse_dates=True))
    expected_sectors_shorted = to_utc(read_csv(
        __location__ + '/test_data/expected_sectors_shorted.csv',
        index_col=0, parse_dates=True))
    expected_sectors_grossed = to_utc(read_csv(
        __location__ + '/test_data/expected_sectors_grossed.csv',
        index_col=0, parse_dates=True))
    test_caps = to_utc(read_csv(
        __location__ + '/test_data/test_caps.csv',
        index_col=0, parse_dates=True))
    expected_caps_longed = to_utc(read_csv(
        __location__ + '/test_data/expected_caps_longed.csv',
        index_col=0, parse_dates=True))
    expected_caps_shorted = to_utc(read_csv(
        __location__ + '/test_data/expected_caps_shorted.csv',
        index_col=0, parse_dates=True))
    expected_caps_grossed = to_utc(read_csv(
        __location__ + '/test_data/expected_caps_grossed.csv',
        index_col=0, parse_dates=True))
    expected_caps_netted = to_utc(read_csv(
        __location__ + '/test_data/expected_caps_netted.csv',
        index_col=0, parse_dates=True))
    test_shares_held = to_utc(read_csv(
        __location__ + '/test_data/test_shares_held.csv',
        index_col=0, parse_dates=True))
    test_volumes = to_utc(read_csv(
        __location__ + '/test_data/test_volumes.csv',
        index_col=0, parse_dates=True))
    expected_volumes = to_utc(read_csv(
        __location__ + '/test_data/expected_volumes.csv',
        index_col=0, parse_dates=True))

    test_dict = {}
    styles = ['LT_MOMENTUM', 'LMCAP', 'VLTY', 'MACDSignal']
    for style in styles:
        df = to_utc(read_csv(
            __location__ + '/test_data/test_{}.csv'.format(style),
            index_col=0, parse_dates=True))
        test_dict.update({style: df})
    test_styles = pd.Panel()
    test_styles = test_styles.from_dict(test_dict)

    expected_styles = to_utc(read_csv(
        __location__ + '/test_data/expected_styles.csv',
        index_col=0, parse_dates=True))

    @parameterized.expand([
        (test_pos, test_styles, expected_styles)
    ])
    def test_compute_style_factor_exposures(self, positions,
                                            risk_factor_panel, expected):
        style_list = []
        for name, value in risk_factor_panel.iteritems():
            risk_factor_panel[name].columns = \
                risk_factor_panel[name].columns.astype(int)
            style_list.append(
                compute_style_factor_exposures(positions,
                                               risk_factor_panel[name])
                )
        expected.columns = expected.columns.astype(int)
        assert_frame_equal(pd.concat(style_list, axis=1), expected)

    @parameterized.expand([
        (test_pos, test_sectors, expected_sectors_longed,
         expected_sectors_shorted, expected_sectors_grossed)
    ])
    def test_compute_sector_exposures(self, positions, sectors,
                                      expected_longed, expected_shorted,
                                      expected_grossed):
        sectors.columns = sectors.columns.astype(int)
        sector_exposures = compute_sector_exposures(positions, sectors)

        expected_longed.columns = expected_longed.columns.astype(int)
        expected_shorted.columns = expected_shorted.columns.astype(int)
        expected_grossed.columns = expected_grossed.columns.astype(int)

        assert_frame_equal(pd.concat(sector_exposures[0], axis=1),
                           expected_longed)
        assert_frame_equal(pd.concat(sector_exposures[1], axis=1),
                           expected_shorted)
        assert_frame_equal(pd.concat(sector_exposures[2], axis=1),
                           expected_grossed)

    @parameterized.expand([
        (test_pos, test_caps, expected_caps_longed, expected_caps_shorted,
         expected_caps_grossed, expected_caps_netted)
    ])
    def test_compute_cap_exposures(self, positions, caps,
                                   expected_longed, expected_shorted,
                                   expected_grossed, expected_netted):
        caps.columns = caps.columns.astype(int)
        cap_exposures = compute_cap_exposures(positions, caps)

        expected_longed.columns = expected_longed.columns.astype(int)
        expected_shorted.columns = expected_shorted.columns.astype(int)
        expected_grossed.columns = expected_grossed.columns.astype(int)
        expected_netted.columns = expected_netted.columns.astype(int)

        assert_frame_equal(pd.concat(cap_exposures[0], axis=1),
                           expected_longed)
        assert_frame_equal(pd.concat(cap_exposures[1], axis=1),
                           expected_shorted)
        assert_frame_equal(pd.concat(cap_exposures[2], axis=1),
                           expected_grossed)
        assert_frame_equal(pd.concat(cap_exposures[3], axis=1),
                           expected_netted)

    @parameterized.expand([
        (test_shares_held, test_volumes, 0.1, expected_volumes)
    ])
    def test_compute_volume_exposures(self, shares_held, volumes,
                                      percentile, expected):
        l_thresh, s_thresh, g_thresh = compute_volume_exposures(shares_held,
                                                                volumes,
                                                                percentile)

        assert_series_equal(l_thresh, expected['0'], check_names=False)
        assert_series_equal(s_thresh, expected['1'], check_names=False)
        assert_series_equal(g_thresh, expected['2'], check_names=False)
