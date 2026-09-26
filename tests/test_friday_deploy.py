"""Small offline regressions for the Friday date and report bugs."""
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from main import deploy_sleeves as deploy


class FridayDeployTests(unittest.TestCase):
    def setUp(self):
        self.signal_date = date(2026, 9, 25)
        self.start = date(2025, 2, 3)
        self.fresh = pd.DataFrame({'SPY': [767.18, 771.35], 'AAPL': [335.92, 341.07]},
                                  index=pd.to_datetime(['2026-09-24', '2026-09-25']))

    def test_missing_friday_bar_recovers_before_using_panel(self):
        stale = self.fresh.iloc[:1]
        with patch.object(deploy, 'fetch_close_matrix', side_effect=[stale, self.fresh]) as fetch, \
                patch.object(deploy.time, 'sleep') as sleep:
            actual = deploy.fetch_completed_prices(['SPY', 'AAPL'], self.start, self.signal_date)
        pd.testing.assert_frame_equal(actual, self.fresh)
        self.assertEqual(fetch.call_count, 2)
        fetch.assert_called_with(['SPY', 'AAPL'], self.start, date(2026, 9, 26))
        sleep.assert_called_once_with(30)

    def test_missing_spy_or_download_error_can_recover(self):
        for first in [self.fresh.drop(columns='SPY'), pd.DataFrame(), TimeoutError('timeout')]:
            with self.subTest(first=type(first).__name__), \
                    patch.object(deploy, 'fetch_close_matrix', side_effect=[first, self.fresh]), \
                    patch.object(deploy.time, 'sleep'):
                actual = deploy.fetch_completed_prices(['SPY', 'AAPL'], self.start, self.signal_date)
                pd.testing.assert_frame_equal(actual, self.fresh)

    def test_forward_fill_cannot_fabricate_friday_spy_bar(self):
        missing = self.fresh.copy()
        missing.loc['2026-09-25', 'SPY'] = float('nan')
        with patch.object(deploy, 'fetch_close_matrix', return_value=missing) as fetch, \
                patch.object(deploy.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'latest received: 2026-09-24.*no orders submitted'):
                deploy.fetch_completed_prices(['SPY', 'AAPL'], self.start, self.signal_date)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 60])

    def test_future_rows_are_excluded_without_discarding_friday(self):
        future = pd.DataFrame({'SPY': [999.], 'AAPL': [999.]},
                              index=pd.to_datetime(['2026-09-28']))
        with patch.object(deploy, 'fetch_close_matrix', return_value=pd.concat([self.fresh, future])), \
                patch.object(deploy.time, 'sleep') as sleep:
            actual = deploy.fetch_completed_prices(['SPY', 'AAPL'], self.start, self.signal_date)
        pd.testing.assert_frame_equal(actual, self.fresh)
        sleep.assert_not_called()

    def test_persistent_empty_or_failed_download_is_bounded(self):
        for panel in [pd.DataFrame(), self.fresh.drop(columns='SPY'), TimeoutError('timeout')]:
            with self.subTest(panel=type(panel).__name__), \
                    patch.object(deploy, 'fetch_close_matrix', side_effect=[panel] * 3) as fetch, \
                    patch.object(deploy.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'exhausted 3 attempts'):
                    deploy.fetch_completed_prices(['SPY', 'AAPL'], self.start, self.signal_date)
                self.assertEqual(fetch.call_count, 3)

    def test_persistent_missing_data_reports_failure_before_broker_access(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(deploy, 'LATEST_REPORT_PATH', Path(folder) / 'latest.md'), \
                patch.object(deploy, 'REPORT_PATH', Path(folder) / 'history.md'), \
                patch.object(deploy.sys, 'argv', ['deploy_sleeves.py', '--execute']), \
                patch.object(deploy, 'datetime') as clock, \
                patch('src.data.universe.fetch_sp500_symbols', return_value=['AAPL']), \
                patch.object(deploy, 'fetch_close_matrix', return_value=self.fresh.iloc[:1]), \
                patch.object(deploy.time, 'sleep'), \
                patch('alpaca.trading.client.TradingClient') as broker:
            clock.now.return_value = datetime.fromisoformat('2026-09-26T00:01:10+00:00')
            with self.assertRaisesRegex(RuntimeError, 'exhausted 3 attempts'):
                deploy.main()
            broker.assert_not_called()
            report = Path(deploy.LATEST_REPORT_PATH).read_text()
            self.assertIn('2026-09-25', report)
            self.assertIn('latest received: 2026-09-24', report)
            self.assertIn('no orders submitted', report)
            self.assertEqual(report, Path(deploy.REPORT_PATH).read_text())

    def test_delayed_utc_saturday_is_still_friday_in_new_york(self):
        now = datetime.fromisoformat('2026-08-29T03:15:00+00:00')
        self.assertEqual(now.astimezone(deploy.MARKET_TZ).date(), date(2026, 8, 28))
        self.assertEqual(deploy.latest_completed_session(now), date(2026, 8, 28))

    def test_includes_friday_only_after_close(self):
        for stamp, expected in [('2026-09-18T21:30:00+00:00', date(2026, 9, 18)),
                                ('2026-09-18T19:30:00+00:00', date(2026, 9, 17)),
                                ('2026-01-09T20:30:00+00:00', date(2026, 1, 8))]:
            self.assertEqual(deploy.latest_completed_session(datetime.fromisoformat(stamp)), expected)

    def test_current_report_never_contains_the_previous_run(self):
        with tempfile.TemporaryDirectory() as folder:
            latest, history = Path(folder) / 'latest.md', Path(folder) / 'history.md'
            with patch.object(deploy, 'LATEST_REPORT_PATH', latest), patch.object(deploy, 'REPORT_PATH', history):
                deploy.write_report(['old run'])
                deploy.write_report(['new run skipped'])
            self.assertEqual(latest.read_text().strip(), 'new run skipped')
            self.assertIn('old run', history.read_text())


if __name__ == '__main__':
    unittest.main()
