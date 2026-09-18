"""Small offline regressions for the Friday date and report bugs."""
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

from main import deploy_sleeves as deploy


class FridayDeployTests(unittest.TestCase):
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
