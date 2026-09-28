"""Offline checks for report accuracy and separation from broker audit details."""
import dataclasses
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from main import deploy_sleeves as deploy


class WeeklyReportTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.fromisoformat('2026-10-03T14:30:00+00:00')
        self.dates = pd.bdate_range(end='2026-10-02', periods=330)
        rng = np.random.default_rng(7)
        self.panel = pd.DataFrame(100 * np.exp(np.cumsum(
            rng.normal(.001, .02, (330, 3)), axis=0)), index=self.dates,
            columns=['AAA', 'BBB', 'CCC'])
        self.spy = pd.Series(np.linspace(100, 130, 330), index=self.dates)
        self.sleeve = dataclasses.replace(deploy.MOMENTUM, top_k=2, weight_cap=.6)

    def test_diagnostics_preserve_strategy_weights(self):
        universe = list(self.panel.columns)
        expected = deploy.compute_target_weights(
            self.panel, self.spy, top_k=2, weight_cap=.6,
            low_exposure=self.sleeve.low_exposure,
            selector=deploy.BufferedSelector(self.sleeve.buffer_mult),
            exposure_fn=deploy.make_vol_target_exposure(
                target_vol=self.sleeve.target_vol, with_regime_gate=True,
                low_exposure=self.sleeve.low_exposure))
        expected = deploy.apply_no_trade_band(expected, pd.Series(dtype=float),
                                             self.sleeve.min_trade_fraction)
        result = deploy.compute_sleeve(self.sleeve, self.panel, self.spy, {},
                                      (self.sleeve,), universe, 100000)
        self.assertEqual(result.status, 'ok')
        pd.testing.assert_series_equal(result.weights, expected)
        self.assertAlmostEqual(result.risk['vol21'],
                               deploy.realized_portfolio_vol(self.panel, expected, 21))
        self.assertEqual(result.risk['gate'], 1.0)

    def report(self, spy=None, risk=None):
        result = deploy.SleeveResult(self.sleeve, 'ok', pd.Series({'AAA': .5}),
                                    {'AAA'}, signal_date=self.dates[-1].date(),
                                    risk=risk or dict(vol21=.5, vol63=.3, gate=.4,
                                                      gate_enabled=True, model_exposure=.16))
        return '\n'.join(deploy.portfolio_report(
            self.now, self.dates[-1].date(), 100000, {'AAA': 20000, 'TLT': 10000},
            pd.Series({'AAA': .3}), [result], self.spy if spy is None else spy, True))

    def test_exposure_includes_untouched_holdings_and_volatility_driver(self):
        report = self.report()
        self.assertIn('30.0% -> 40.0% target', report)
        self.assertIn('Implied cash at target: 60.0%', report)
        self.assertIn('Includes 10.0%', report)
        self.assertIn('momentum: 20.0% -> 30.0% target', report)
        self.assertIn('21d 50.0%, 63d 30.0%', report)
        self.assertIn('using 21-day estimate, above target', report)
        self.assertIn('trend gate reducing exposure (x0.40)', report)
        self.assertNotIn('WARNINGS: none', report)

    def test_slow_volatility_and_disabled_gate(self):
        report = self.report(risk=dict(vol21=.1, vol63=.15, gate=1.,
                                      gate_enabled=False, model_exposure=1.))
        self.assertIn('using 63-day estimate, within target', report)
        self.assertIn('trend gate not applied', report)

    def test_market_trend_and_missing_estimate_are_not_invented(self):
        falling = pd.Series(np.linspace(130, 100, 330), index=self.dates)
        self.assertIn('below trend threshold', self.report(spy=falling))
        report = self.report(spy=self.spy.iloc[:10], risk=dict(
            vol21=float('nan'), vol63=float('nan'), gate=1.,
            gate_enabled=True, model_exposure=1.))
        self.assertIn('insufficient SPY history', report)
        self.assertIn('21d unavailable, 63d unavailable', report)

    def test_email_omits_routine_status_but_keeps_failures_and_audit(self):
        result = deploy.SleeveResult(self.sleeve, 'ok', pd.Series({'AAA': .5}),
                                    {'AAA'}, signal_date=self.dates[-1].date())
        for outcome in ['accepted', 'rejected', RuntimeError('broker unavailable')]:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as folder, \
                    patch.object(deploy, 'LATEST_REPORT_PATH', Path(folder) / 'latest.md'), \
                    patch.object(deploy, 'REPORT_PATH', Path(folder) / 'history.md'), \
                    patch.object(deploy.sys, 'argv', ['deploy_sleeves.py', '--execute']), \
                    patch.object(deploy, 'datetime') as clock, \
                    patch('src.data.universe.fetch_sp500_symbols', return_value=['AAA']), \
                    patch.object(deploy, 'fetch_completed_prices', return_value=self.panel.assign(SPY=self.spy)), \
                    patch.object(deploy, 'resolve_sleeves', return_value=((self.sleeve,), (self.sleeve,))), \
                    patch.object(deploy, 'compute_sleeve', return_value=result), \
                    patch.object(deploy, 'build_orders', return_value=[('AAA', 'buy', 1000, False),
                                                                    ('BBB', 'buy', 500, False)]), \
                    patch('alpaca.trading.client.TradingClient') as broker:
                clock.now.return_value = self.now
                client = broker.return_value
                client.get_account.return_value = SimpleNamespace(equity=100000)
                client.get_all_positions.return_value = []
                if isinstance(outcome, Exception):
                    client.submit_order.side_effect = outcome
                else:
                    client.submit_order.return_value = SimpleNamespace(
                        id='audit-order-id', status=SimpleNamespace(value=outcome),
                        filled_qty=0, filled_avg_price=None)
                if outcome == 'accepted':
                    deploy.main()
                    self.assertEqual(client.submit_order.call_count, 2)
                else:
                    with self.assertRaises(SystemExit):
                        deploy.main()
                    self.assertEqual(client.submit_order.call_count, 1)
                email = Path(deploy.LATEST_REPORT_PATH).read_text()
                history = Path(deploy.REPORT_PATH).read_text()
                self.assertNotIn('status=', email)
                self.assertNotIn('audit-order-id', email)
                self.assertNotIn('filled_qty', email)
                self.assertIn('BUY  $1000 AAA', email)
                if outcome == 'accepted':
                    self.assertIn('status=accepted', history)
                    self.assertIn('fills not yet confirmed', email)
                else:
                    self.assertIn('FAILED: remaining orders stopped', email)
                    self.assertNotIn('BUY  $500 BBB', email)
                    self.assertIn('rejected' if outcome == 'rejected' else 'broker unavailable', email)


if __name__ == '__main__':
    unittest.main()
