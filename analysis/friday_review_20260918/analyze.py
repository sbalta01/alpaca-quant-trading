"""Read-only analysis of saved deployment reports and downloaded evidence."""
import datetime as dt
import json
import pathlib
import re

OUT = pathlib.Path(__file__).resolve().parent
ROOT = OUT.parent.parent

def load(name):
    return json.loads((OUT / 'evidence' / (name + '.json')).read_text(encoding='utf-8-sig'))

log = (ROOT / 'live_weekly_momentum.md').read_text()
blocks = re.split(r'(?=^2026-\d\d-\d\d .*weekly momentum rebalance)', log, flags=re.M)
records = []
for block in blocks:
    if '(EXECUTED)' not in block:
        continue
    date = block[:10]
    equity = float(re.search(r'equity \$([\d,.]+)', block, re.I)[1].replace(',', ''))
    exposure = float(re.search(r'(?:gross|combined) exposure ([\d.]+)', block)[1])
    records.append(dict(date=date, equity=equity, target_exposure=exposure,
                        submitted=block.count('[submitted]'), two_sleeve='two-sleeve' in block))

bars = load('benchmark_bars')
assert bars.get('next_page_token') is None, 'Incomplete benchmark pagination'
prices = {symbol: {bar['t'][:10]: bar['c'] for bar in series}
          for symbol, series in bars['bars'].items()}
last = records[-1]
peak = records[0]['equity']
max_dd = 0
rows = []
for i, r in enumerate(records):
    peak = max(peak, r['equity'])
    max_dd = min(max_dd, r['equity'] / peak - 1)
    change = r['equity'] / records[max(i-1, 0)]['equity'] - 1
    rows.append(f"| {r['date']} | ${r['equity']:,.2f} | {change:+.2%} | {r['target_exposure']:.0%} | {r['submitted']} |")

comparisons = []
for start in ['2026-07-20', '2026-07-24', '2026-08-21']:
    baseline = next(r['equity'] for r in records if r['date'] == start)
    row = {'start': start, 'end': last['date'], 'strategy': last['equity'] / baseline - 1}
    row.update({s: p[last['date']] / p[start] - 1 for s, p in prices.items()})
    comparisons.append(row)
compare_rows = '\n'.join(f"| {r['start']} to {r['end']} | {r['strategy']:+.2%} | {r['SPY']:+.2%} | {r['QQQ']:+.2%} | {r['BIL']:+.2%} |" for r in comparisons)
runs = [r for r in load('workflow_runs')['workflow_runs'] if r['event'] == 'schedule']
summary = dict(records=records, comparisons=comparisons, snapshot_max_drawdown=max_dd,
               scheduled_runs=len(runs), scheduled_reports=len(records)-1,
               submitted_orders=sum(r['submitted'] for r in records),
               account_match=False, valuation_through=last['date'])
(OUT / 'results.json').write_text(json.dumps(summary, indent=2))

report = f"""# Friday paper deployment: efficacy review

Prepared September 18, 2026. Strategy valuations are available through **September 11**, not September 18. This review uses the committed deployment ledger, GitHub Actions run and job records, deployment source, and Alpaca SIP daily benchmark bars. All broker requests were read-only; no strategy settings, orders or schedules were changed.

**Assessment: a modest positive start, but an unproven strategy and an operational validation process that needs improvement.** The logged account grew from $100,000 to $101,843.98 (+1.84%). It lagged fully invested stock benchmarks over the observed overall period, while the short period after adding diversification was encouraging. Its substantial cash balance makes raw stock-index comparisons informative but insufficient to judge skill. The available evidence does not support a reliable Sharpe ratio, true daily maximum drawdown, or a claim of durable alpha.

## Duration and strategy versions

- July 17: dry run against a different-looking, older portfolio. Excluded from returns.
- July 20: first report marked EXECUTED, with a $100,000 baseline. Submission is recorded; fills are not verified.
- July 24: first scheduled Friday execution report, also showing exactly $100,000.
- July 20–August 21: original S&P 500 momentum strategy, top 10, volatility target 20%, inverse-volatility weights, 200-day market gate and rank buffering.
- August 21 onward: 60% of capital allocated to momentum and 40% to an ETF diversifier. This is a changed strategy, not a continuous trial of one fixed configuration. The allocations are sleeve budgets, not actual invested weights.
- September 11: latest saved account valuation. That is **53 days / 7 weeks 4 days** after the first execution report. The first scheduled Friday to the latest Friday spans **7 weeks**.
- As of September 18, **60 days / 8 weeks 4 days** have elapsed since July 20, but the last week is unmeasured. The two-sleeve configuration has been enabled for four calendar weeks; its saved valuations cover only **three weeks**, beginning before the first diversification orders could fill.

## Recorded account results

| Report date | Account equity | Change since previous report | Target invested exposure | Orders marked submitted |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The September 4 interval covers **two weeks** because August 28 produced no new report. Equity is captured before that run's orders; target exposure is intended exposure, not a verified post-fill holding weight.

- Net gain: **$1,843.98 / +1.84%**, assuming no deposits, withdrawals or account resets between the logged observations. The actual deployment account's cash flows could not be checked.
- Highest reported balance: **$102,619.29 on August 14**, +2.62% from the baseline.
- Largest observed decline between reported peaks and later snapshots: **{max_dd:.2%}**, August 14–21. The true daily/intraday maximum drawdown could be materially worse; weekly snapshots cannot establish it.
- Best observed one-week interval: **+2.35%**, August 7–14. Worst: **−1.80%**, August 14–21.
- Before the two-sleeve switch, cumulative return was **+0.77%**. From the switch's pre-order August 21 valuation to September 11, return was **+1.06%**. These compound to the overall return; they are not per-sleeve attribution.

## Market comparison

| Valuation window | Logged strategy | SPY | QQQ | BIL cash-proxy ETF |
|---|---:|---:|---:|---:|
{compare_rows}

Benchmark method: Alpaca historical daily SIP bars requested with `adjustment=all`; ratios of returned closing prices. BIL is a cash-like Treasury-bill ETF comparator, not interest actually received by the strategy. No benchmark trading costs are charged. The July 20 strategy baseline was observed intraday, so that row is an approximate date-matched comparison. July 24–September 11 gives the cleaner Friday-to-Friday comparison, although account snapshots taken after market close may still reflect extended-hours marks. These are adjusted-price comparisons, not a reconstructed executable benchmark or a separate dividend-ledger total-return calculation.

Over the cleaner seven-week window, the strategy trailed SPY by **1.59 percentage points** and QQQ by **2.64 points**, while beating BIL by **1.36 points**. This does not alone show poor risk-adjusted performance: the strategy targeted only **21–25% invested** initially and **44–49% invested** after diversification, leaving roughly **51–79% in cash** at the recorded targets. It deliberately takes less exposure than a fully invested index. An exposure-matched benchmark and daily realized volatility are needed to judge that tradeoff fairly.

The diversification window is encouraging: +1.06% versus SPY −0.19% and QQQ +0.20%. However, there are only three saved two-sleeve snapshots and one missed weekly rebalance in that window. The account contains both sleeves, and the first snapshot precedes ETF fills, so this is **not evidence that the ETF sleeve itself generated +1.06%**, nor proof it improved Sharpe or drawdown. The initial momentum holdings are concentrated in related hardware/semiconductor stocks; UUP, DBC and SLV were the diversifier purchases. The diversifier's selected assets still carry market risk.

## Operational findings

**1. A scheduled rebalance was silently skipped.** All eight scheduled workflow runs July 24–September 11 show success, but only seven created execution reports: **87.5% report-producing coverage**, not an 87.5% fill-success rate. The August 28 event started August 29 at 03:15 UTC (August 28, 11:15 p.m. New York), 5 hours 45 minutes after its intended start. Source at that run's commit tests the UTC weekday and exits successfully on Saturday. Its rebalance step lasted one second and no new report appeared. These records and the deterministic guard strongly identify the skip; raw job console logs were not obtained. See [the affected run](https://github.com/sbalta01/alpaca-quant-trading/actions/runs/33230922926). GitHub explicitly documents that [scheduled jobs can be delayed](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows).

**2. The job uses Thursday data despite running after Friday's close.** Every saved Friday report lists Thursday as its signal date. `fetch_close_matrix` passes the current calendar date as `end`; [yfinance documents end as exclusive](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html). The strategy therefore excludes Friday's completed bar. This is a confirmed mismatch between the intended Friday-close strategy and deployed signal timing. It can change rankings, exposure and trades; its financial impact has not been quantified here.

**3. Successful submission is not confirmed execution.** Across eight EXECUTED reports, **49 order lines** say submitted and none log an immediate ERROR. The report does not store broker order identifiers, eventual fills, cancellations, rejected orders, execution prices, or realized slippage. Errors during submission are caught and written to the report without necessarily failing the job. The unchanged $100,000 balance and fresh buys on both July 20 and July 24 need reconciliation: the records do not establish whether the initial orders filled, were canceled, or the paper account was reset.

**4. A skipped or failed run can resend an old report.** The workflow extracts the last report block and emails it even when no new block was produced. The August 28 job shows the extraction and email steps succeeded after the skip; the workflow therefore selected the prior August 21 summary. Email delivery/content was not independently inspected. A green workflow and a familiar email are insufficient evidence of a completed new rebalance.

**5. The accessible local broker account does not reconcile to this deployment.** Its September 18 holdings were AXON, HONA and TTWO, with no orders returned since July 17, whereas the Friday log shows a momentum/ETF portfolio. Its July 24 daily equity was $105,225.01, compared with the deployed report's $100,000. The history response also ended August 17 despite current account marks being available. I excluded that account's current balance and history from strategy returns. It appears to be a different paper account, but the GitHub secret/account mapping remains unconfirmed. Public benchmark bars from its market-data credentials are independent of its holdings and remain usable.

## What can and cannot be concluded

The strategy has made a small recorded profit and appears to have participated in gains with limited intended market exposure. The short diversification window looks useful. Those are reasons to continue a controlled paper evaluation, not reasons to call the edge proven. Overall index outperformance has not been demonstrated. Risk-adjusted superiority cannot be measured honestly from seven uneven return intervals spanning a strategy change.

The repository quotes attractive historical backtests (including two-sleeve Sharpe 1.22 and maximum drawdown −15.9%). Those are documented research results, **not results achieved during this paper trial**, and were not reproduced in this review. The documentation acknowledges using today's stock-index membership in historical tests, which can inflate results. The different live signal/fill timing further limits direct backtest comparisons.

[Alpaca's paper-trading documentation](https://docs.alpaca.markets/us/docs/paper-trading) describes simulation limitations, including no simulated dividends and omitted real-market effects. Even a fully reconciled paper record would still need to be interpreted with those limits.

## Recommended next steps

1. Match the GitHub paper account to read-only broker access and reconcile all 49 submissions, especially July 20/24; obtain daily equity and deposits/withdrawals. Keep credentials out of reports and chat.
2. Correct the market-session/date handling, include the latest completed Friday bar, and explicitly report skipped, failed and completed runs. Add duplicate-run protection before permitting delayed retries.
3. Record actual fills, post-fill holdings, cash, costs and daily benchmark comparisons. Attach every report to its current run and stop reusing old summaries as fresh results.
4. Keep the strategy configuration fixed for a further observation period. The repository's own minimum is 8–12 paper weeks; the current two-sleeve version has only three measured weeks. Even 8–12 weeks would establish basic operational behavior more than statistical evidence of alpha. Evaluate varied market conditions and a meaningful drawdown before making a stronger claim.

**My judgment:** cautiously encouraging investment results, insufficient evidence of a durable advantage, and clear reliability gaps. I would continue paper evaluation and fix measurement/execution issues before using these results to justify real-money deployment.

## Reproducibility and evidence

- [Original deployment ledger](../../live_weekly_momentum.md)
- [Deployment guide and historical backtest claims](../../DEPLOYMENT.md)
- [Workflow source](../../.github/workflows/deploying-weekly-momentum.yml)
- [Two-sleeve deployer](../../main/deploy_sleeves.py)
- [Price-fetch helper](../../main/backtest_weekly_momentum.py)
- [Calculated results](results.json)
- [Saved public workflow records](evidence/workflow_runs.json)
- [Saved affected-job metadata](evidence/aug28_jobs.json)
- [Benchmark API response](evidence/benchmark_bars.json)

The `evidence` directory also contains private local-account snapshots used only to detect the mismatch. Do not publish those files. No credentials were written to the output artifacts.
"""
(OUT / 'friday_paper_review.md').write_text(report, encoding='utf-8')
print(json.dumps(summary, indent=2))
print('Report:', OUT / 'friday_paper_review.md')
