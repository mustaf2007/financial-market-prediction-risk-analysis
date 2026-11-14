# Financial AI System
## V2.2.7 — Final Acceptance & Research Report

**SPY | Intraday research configuration | 12 November 2026**

### 1. Executive summary

V2.2.7 completed the submitted synthetic-data and real-SPY runs and generated the expected output artifacts. The real-SPY holdout run selected XGBoost but did not identify a trading threshold that satisfied the configured economic and minimum-activity rules. The portfolio therefore remained in cash. The controlled active-trade acceptance test subsequently passed all six deterministic checks.

The software execution and tested return arithmetic are accepted for the tested paths. The research results do not establish predictive advantage or investment readiness. The system should be treated as a research and educational tool, not a profitable trading strategy.

### 2. System scope

- Asset: SPY; strategy mode: intraday.
- Signal is generated after a session close; execution return is measured on the following session from open to close.
- Historical run period: 2018-01-01 to 2026-11-01 (end date exclusive).
- Real-data holdout signal dates: 2025-03-24 to 2026-10-29.
- Real-data execution dates: 2025-03-25 to 2026-10-30.
- Models evaluated: Logistic Regression, Random Forest, HistGradientBoosting and XGBoost.
- Transaction-cost assumption: 5 basis points per side, 10 basis points round trip.

### 3. Real SPY holdout results

| Metric | Result | Interpretation |
|---|---:|---|
| Selected model | XGBoost | Selected by chronological validation |
| Mean fold ROC-AUC | 0.52875 | Walk-forward average; not final holdout |
| Selection-period ROC-AUC | 0.51398 | Earlier 70% OOF selection period |
| Holdout accuracy | 50.7426% | Below majority baseline 51.2376% |
| Holdout ROC-AUC | 0.5054 | Close to random discrimination |
| Brier score | 0.25084 | Slightly worse than base-rate reference 0.249847 |
| Predicted UP rate | 97.0297% | Strongly one-sided predictions |
| Portfolio decision | CASH_NO_TRADE | No threshold met configured rules |
| Strategy net return | 0.0000% | No positions were opened |
| SPY same-window return | +13.9519% | Buy-and-hold comparison |

### 4. Model and threshold selection

The earlier 70% of finite chronological out-of-fold predictions was used for model selection. The later 30% was reserved for economic-threshold selection. The final holdout was not used for these selection steps. The design reduces leakage but is not a fully nested outer walk-forward estimate; threshold selection can still be optimistic.

The configured threshold rules required exposure of at least 10%, at least 20 active observations, positive net return and positive Sharpe. No threshold passed all conditions. The cash/no-trade decision is therefore consistent with the system's stated policy.

### 5. Feature and regime experiments

On the real SPY run, the feature-group mean validation AUCs were: volatility 0.5308, volume 0.5246, price action 0.5085, trend 0.4866 and momentum 0.4859. These are exploratory validation comparisons, not proof that a feature family has a durable predictive edge.

Regime-specific validation reported bullish AUC 0.5098 and high-volatility AUC 0.5335. Bearish and sideways regimes had insufficient validation folds. Holdout regime sample sizes were small for bearish (40 observations) and sideways (18 observations); related statistics are unstable and should not be used as evidence of a reliable regime strategy.

### 6. Acceptance test results

The deterministic controlled active-trade test used four observations, a 0.60 probability threshold, 100% position sizing, and 5 bps cost per side. It verified the active mask, position size, round-trip costs, gross-to-net arithmetic, zero costs on inactive days and two-sided costs on active days.

| Check | Expected result | Observed |
|---|---|---|
| Active trade mask | [1, 1, 0, 1] | PASS |
| Position sizes | [1, 1, 0, 1] | PASS |
| Round-trip costs | [0.001, 0.001, 0, 0.001] | PASS |
| Net returns | [0.009, -0.011, 0, 0.004] | PASS |
| Inactive-day cost | 0 | PASS |
| Two-sided cost on active days | 0.001 | PASS |

**Overall controlled active-trade test result: PASS (6 of 6 checks).**

### 7. Acceptance decision

- Software execution: accepted for the submitted runs.
- Output consistency: independently checked for key holdout metrics and return arithmetic.
- Controlled active-trade calculations: passed the deterministic test.
- Predictive research: no useful predictive advantage demonstrated on the final holdout.
- Investment readiness: not established.

### 8. Limitations and risk disclosures

- The model's holdout ROC-AUC is 0.5054, close to 0.50.
- Accuracy is below the majority-class baseline, and Brier score is slightly worse than the base-rate reference.
- The classifier predicts UP for approximately 97% of holdout observations.
- The real SPY strategy remained in cash; the historical run does not demonstrate active strategy profitability.
- The active-trade test is a deterministic unit test, not a broker execution or market-impact simulation.
- The validation design is not fully nested; threshold selection may be optimistic.
- Feature-group and regime-specific results are exploratory and subject to sample-size and selection limitations.
- Historical market data and adjusted prices may be revised by data providers.

### 9. Release freeze and change control

V2.2.7 is the final planned development release. The project should proceed to documentation, archival and closure rather than another feature-development cycle. A further code change is justified only by a confirmed material defect. Any such repair must identify the defect, document the change and include a regression test.

### 10. Package contents

The original publication package includes the main Python source, dependency list, controlled acceptance test, README, final report and the real-SPY output artifacts.

**Research and educational use only.** This project does not provide investment advice, a recommendation to buy or sell securities, or a guarantee of returns. It has not demonstrated investment-ready performance.
