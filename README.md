# AI-Powered Financial Market Prediction and Risk Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Release](https://img.shields.io/badge/release-V2.2.7-informational)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research Status](https://img.shields.io/badge/research_edge-not_established-orange)]()

A Python-based financial machine-learning research system for directional market classification, chronological model evaluation, backtesting, transaction-cost assessment, and portfolio risk analysis.

**Final planned development release: V2.2.7**

> **Research disclaimer:** This repository is an educational and quantitative research project. The tested SPY model did not demonstrate a reliable predictive advantage or investment-ready profitability. It is not financial advice and is not a live-trading system.

## Overview

The system evaluates market-direction classifiers and applies a rule-based portfolio layer with exposure and transaction-cost controls. It includes:

- Historical SPY data retrieval and synthetic-data execution.
- Technical and price/volume-derived feature engineering.
- Chronological walk-forward model evaluation.
- Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost comparisons.
- Probability calibration and model/threshold selection.
- Intraday signal-to-next-session execution alignment.
- Position sizing, transaction costs, and a cash/no-trade decision.
- Classification, benchmark, drawdown, volatility, and risk reporting.
- Feature-group and market-regime exploratory experiments.

## Final release findings

The saved real-SPY run used an intraday strategy configuration and a holdout execution window from **2025-03-25 through 2026-10-30**.

| Measure | Result |
|---|---:|
| Selected model | XGBoost |
| Mean walk-forward fold ROC-AUC | 0.5287 |
| Earlier OOF model-selection ROC-AUC | 0.5140 |
| Holdout accuracy | 50.7426% |
| Majority-class baseline accuracy | 51.2376% |
| Holdout ROC-AUC | 0.5054 |
| Brier score | 0.25084 |
| Predicted UP rate | 97.0297% |
| Portfolio decision | `CASH_NO_TRADE` |
| Strategy net return | 0.0000% |
| Market return over same execution window | +13.9519% |

No candidate trading threshold satisfied all configured economic and minimum-activity conditions. The system therefore remained in cash. This is evidence that the configured no-trade rule operated; it is **not** evidence of a profitable trading strategy.

The model's holdout discrimination was close to random, its accuracy was below the majority-class baseline, and its Brier score was slightly worse than the base-rate reference. No reliable predictive edge was established.

## Repository contents

| File | Purpose |
|---|---|
| `financial_ai_system_v227.py` | Main research application |
| `requirements_v227.txt` | Python dependencies |
| `acceptance_test_v227.py` | Deterministic active-trade calculation test |
| `V227_Final_Acceptance_and_Research_Report.docx` | Final acceptance and research report |
| `spy_v227_backtest.csv` | Real-SPY holdout backtest |
| `spy_v227_metrics.json` | Machine-readable metrics and configuration |
| `spy_v227_benchmarks.csv` | Model/benchmark comparison |
| `spy_v227_threshold_analysis.csv` | Threshold economics |
| `spy_v227_feature_importance.csv` | Feature importance results |
| `spy_v227_feature_group_experiments.csv` | Exploratory feature-group results |
| `spy_v227_regime_diagnostics.csv` | Market-regime diagnostics |
| `spy_v227_regime_model_comparison.csv` | Exploratory regime-specific model results |
| `LICENSE` | MIT License |
| `CITATION.cff` | Citation metadata |
| `SECURITY.md` | Security reporting guidance |

## Requirements

- Python 3.10 or later recommended.
- Internet access is required for the default Yahoo Finance data download.
- Dependencies are listed in `requirements_v227.txt`.

## Installation

### Windows Command Prompt

```cmd
git clone https://github.com/mustaf2007/financial-market-prediction-risk-analysis.git
cd financial-market-prediction-risk-analysis

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements_v227.txt
```

## Run the system

### Real SPY data

```cmd
python financial_ai_system_v227.py
```

### Synthetic data

```cmd
python financial_ai_system_v227.py --synthetic
```

### Controlled active-trade acceptance test

```cmd
python acceptance_test_v227.py
```

The controlled acceptance test checks active-trade detection, position sizing, two-sided transaction costs, gross-to-net arithmetic, inactive-day cost handling, and round-trip cost application. The recorded run passed all six checks.

## Methodology and limitations

The final release uses chronological out-of-fold predictions. The earlier 70% of finite OOF predictions is used for model selection and the later 30% for economic-threshold selection. The final holdout is not used for those selections. This is not a fully nested outer walk-forward estimate, and threshold selection can still be optimistic.

Feature-group and regime-specific experiments are exploratory diagnostics. Small regime samples can produce unstable statistics and should not be treated as evidence of robust regime-specific returns.

The real-SPY holdout selected no trades. Consequently, the historical run validates the cash/no-trade path but does not itself validate active market execution or real-world slippage. The separate deterministic test verifies the mathematical transaction-cost path, not broker execution, market impact, liquidity, or live fills.

Historical data providers may revise adjusted prices or historical records. The saved CSV and JSON outputs are the reference artifacts for the reported run; a future data download may not reproduce identical results.

## Release policy

V2.2.7 is the final planned development release. The source is frozen. Do not create another version merely to add indicators, algorithms, or features. Reopen development only to address a confirmed material defect, and document the defect, repair, and regression test.

## Risk notice

This project is for research and educational purposes only. It does not provide investment advice, a recommendation to buy or sell securities, or a guarantee of returns. It has not demonstrated investment-ready performance. Do not deploy capital based solely on this system or its historical outputs.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).