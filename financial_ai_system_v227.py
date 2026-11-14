"""
Financial AI System V2.2.7 — Final planned development release
Chronological calibration, economic validation and risk controls.

Research prototype only; not investment advice.

Design:
- Signal is formed after session t close.
- "intraday": execute next session open-to-close.
- "close-close": enter at next session close and exit at the following session close.
- Model fitting and Platt calibration are chronological within each walk-forward fold.
- Model selection and threshold selection use separate chronological OOF segments.
- Threshold selection uses the same volatility sizing and cost assumptions as the backtest.
- If no threshold meets minimum exposure AND positive OOF Sharpe, the system stays in cash.
- Final holdout is not used for model or threshold selection.
- Purging is strategy-aware: one row for intraday and two for close-close.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


@dataclass
class Config:
    ticker: str = "SPY"
    start: str = "2018-01-01"
    end: str = "2026-11-01"
    test_size: float = 0.20
    cost_bps_per_side: float = 5.0
    annualization: int = 252
    random_state: int = 42
    output_dir: str = "results_v227"
    strategy: str = "intraday"
    min_train: int = 500
    n_splits: int = 5
    min_exposure: float = 0.10
    target_volatility: float = 0.15
    max_position: float = 1.0
    calibration_fraction: float = 0.20
    min_calibration_rows: int = 60
    min_active_trades: int = 20


FEATURES = [
    "Return_1D", "Return_5D", "Return_10D", "Return_21D",
    "Volatility_10D", "Volatility_21D", "SMA_Ratio_20",
    "SMA_Ratio_50", "SMA_Ratio_200", "RSI_14", "MACD",
    "MACD_Signal", "MACD_Hist", "BB_Position", "ATR_Pct",
    "Volume_Change", "Volume_Ratio_20", "High_Low_Range",
    "Regime_Trend", "Regime_HighVol"
]


def normalise_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        price_names = {"Open", "High", "Low", "Close", "Volume"}
        for level in range(data.columns.nlevels):
            values = set(map(str, data.columns.get_level_values(level)))
            if price_names.issubset(values):
                data.columns = data.columns.get_level_values(level)
                break
            if price_names.intersection(values):
                data.columns = data.columns.get_level_values(level)
                break
    data.columns = [str(c).title() for c in data.columns]
    return data


def download_data(cfg: Config) -> pd.DataFrame:
    print(f"[*] Downloading {cfg.ticker}: {cfg.start} to {cfg.end} (end exclusive)")
    data = yf.download(
        cfg.ticker, start=cfg.start, end=cfg.end,
        auto_adjust=True, progress=False, group_by="column"
    )
    if data.empty:
        raise RuntimeError("No market data downloaded. Check ticker, dates and internet.")
    data = normalise_yfinance_columns(data)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"Downloaded data missing columns: {missing}")
    data = data[required].sort_index().copy()
    print(f"[+] Downloaded {len(data):,} observations.")
    return data


def make_synthetic_data(n: int = 2600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    regime = np.zeros(n)
    for i in range(1, n):
        regime[i] = regime[i - 1] if rng.random() > 0.025 else rng.choice([-1, 0, 1])
    close_return = 0.00025 + 0.00015 * regime + rng.normal(0, 0.012, n)
    close = 100 * np.cumprod(1 + close_return)
    open_gap = rng.normal(0.0001, 0.006, n)
    open_price = close / (1 + close_return) * (1 + open_gap)
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.008, n))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.008, n))
    volume = rng.lognormal(15, 0.25, n)
    result = pd.DataFrame({
        "Open": open_price, "High": high, "Low": low,
        "Close": close, "Volume": volume
    }, index=dates)
    print(f"[+] Generated {len(result):,} synthetic observations.")
    return result


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def build_dataset(raw: pd.DataFrame, strategy: str) -> pd.DataFrame:
    d = raw.copy()
    close, high, low = d["Close"], d["High"], d["Low"]

    d["Return_1D"] = close.pct_change()
    for period in (5, 10, 21):
        d[f"Return_{period}D"] = close.pct_change(period)
    d["Volatility_10D"] = d["Return_1D"].rolling(10).std()
    d["Volatility_21D"] = d["Return_1D"].rolling(21).std()

    for period in (20, 50, 200):
        d[f"SMA_Ratio_{period}"] = close / close.rolling(period).mean() - 1

    d["RSI_14"] = calculate_rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    upper, lower = middle + 2 * deviation, middle - 2 * deviation
    d["BB_Position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)
    d["ATR_Pct"] = true_range.rolling(14).mean() / close
    d["Volume_Change"] = d["Volume"].pct_change()
    d["Volume_Ratio_20"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["High_Low_Range"] = (high - low) / close
    d["Regime_Trend"] = (d["SMA_Ratio_50"] > 0).astype(float).where(d["SMA_Ratio_50"].notna())
    vol_median = d["Volatility_21D"].rolling(60, min_periods=40).median()
    d["Regime_HighVol"] = (d["Volatility_21D"] > vol_median).astype(float).where(vol_median.notna())

    # Four-state descriptive regime label. This is used for diagnostics and
    # regime-specific experiments, not as a future-looking target.
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    trend_spread = sma50 / sma200 - 1.0
    # Trend regime is independent of the high-volatility flag.
    # This allows a bullish/high-volatility observation to retain both labels.
    d["Market_Regime"] = np.select(
        [(trend_spread >= 0.01), (trend_spread <= -0.01)],
        ["BULLISH", "BEARISH"],
        default="SIDEWAYS"
    )
    d.loc[trend_spread.isna(), "Market_Regime"] = np.nan

    d["Signal_Date"] = d.index

    if strategy == "intraday":
        # Signal after t close; enter next session open; exit that session close.
        d["Execution_Date"] = pd.Series(d.index, index=d.index).shift(-1)
        d["Execution_Return"] = d["Close"].shift(-1) / d["Open"].shift(-1) - 1
    else:
        # Signal after t close; enter next session close; exit following session close.
        d["Execution_Date"] = pd.Series(d.index, index=d.index).shift(-2)
        d["Execution_Return"] = d["Close"].shift(-2) / d["Close"].shift(-1) - 1

    d["Target"] = (d["Execution_Return"] > 0).astype(int)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["Execution_Date", "Execution_Return"])
    return d


def make_model_candidates(seed: int):
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000,
                random_state=seed
            ))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=350, max_depth=5, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=seed,
                n_jobs=-1
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.04, max_leaf_nodes=7,
                l2_regularization=2.0, random_state=seed
            ))
        ])
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5.0, min_child_weight=8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=seed, n_jobs=-1
            ))
        ])
    return models


def raw_probability(model, X):
    probabilities = model.predict_proba(X)[:, 1]
    return np.clip(probabilities, 1e-5, 1 - 1e-5)


def fit_chronological_calibrated_model(model, X, y, calibration_fraction=0.20,
                                       min_calibration_rows=60):
    """
    Fit base model on earlier observations; fit Platt logistic calibration
    on a later chronological calibration block. The calibration block is
    not used to fit the base estimator.
    """
    n = len(X)
    calibration_n = max(min_calibration_rows, int(n * calibration_fraction))
    split = n - calibration_n
    if split < 100 or calibration_n < 30:
        raise ValueError("Insufficient rows for chronological model calibration.")

    X_fit, X_cal = X.iloc[:split], X.iloc[split:]
    y_fit, y_cal = y[:split], y[split:]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
        raise ValueError("Calibration split must contain both target classes.")

    fitted_base = model
    fitted_base.fit(X_fit, y_fit)
    cal_p = raw_probability(fitted_base, X_cal)
    # Platt scaling on log-odds, fitted only on the later calibration block.
    cal_logit = np.log(cal_p / (1 - cal_p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=2000)
    calibrator.fit(cal_logit, y_cal)
    return fitted_base, calibrator


def calibrated_probability(fitted_base, calibrator, X):
    p = raw_probability(fitted_base, X)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def make_walk_forward_splits(n, min_train, n_splits, purge_rows=1):
    remaining = n - min_train
    if remaining < n_splits * 20:
        raise ValueError("Not enough training observations for walk-forward validation.")
    fold_size = remaining // n_splits
    splits = []
    for fold in range(n_splits):
        val_start = min_train + fold * fold_size
        val_end = n if fold == n_splits - 1 else min(n, val_start + fold_size)
        # Purge labels whose future execution windows overlap validation.
        train_end = val_start - purge_rows
        if train_end > 150 and val_end > val_start:
            splits.append((np.arange(train_end), np.arange(val_start, val_end)))
    return splits


def auc_score(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else 0.5


def position_sizes(volatility, cfg: Config):
    annual_target_daily = cfg.target_volatility / np.sqrt(cfg.annualization)
    estimated = np.maximum(np.asarray(volatility, dtype=float), 0.002)
    return np.minimum(cfg.max_position, annual_target_daily / estimated)


def portfolio_returns(probability, threshold, execution_returns, volatility, cfg: Config):
    active = (np.asarray(probability) >= threshold).astype(float)
    size = position_sizes(volatility, cfg)
    position = active * size
    round_trip_cost = 2 * cfg.cost_bps_per_side / 10000
    gross = position * np.asarray(execution_returns)
    costs = position * round_trip_cost
    net = gross - costs
    return net, active, position, costs


def portfolio_metrics(returns, annualization=252):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}
    equity = np.cumprod(1 + r)
    total = float(equity[-1] - 1)
    years = len(r) / annualization
    cagr = float(equity[-1] ** (1 / years) - 1) if years and equity[-1] > 0 else np.nan
    daily_std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    annual_vol = daily_std * np.sqrt(annualization)
    sharpe = float(np.mean(r) / daily_std * np.sqrt(annualization)) if daily_std > 0 else 0.0
    downside = np.minimum(r, 0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(annualization))
    sortino = float(np.mean(r) * annualization / downside_dev) if downside_dev > 0 else 0.0
    drawdown = equity / np.maximum.accumulate(equity) - 1
    var = float(np.quantile(r, 0.05))
    cvar = float(np.mean(r[r <= var])) if np.any(r <= var) else var
    mdd = float(np.min(drawdown))
    return {
        "total_return_pct": round(total * 100, 4),
        "cagr_pct": round(cagr * 100, 4) if np.isfinite(cagr) else None,
        "annualized_volatility_pct": round(annual_vol * 100, 4),
        "daily_var_95_pct": round(var * 100, 4),
        "daily_cvar_95_pct": round(cvar * 100, 4),
        "maximum_drawdown_pct": round(mdd * 100, 4),
        "sharpe_zero_rf": round(sharpe, 4),
        "sortino_zero_target": round(sortino, 4),
        "calmar_ratio": round(cagr / abs(mdd), 4) if mdd < 0 and np.isfinite(cagr) else None
    }


def trade_statistics(net_returns, active_mask):
    trades = np.asarray(net_returns)[np.asarray(active_mask, dtype=bool)]
    if len(trades) == 0:
        return {"trade_count": 0}
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    return {
        "trade_count": int(len(trades)),
        "win_rate_pct": round(float(np.mean(trades > 0) * 100), 4),
        "mean_trade_return_pct": round(float(np.mean(trades) * 100), 4),
        "median_trade_return_pct": round(float(np.median(trades) * 100), 4),
        "best_trade_pct": round(float(np.max(trades) * 100), 4),
        "worst_trade_pct": round(float(np.min(trades) * 100), 4),
        "mean_winning_trade_pct": round(float(np.mean(wins) * 100), 4) if len(wins) else None,
        "mean_losing_trade_pct": round(float(np.mean(losses) * 100), 4) if len(losses) else None
    }


def evaluate_models(train, cfg):
    X = train[FEATURES]
    y = train["Target"].astype(int).to_numpy()
    candidates = make_model_candidates(cfg.random_state)
    purge_rows = 1 if cfg.strategy == 'intraday' else 2
    splits = make_walk_forward_splits(len(train), cfg.min_train, cfg.n_splits, purge_rows)
    score_map, selection_scores, oof_map = {}, {}, {}

    print("\n--- CHRONOLOGICAL WALK-FORWARD MODEL EVALUATION ---")
    for name, model in candidates.items():
        oof = np.full(len(train), np.nan)
        fold_aucs = []
        for fit_idx, val_idx in splits:
            try:
                base, calibrator = fit_chronological_calibrated_model(
                    model, X.iloc[fit_idx], y[fit_idx],
                    cfg.calibration_fraction, cfg.min_calibration_rows
                )
                p = calibrated_probability(base, calibrator, X.iloc[val_idx])
                oof[val_idx] = p
                fold_aucs.append(auc_score(y[val_idx], p))
            except ValueError as exc:
                print(f"[!] Skipping {name} fold: {exc}")
        score_map[name] = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        oof_map[name] = oof

    # Separate model selection from economic-threshold selection:
    # choose the model using the earlier 70% of available OOF observations.
    valid_positions = np.flatnonzero(np.isfinite(next(iter(oof_map.values()))))
    if len(valid_positions) < 100:
        raise RuntimeError("Too few out-of-fold predictions for reliable model selection.")
    selection_end = valid_positions[max(0, int(len(valid_positions) * 0.70) - 1)]
    selection_scores = {}
    for name, oof in oof_map.items():
        mask = np.isfinite(oof) & (np.arange(len(oof)) <= selection_end)
        selection_scores[name] = auc_score(y[mask], oof[mask]) if mask.sum() else 0.5

    selected = max(selection_scores, key=selection_scores.get)
    print("Model AUC (mean across validation folds) | Selection AUC (earlier 70% OOF)")
    for name in candidates:
        selection_scores[name] = float(selection_scores[name])
        print(f"  {name}: {score_map[name]:.4f} | {selection_scores[name]:.4f}")
    print(f"[+] Selected model: {selected}")
    print("[+] Model selection uses the earlier 70% of OOF predictions.")
    print("[+] The later 30% is reserved for economic-threshold selection.")
    return selected, {
        "mean_fold_auc": score_map,
        "selection_period_auc": selection_scores
    }, oof_map[selected], candidates[selected], splits


def select_economic_threshold(train, oof_probability, cfg):
    valid_positions = np.flatnonzero(np.isfinite(oof_probability))
    if len(valid_positions) < 100:
        raise RuntimeError("Too few OOF predictions for threshold selection.")
    # This chronological segment was not used to select the winning model.
    threshold_start = valid_positions[int(len(valid_positions) * 0.70)]
    valid = np.isfinite(oof_probability) & (np.arange(len(oof_probability)) >= threshold_start)
    p = oof_probability[valid]
    returns = train["Execution_Return"].to_numpy()[valid]
    vol = train["Volatility_21D"].to_numpy()[valid]

    rows = []
    best_threshold, best_sharpe = None, -np.inf
    for threshold in np.arange(0.40, 0.701, 0.025):
        net, active, positions, costs = portfolio_returns(
            p, threshold, returns, vol, cfg
        )
        metrics = portfolio_metrics(net, cfg.annualization)
        exposure = float(np.mean(active)) if len(active) else 0.0
        trade_count = int(active.sum())
        sharpe = metrics.get("sharpe_zero_rf", 0.0)
        eligible = (
            exposure >= cfg.min_exposure
            and trade_count >= cfg.min_active_trades
            and sharpe > 0
            and metrics.get("total_return_pct", 0) > 0
        )
        rows.append({
            "threshold": round(float(threshold), 3),
            "oof_exposure_pct": round(exposure * 100, 4),
            "oof_active_trades": trade_count,
            "oof_net_return_pct": metrics.get("total_return_pct"),
            "oof_sharpe": sharpe,
            "eligible": bool(eligible)
        })
        if eligible and sharpe > best_sharpe:
            best_threshold, best_sharpe = float(threshold), sharpe

    table = pd.DataFrame(rows)
    if best_threshold is None:
        print("[!] No threshold met minimum exposure, trade-count and positive-performance rules.")
        print("[!] Portfolio decision: CASH / NO TRADE.")
        return None, table
    return best_threshold, table


def prediction_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy_pct": round(accuracy_score(y_true, prediction) * 100, 4),
        "balanced_accuracy_pct": round(balanced_accuracy_score(y_true, prediction) * 100, 4),
        "precision_up_pct": round(precision_score(y_true, prediction, zero_division=0) * 100, 4),
        "recall_up_pct": round(recall_score(y_true, prediction, zero_division=0) * 100, 4),
        "f1_up": round(f1_score(y_true, prediction, zero_division=0), 4),
        "roc_auc": round(auc_score(y_true, probability), 4),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 5),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist()
    }


def make_benchmark_rows(test, cfg):
    execution = test["Execution_Return"].to_numpy()
    all_in_metrics = portfolio_metrics(execution, cfg.annualization)
    rows = [{"benchmark": "Always_Invested_Same_Execution_Window", **all_in_metrics}]
    rows.append({
        "benchmark": "Cash",
        **portfolio_metrics(np.zeros(len(test)), cfg.annualization)
    })
    return pd.DataFrame(rows)


def validation_feature_importance(model, train, selected_splits, cfg):
    """
    Validation-fold permutation importance using ROC-AUC.
    This direct implementation avoids sklearn estimator-interface requirements
    for calibrated wrappers. Negative values mean permutation improved AUC.
    """
    X, y = train[FEATURES], train["Target"].astype(int).to_numpy()
    fold_importances = []
    rng = np.random.default_rng(cfg.random_state)

    for fit_idx, val_idx in selected_splits:
        try:
            base, calibrator = fit_chronological_calibrated_model(
                model, X.iloc[fit_idx], y[fit_idx],
                cfg.calibration_fraction, cfg.min_calibration_rows
            )
            X_val = X.iloc[val_idx].copy()
            y_val = y[val_idx]
            if len(np.unique(y_val)) < 2:
                continue

            baseline_p = calibrated_probability(base, calibrator, X_val)
            baseline_auc = roc_auc_score(y_val, baseline_p)
            importance = []

            for feature in FEATURES:
                drops = []
                original = X_val[feature].to_numpy(copy=True)
                for _ in range(5):
                    permuted = X_val.copy()
                    permuted[feature] = rng.permutation(original)
                    permuted_p = calibrated_probability(base, calibrator, permuted)
                    permuted_auc = roc_auc_score(y_val, permuted_p)
                    drops.append(baseline_auc - permuted_auc)
                importance.append(float(np.mean(drops)))

            fold_importances.append(importance)
        except Exception as exc:
            print(f"[!] Feature-importance fold skipped: {exc}")

    if not fold_importances:
        return pd.DataFrame(columns=["Feature", "Importance_Mean", "Importance_Std"])

    values = np.vstack(fold_importances)
    return pd.DataFrame({
        "Feature": FEATURES,
        "Importance_Mean": values.mean(axis=0),
        "Importance_Std": values.std(axis=0)
    }).sort_values("Importance_Mean", ascending=False)



def regime_performance_diagnostics(test, probability, cfg):
    """Daily regime statistics plus returns compounded only within contiguous episodes."""
    frame = test[["Market_Regime", "Execution_Return", "Target"]].copy()
    frame["Probability_UP"] = np.asarray(probability)
    rows = []
    regime_masks = {
        "BULLISH": frame["Market_Regime"].to_numpy() == "BULLISH",
        "BEARISH": frame["Market_Regime"].to_numpy() == "BEARISH",
        "SIDEWAYS": frame["Market_Regime"].to_numpy() == "SIDEWAYS",
        "HIGH_VOLATILITY": test["Regime_HighVol"].to_numpy() == 1.0
    }

    for regime, mask in regime_masks.items():
        group = frame.loc[mask]
        r = group["Execution_Return"].to_numpy(dtype=float)
        daily_mean = float(np.mean(r)) if len(r) else np.nan
        daily_std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        annual_vol = daily_std * np.sqrt(cfg.annualization)
        sharpe = daily_mean / daily_std * np.sqrt(cfg.annualization) if daily_std > 0 else 0.0

        # Identify contiguous runs in the original chronological sequence.
        episode_ids = np.cumsum(np.r_[True, mask[1:] != mask[:-1]])
        episode_returns = []
        for episode_id in np.unique(episode_ids[mask]):
            episode_mask = mask & (episode_ids == episode_id)
            episode_r = frame.loc[episode_mask, "Execution_Return"].to_numpy(dtype=float)
            if len(episode_r):
                episode_returns.append(float(np.prod(1 + episode_r) - 1))

        rows.append({
            "regime": regime,
            "observations": int(len(group)),
            "actual_up_rate_pct": round(float(group["Target"].mean() * 100), 4),
            "mean_predicted_probability_up": round(float(group["Probability_UP"].mean()), 6),
            "mean_daily_return_pct": round(daily_mean * 100, 5),
            "annualized_volatility_pct": round(annual_vol * 100, 4),
            "daily_sharpe_zero_rf": round(sharpe, 4),
            "episode_count": len(episode_returns),
            "best_episode_return_pct": round(max(episode_returns) * 100, 4) if episode_returns else None,
            "worst_episode_return_pct": round(min(episode_returns) * 100, 4) if episode_returns else None
        })
    return pd.DataFrame(rows)


def regime_specific_walk_forward(selected_model, train, splits, cfg):
    """Exploratory regime-specific refits; never used to select the deployed model."""
    rows = []
    regimes = ["BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOLATILITY"]
    for regime in regimes:
        fold_aucs, fold_counts = [], []
        for fit_idx, val_idx in splits:
            fit_frame, val_frame = train.iloc[fit_idx], train.iloc[val_idx]
            if regime == "HIGH_VOLATILITY":
                fit_mask = fit_frame["Regime_HighVol"].to_numpy() == 1.0
                val_mask = val_frame["Regime_HighVol"].to_numpy() == 1.0
            else:
                fit_mask = fit_frame["Market_Regime"].to_numpy() == regime
                val_mask = val_frame["Market_Regime"].to_numpy() == regime
            regime_fit_idx = np.asarray(fit_idx)[fit_mask]
            regime_val_idx = np.asarray(val_idx)[val_mask]
            if len(regime_fit_idx) < max(250, cfg.min_train // 2) or len(regime_val_idx) < 20:
                continue
            X_fit = train.iloc[regime_fit_idx][FEATURES]
            y_fit = train.iloc[regime_fit_idx]["Target"].astype(int).to_numpy()
            X_val = train.iloc[regime_val_idx][FEATURES]
            y_val = train.iloc[regime_val_idx]["Target"].astype(int).to_numpy()
            try:
                base, calibrator = fit_chronological_calibrated_model(
                    selected_model, X_fit, y_fit,
                    cfg.calibration_fraction, cfg.min_calibration_rows
                )
                p = calibrated_probability(base, calibrator, X_val)
                if len(np.unique(y_val)) == 2:
                    fold_aucs.append(roc_auc_score(y_val, p))
                    fold_counts.append(len(y_val))
            except (ValueError, IndexError):
                continue
        rows.append({
            "regime": regime,
            "valid_folds": len(fold_aucs),
            "validation_observations": int(sum(fold_counts)),
            "mean_validation_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
            "std_validation_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
            "interpretation": "Insufficient regime-specific validation data" if not fold_aucs
                else "Exploratory only; compare with global model and outer holdout"
        })
    return pd.DataFrame(rows)


def feature_group_experiment(train, selected_model, splits, cfg):
    """Exploratory chronological feature-family comparison."""
    groups = {
        "PRICE_ACTION": ["Return_1D", "Return_5D", "Return_10D", "Return_21D", "High_Low_Range", "ATR_Pct"],
        "VOLATILITY": ["Volatility_10D", "Volatility_21D", "ATR_Pct", "Regime_HighVol"],
        "MOMENTUM": ["Return_5D", "Return_10D", "Return_21D", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Position"],
        "TREND": ["SMA_Ratio_20", "SMA_Ratio_50", "SMA_Ratio_200", "Regime_Trend"],
        "VOLUME": ["Volume_Change", "Volume_Ratio_20"]
    }
    rows = []
    y_all = train["Target"].astype(int).to_numpy()
    for group_name, columns in groups.items():
        fold_scores = []
        for fit_idx, val_idx in splits:
            try:
                base, calibrator = fit_chronological_calibrated_model(
                    selected_model, train.iloc[fit_idx][columns], y_all[fit_idx],
                    cfg.calibration_fraction, cfg.min_calibration_rows
                )
                y_val = y_all[val_idx]
                if len(np.unique(y_val)) < 2:
                    continue
                p = calibrated_probability(base, calibrator, train.iloc[val_idx][columns])
                fold_scores.append(roc_auc_score(y_val, p))
            except (ValueError, IndexError):
                continue
        rows.append({
            "feature_group": group_name,
            "feature_count": len(columns),
            "valid_folds": len(fold_scores),
            "mean_validation_auc": round(float(np.mean(fold_scores)), 4) if fold_scores else None,
            "std_validation_auc": round(float(np.std(fold_scores)), 4) if fold_scores else None,
            "features": ";".join(columns)
        })
    return pd.DataFrame(rows)

def run():
    parser = argparse.ArgumentParser(description="Financial AI System V2.2.7")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cost-bps-per-side", type=float, default=5.0)
    parser.add_argument("--strategy", choices=["intraday", "close-close"], default="intraday")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default="results_v227")
    args = parser.parse_args()

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_size=args.test_size, cost_bps_per_side=args.cost_bps_per_side,
        output_dir=args.output_dir, strategy=args.strategy
    )

    raw = make_synthetic_data(seed=cfg.random_state) if args.synthetic else download_data(cfg)
    data = build_dataset(raw, args.strategy)
    print(f"[+] Feature dataset: {len(data):,} rows; {len(FEATURES)} features.")
    if len(data) < cfg.min_train + 150:
        raise RuntimeError("Insufficient observations after feature engineering.")

    split = int(len(data) * (1 - cfg.test_size))
    purge_rows = 1 if args.strategy == "intraday" else 2
    train = data.iloc[:split - purge_rows].copy()
    test = data.iloc[split:].copy()
    print(f"Training rows: {len(train):,}; purged rows: {purge_rows}; test rows: {len(test):,}")
    print(f"Training: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Signal holdout: {test.index.min().date()} to {test.index.max().date()}")
    print(f"Execution: {pd.to_datetime(test['Execution_Date']).min().date()} to {pd.to_datetime(test['Execution_Date']).max().date()}")

    selected_name, scores, oof, selected_model, splits = evaluate_models(train, cfg)
    threshold, threshold_table = select_economic_threshold(train, oof, cfg)

    X_train, y_train = train[FEATURES], train["Target"].astype(int).to_numpy()
    X_test, y_test = test[FEATURES], test["Target"].astype(int).to_numpy()

    # Final model/calibrator are fit chronologically on training data only.
    final_base, final_calibrator = fit_chronological_calibrated_model(
        selected_model, X_train, y_train,
        cfg.calibration_fraction, cfg.min_calibration_rows
    )
    test_probability = calibrated_probability(final_base, final_calibrator, X_test)

    if threshold is None:
        active = np.zeros(len(test), dtype=int)
        positions = np.zeros(len(test), dtype=float)
        costs = np.zeros(len(test), dtype=float)
        net = np.zeros(len(test), dtype=float)
        gross = np.zeros(len(test), dtype=float)
        decision = "CASH_NO_TRADE"
    else:
        net, active, positions, costs = portfolio_returns(
            test_probability, threshold, test["Execution_Return"].to_numpy(),
            test["Volatility_21D"].to_numpy(), cfg
        )
        gross = positions * test["Execution_Return"].to_numpy()
        decision = "TRADE_ALLOWED"

    # Report directional classification at a neutral 0.50 cutoff; trading threshold
    # is a separate portfolio decision and must not distort prediction diagnostics.
    pred_metrics = prediction_metrics(y_test, test_probability, 0.50)
    majority_class = int(np.mean(y_test) >= 0.5)
    majority_baseline_accuracy = float(np.mean(y_test == majority_class) * 100)
    pred_metrics["actual_up_rate_pct"] = round(float(np.mean(y_test) * 100), 4)
    pred_metrics["majority_class_baseline_accuracy_pct"] = round(majority_baseline_accuracy, 4)
    pred_metrics["predicted_up_rate_pct"] = round(float(np.mean(test_probability >= 0.50) * 100), 4)
    base_rate = float(np.mean(y_test))
    base_brier = base_rate * (1.0 - base_rate)
    model_brier = float(brier_score_loss(y_test, test_probability))
    pred_metrics["base_rate_brier_reference"] = round(base_brier, 6)
    pred_metrics["brier_skill_score_vs_base_rate"] = round(
        float(1.0 - model_brier / max(base_brier, 1e-12)), 6
    )
    strategy_metrics = portfolio_metrics(net, cfg.annualization)
    market_metrics = portfolio_metrics(test["Execution_Return"].to_numpy(), cfg.annualization)

    backtest = pd.DataFrame({
        "Signal_Date": test["Signal_Date"].values,
        "Execution_Date": test["Execution_Date"].values,
        "Probability_UP": test_probability,
        "Trading_Threshold": threshold if threshold is not None else "NO_TRADE",
        "Signal": active,
        "Position_Size": positions,
        "Execution_Return": test["Execution_Return"].to_numpy(),
        "Gross_Strategy_Return": gross,
        "Transaction_Cost": costs,
        "Net_Strategy_Return": net
    })

    benchmarks = make_benchmark_rows(test, cfg)

    # Regime metrics preserve chronological ordering and contiguous episodes.
    regime_diagnostics = regime_performance_diagnostics(test, test_probability, cfg)
    regime_model_diagnostics = regime_specific_walk_forward(
        selected_model, train, splits, cfg
    )
    feature_group_results = feature_group_experiment(
        train, selected_model, splits, cfg
    )

    feature_importance = validation_feature_importance(
        selected_model, train, splits, cfg
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.ticker.lower()
    backtest.to_csv(out_dir / f"{prefix}_v227_backtest.csv", index=False)
    threshold_table.to_csv(out_dir / f"{prefix}_v227_threshold_analysis.csv", index=False)
    benchmarks.to_csv(out_dir / f"{prefix}_v227_benchmarks.csv", index=False)
    feature_importance.to_csv(out_dir / f"{prefix}_v227_feature_importance.csv", index=False)
    regime_diagnostics.to_csv(out_dir / f"{prefix}_v227_regime_diagnostics.csv", index=False)
    regime_model_diagnostics.to_csv(out_dir / f"{prefix}_v227_regime_model_comparison.csv", index=False)
    feature_group_results.to_csv(out_dir / f"{prefix}_v227_feature_group_experiments.csv", index=False)

    active_day_returns = net[np.asarray(active, dtype=bool)]
    diagnostics = {
        "probability_mean": round(float(np.mean(test_probability)), 6),
        "probability_std": round(float(np.std(test_probability)), 6),
        "probability_min": round(float(np.min(test_probability)), 6),
        "probability_max": round(float(np.max(test_probability)), 6),
        "probability_above_threshold_pct": round(float(np.mean(active) * 100), 4)
    }
    metrics = {
        "version": "2.2.7",
        "release_status": "FINAL_PLANNED_DEVELOPMENT_RELEASE",
        "project_policy": "After V2.2.7, proceed to testing, acceptance, documentation and freeze. Do not create another version unless a material defect is confirmed.",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_design": {
            "model_selection": "Earlier 70% of finite chronological out-of-fold predictions",
            "threshold_selection": "Later 30% of finite chronological out-of-fold predictions",
            "final_holdout_used_for_selection": False,
            "purge_rows": 1 if args.strategy == "intraday" else 2,
            "note": "This is a chronological holdout design, not a fully nested outer walk-forward estimate. Threshold selection can still be optimistic."
        },
        "ticker": cfg.ticker,
        "strategy": args.strategy,
        "synthetic_data": bool(args.synthetic),
        "selected_model": selected_name,
        "walk_forward_model_scores": scores,
        "regime_diagnostics_file": f"{prefix}_v227_regime_diagnostics.csv",
        "regime_model_comparison_file": f"{prefix}_v227_regime_model_comparison.csv",
        "feature_group_experiments_file": f"{prefix}_v227_feature_group_experiments.csv",
        "research_note": "Regime-specific and feature-group results are exploratory diagnostics; they do not select the deployed model or establish profitability.",
        "holdout_target_up_rate_pct": round(float(test["Target"].mean() * 100), 4),
        "decision": decision,
        "selected_threshold": threshold,
        "prediction_reporting_threshold": 0.50,
        "threshold_rule": (
            "Model is selected using the earlier 70% of finite chronological OOF predictions. "
            "Threshold is selected on the later 30%; requires exposure >= 10%, at least 20 active "
            "observations, positive net return and positive Sharpe; otherwise cash. "
            "This separation reduces, but does not eliminate, selection bias."
        ),
        "prediction_metrics": pred_metrics,
        "strategy_risk_metrics": strategy_metrics,
        "same_execution_window_market_metrics": market_metrics,
        "trade_statistics": trade_statistics(net, active),
        "trading_metrics": {
            "market_exposure_pct": round(float(np.mean(active) * 100), 4),
            "average_position_pct": round(float(np.mean(positions) * 100), 4),
            "active_trades": int(np.sum(active)),
            "round_trip_cost_bps": 2 * cfg.cost_bps_per_side,
            "cost_per_side_bps": cfg.cost_bps_per_side
        },
        "probability_diagnostics": diagnostics,
        "periods": {
            "training_start": str(train.index.min().date()),
            "training_end": str(train.index.max().date()),
            "signal_test_start": str(test.index.min().date()),
            "signal_test_end": str(test.index.max().date()),
            "execution_start": str(pd.to_datetime(test["Execution_Date"]).min().date()),
            "execution_end": str(pd.to_datetime(test["Execution_Date"]).max().date())
        },
        "configuration": asdict(cfg)
    }
    with open(out_dir / f"{prefix}_v227_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n--- V2.2.7 HOLDOUT RESULTS ---")
    print(f"Selected model: {selected_name}")
    print(f"Decision: {decision}")
    print(f"Threshold: {threshold if threshold is not None else 'NONE'}")
    print(f"Accuracy: {pred_metrics['accuracy_pct']:.2f}%")
    print(f"ROC-AUC: {pred_metrics['roc_auc']:.4f}")
    print(f"Brier score: {pred_metrics['brier_score']:.5f}")
    print(f"Strategy net return: {strategy_metrics['total_return_pct']:.4f}%")
    print(f"Market return, same execution window: {market_metrics['total_return_pct']:.4f}%")
    print(f"Exposure: {np.mean(active) * 100:.2f}%")
    print(f"Active trades: {int(np.sum(active))}")
    print("\n--- REGIME DIAGNOSTICS ---")
    print(regime_diagnostics.to_string(index=False))
    print("\n--- FEATURE-GROUP EXPERIMENTS ---")
    print(feature_group_results.to_string(index=False))
    print("\n--- REGIME-SPECIFIC WALK-FORWARD EXPERIMENTS ---")
    print(regime_model_diagnostics.to_string(index=False))
    print("\n--- SAVED OUTPUTS ---")
    for path in sorted(out_dir.iterdir()):
        print(f"[+] {path}")


if __name__ == "__main__":
    run()