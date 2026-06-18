#!/usr/bin/env python3
"""
LAB 16 - CPU fallback benchmark (Plan B): LightGBM on Credit Card Fraud Detection.

Trains a LightGBM gradient-boosting classifier on the Kaggle
`mlg-ulb/creditcardfraud` dataset (284,807 transactions, highly imbalanced),
then measures training time, classification quality, and inference speed.

Produces:
  - Console output with every metric from README Part 7.6
  - benchmark_result.json with the full metric set (submission deliverable)

Usage (on the n2-standard-8 VM, after `kaggle datasets download ... --unzip`):
    python3 benchmark.py
    python3 benchmark.py --data ~/ml-benchmark/creditcard.csv
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

TARGET_COL = "Class"
RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM fraud-detection benchmark")
    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Path to the creditcard.csv dataset (default: ./creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_result.json",
        help="Where to write the JSON metrics (default: ./benchmark_result.json)",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------- load
    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_time = time.perf_counter() - t0
    print(f"[1/4] Loaded {len(df):,} rows x {df.shape[1]} cols in {load_time:.2f}s")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"      Class balance: {n_pos:,} fraud / {n_neg:,} legit "
          f"({100 * n_pos / len(df):.3f}% positive)")

    # ------------------------------------------------------------- split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --------------------------------------------------------------- training
    # scale_pos_weight handles the extreme class imbalance.
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_time = time.perf_counter() - t0
    best_iteration = model.best_iteration
    print(f"[2/4] Trained in {train_time:.2f}s "
          f"(best iteration: {best_iteration})")

    # --------------------------------------------------------------- evaluate
    y_proba = model.predict(X_test, num_iteration=best_iteration)
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"[3/4] AUC-ROC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  "
          f"Precision={precision:.4f}  Recall={recall:.4f}")

    # ------------------------------------------------------ inference timing
    # Single-row latency: average over many calls for a stable number.
    one_row = X_test.iloc[[0]]
    for _ in range(10):  # warm-up
        model.predict(one_row, num_iteration=best_iteration)
    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict(one_row, num_iteration=best_iteration)
    latency_1_row_ms = (time.perf_counter() - t0) / n_iter * 1000.0

    # Throughput: 1000 rows in one batch.
    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    model.predict(batch, num_iteration=best_iteration)
    batch_time = time.perf_counter() - t0
    throughput_1000 = len(batch) / batch_time
    print(f"[4/4] Latency(1 row)={latency_1_row_ms:.3f}ms  "
          f"Throughput(1000 rows)={throughput_1000:,.0f} rows/s")

    # ------------------------------------------------------------- write JSON
    result = {
        "dataset": "mlg-ulb/creditcardfraud",
        "model": "LightGBM (gbdt, binary)",
        "instance": "n2-standard-8 (8 vCPU / 32 GB)",
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "fraud_ratio": round(n_pos / len(df), 6),
        "load_time_sec": round(load_time, 3),
        "train_time_sec": round(train_time, 3),
        "best_iteration": int(best_iteration),
        "auc_roc": round(float(auc), 4),
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "inference_latency_1row_ms": round(latency_1_row_ms, 4),
        "inference_throughput_1000rows_per_sec": round(throughput_1000, 1),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
