#!/usr/bin/env python3
"""Train the DMMM XGBoost response model on a preprocessed table.

Faithful, self-contained port of DMMM.train + response_model.train_XGB from the
AMCIS package, but headless (no plotting / progress bar) and emitting a JSON
summary on stdout.

Pipeline:
  1. Time-based train/valid split by YMW (validation = earliest 1/N of weeks),
     matching DMMM.train.
  2. LabelEncoder fit on train+valid categories (so every channel is known).
  3. Optuna (TPE) search minimizing validation RMSE, same search space as the paper.
  4. Refit best params on the train fold; pickle (predictor, label_encoders, rmse).
  5. Optional: evaluate on a held-out test CSV (daily-aggregated RMSE / MAPE / bias).

Input is the output of preprocess.py (columns: Channel, Date, Cost, <KPI>, Year,
Month, WeekNum, Weekday, YMW).

Dependencies: pandas, numpy, xgboost, optuna, scikit-learn.
"""

import argparse
import json
import random
import sys

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


def split_by_ymw(df: pd.DataFrame, valid_fraction_denom: int):
    """Earliest 1/denom of distinct YMW weeks -> validation; rest -> train.

    Mirrors DMMM.train: valid = first weeks, train = later weeks.
    """
    n_valid = max(1, df["YMW"].nunique() // valid_fraction_denom)
    ymw_sorted = sorted(df["YMW"].unique())
    valid_ymw = ymw_sorted[:n_valid]
    train_ymw = ymw_sorted[n_valid:]
    trainset = df[df["YMW"].isin(train_ymw)].copy()
    validset = df[df["YMW"].isin(valid_ymw)].copy()
    return trainset, validset


def train_xgb(kpi, numerical_features, categorical_features, trainset, validset,
              timeout, n_trials, seed):
    np.random.seed(seed)
    random.seed(seed)
    xgb.set_config(verbosity=0)

    features = numerical_features + categorical_features
    cat_all = pd.concat([trainset[categorical_features], validset[categorical_features]])

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder().fit(cat_all[col])
        label_encoders[col] = le
        for ds in (trainset, validset):
            ds[col] = le.transform(ds[col])
            ds[col] = ds[col].astype("category")

    xgb_train = xgb.DMatrix(trainset[features], label=trainset[kpi], enable_categorical=True)
    xgb_valid = xgb.DMatrix(validset[features], label=validset[kpi], enable_categorical=True)
    y_valid = validset[kpi]

    def objective(trial):
        param = {
            "objective": trial.suggest_categorical(
                "objective", ["reg:squarederror", "reg:absoluteerror", "reg:tweedie"]),
            "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
            "device": "cpu",
            "eta": trial.suggest_float("eta", 1e-2, 1),
            "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),
            "num_rounds": 5000,
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 5, 20),
            "seed": seed,
            "verbosity": 0,
        }
        bst = xgb.train(param, xgb_train,
                        evals=[(xgb_train, "train"), (xgb_valid, "eval")],
                        verbose_eval=False)
        preds = bst.predict(xgb_valid)
        return root_mean_squared_error(y_valid, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=seed),
                                pruner=HyperbandPruner())
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = dict(study.best_trial.params)
    best_params.update({"seed": seed, "num_rounds": 5000, "verbosity": 0})
    predictor = xgb.train(best_params, xgb_train)

    pred = predictor.predict(xgb.DMatrix(validset[features], enable_categorical=True))
    rmse = float(root_mean_squared_error(y_valid, pred))
    return predictor, label_encoders, rmse, best_params, len(study.trials)


def evaluate(predictor, label_encoders, kpi, numerical_features, categorical_features,
             test_df):
    """Daily-aggregated test metrics, matching notebook cell-6."""
    features = numerical_features + categorical_features
    df = test_df.copy()
    # drop test rows whose channel is unknown to the encoder (would crash transform)
    known = set(label_encoders["Channel"].classes_)
    unknown = sorted(set(df["Channel"]) - known)
    df = df[df["Channel"].isin(known)].copy()
    df["Channel"] = label_encoders["Channel"].transform(df["Channel"])
    pred = predictor.predict(xgb.DMatrix(df[features], enable_categorical=True))

    daily = pd.DataFrame({"Date": test_df.loc[df.index, "Date"].values,
                          "y_true": df[kpi].values, "y_pred": pred})
    daily = daily.groupby("Date")[["y_true", "y_pred"]].sum()
    return {
        "rmse": float(root_mean_squared_error(daily["y_true"], daily["y_pred"])),
        "mape": float(mean_absolute_percentage_error(daily["y_true"], daily["y_pred"])),
        "bias": float((daily["y_pred"].sum() - daily["y_true"].sum()) / daily["y_true"].sum()),
        "skipped_unknown_channels": unknown,
    }


def main():
    p = argparse.ArgumentParser(description="Train the DMMM XGBoost response model")
    p.add_argument("--input", "-i", required=True,
                   help="Preprocessed CSV (from preprocess.py)")
    p.add_argument("--model-out", "-o", default="dmmm_model.pkl",
                   help="Output path for the trained model pickle")
    p.add_argument("--kpi", default="Install", help="KPI / target column (default: Install)")
    p.add_argument("--numerical-features", default="Cost",
                   help="Comma-separated numerical features (default: Cost)")
    p.add_argument("--categorical-features", default="Channel",
                   help="Comma-separated categorical features (default: Channel)")
    p.add_argument("--train-time", type=int, default=30,
                   help="Optuna timeout in seconds (default: 30)")
    p.add_argument("--n-trials", type=int, default=5000,
                   help="Max Optuna trials (default: 5000; usually capped by --train-time)")
    p.add_argument("--valid-fraction-denom", type=int, default=5,
                   help="Validation = earliest 1/N of weeks (default: 5 => ~20%%)")
    p.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    p.add_argument("--test-input", default=None,
                   help="Optional held-out test CSV for daily-aggregated metrics")
    args = p.parse_args()

    numerical = [c.strip() for c in args.numerical_features.split(",") if c.strip()]
    categorical = [c.strip() for c in args.categorical_features.split(",") if c.strip()]

    df = pd.read_csv(args.input)
    for col in [args.kpi, *numerical, *categorical, "YMW"]:
        if col not in df.columns:
            raise SystemExit(
                f"ERROR: column '{col}' not in input. Columns: {list(df.columns)}")

    trainset, validset = split_by_ymw(df, args.valid_fraction_denom)
    if len(validset) == 0 or len(trainset) == 0:
        raise SystemExit("ERROR: train/valid split produced an empty fold "
                         "(too few distinct weeks). Provide more data.")

    predictor, label_encoders, rmse, best_params, n_done = train_xgb(
        args.kpi, numerical, categorical, trainset, validset,
        args.train_time, args.n_trials, args.seed)

    pd.to_pickle((predictor, label_encoders, rmse), args.model_out)

    summary = {
        "model_file": args.model_out,
        "kpi": args.kpi,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "n_train_rows": int(len(trainset)),
        "n_valid_rows": int(len(validset)),
        "channels": sorted(label_encoders["Channel"].classes_.tolist()),
        "validation_rmse": round(rmse, 4),
        "optuna_trials_run": n_done,
        "best_params": best_params,
    }

    if args.test_input:
        test_df = pd.read_csv(args.test_input)
        summary["test_metrics"] = evaluate(
            predictor, label_encoders, args.kpi, numerical, categorical, test_df)

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
