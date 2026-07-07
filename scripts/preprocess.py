#!/usr/bin/env python3
"""Preprocess raw marketing data into the training table the DMMM (XGBoost) model expects.

Replicates the inline preprocessing from the DMMM notebooks (Fix_DMMM_budget_225.ipynb,
cell-2) as a reusable, parameterized CLI:

  1. Standardize columns        (Date_->Date, Cost_->Cost, <kpi raw>-><kpi clean>)
  2. Build the Channel column    (from --channel-source, lowercased)
  3. Aggregate to (Channel x Date)  (sum Cost and KPI; collapses campaign-level rows)
  4. Derive date features + YMW  (Year, Month, WeekNum, Weekday, YMW)
  5. Validate / warn             (NaN & negative counts; train/test channel coverage)

Outputs a cleaned CSV and prints a JSON summary to stdout for the caller to report on.

Dependencies: pandas, numpy (rest is stdlib).
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_week_no(date: datetime) -> int:
    """Week-of-month index used by DMMM (copied verbatim from utils.get_week_no)."""
    firstday = date.replace(day=1)
    if firstday.weekday() == 0:
        origin = firstday
    elif 0 < firstday.weekday() < 4:
        origin = firstday - timedelta(days=firstday.weekday())
    else:
        origin = firstday + timedelta(days=7 - firstday.weekday())
    return (date - origin).days // 7 + 1


def clean_kpi_name(raw: str) -> str:
    """'Install(Total)' -> 'Install'; strips a trailing parenthetical and whitespace."""
    name = raw.split("(")[0].strip()
    return name or raw


def preprocess(
    df: pd.DataFrame,
    kpi_raw: str,
    cost_col: str,
    date_col: str,
    channel_source: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Return (clean_df, kpi_name, warnings)."""
    warnings: list[str] = []
    kpi_name = clean_kpi_name(kpi_raw)

    # 1. Standardize columns
    missing = [c for c in (date_col, cost_col, kpi_raw, channel_source) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: columns not found in input: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={date_col: "Date", cost_col: "Cost", kpi_raw: kpi_name})

    # 2. Build Channel (lowercased), the single categorical feature / encoder key
    df["Channel"] = df[channel_source].astype(str).str.lower()

    # 3. Aggregate to (Channel x Date), summing Cost and the KPI
    df["Date"] = df["Date"].astype("category")
    df["Channel"] = df["Channel"].astype("category")
    agg = (
        df.groupby(["Channel", "Date"], observed=True)[["Cost", kpi_name]]
        .sum()
        .reset_index()
    )
    agg["Date"] = agg["Date"].astype("str")

    # 4. Date features + YMW
    dt = pd.to_datetime(agg["Date"])
    agg["Year"] = dt.dt.year
    agg["Month"] = dt.dt.month
    agg["WeekNum"] = dt.apply(get_week_no)
    agg["Weekday"] = dt.dt.weekday
    agg["YMW"] = agg["Year"] * 10 ** 4 + agg["Month"] * 10 ** 2 + agg["WeekNum"]

    # 5. Validation / warnings
    for col in ("Cost", kpi_name):
        n_nan = int(agg[col].isna().sum())
        n_neg = int((agg[col] < 0).sum())
        if n_nan:
            warnings.append(f"{col}: {n_nan} NaN value(s) after aggregation")
        if n_neg:
            warnings.append(f"{col}: {n_neg} negative value(s) after aggregation")

    return agg, kpi_name, warnings


def channel_counts(df: pd.DataFrame) -> dict:
    return {str(k): int(v) for k, v in df["Channel"].value_counts().sort_index().items()}


def build_summary(
    raw_rows: int,
    clean: pd.DataFrame,
    kpi_name: str,
    warnings: list[str],
) -> dict:
    return {
        "input_rows": raw_rows,
        "output_rows": int(len(clean)),
        "kpi": kpi_name,
        "n_channels": int(clean["Channel"].nunique()),
        "channels": sorted(clean["Channel"].unique().tolist()),
        "date_range": [str(clean["Date"].min()), str(clean["Date"].max())],
        "rows_per_channel": channel_counts(clean),
        "warnings": warnings,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="DMMM (XGBoost) training-data preprocessor")
    p.add_argument("--input", "-i", required=True, help="Path to the raw CSV")
    p.add_argument("--output", "-o", default="dmmm_preprocessed.csv",
                   help="Path for the cleaned CSV (default: dmmm_preprocessed.csv)")
    p.add_argument("--kpi", default="Install(Total)",
                   help="Raw KPI column name (default: 'Install(Total)')")
    p.add_argument("--cost-col", default="Cost_", help="Raw cost column (default: 'Cost_')")
    p.add_argument("--date-col", default="Date_", help="Raw date column (default: 'Date_')")
    p.add_argument("--channel-source", default="Media",
                   help="Column used to build Channel (default: 'Media')")
    p.add_argument("--train-cutoff", default=None,
                   help="Date (YYYY-MM-DD). If set, split into train (Date<=cutoff) and "
                        "test, write *_train.csv / *_test.csv, and report split stats.")
    args = p.parse_args()

    try:
        raw = pd.read_csv(args.input, low_memory=False)
    except Exception as e:  # noqa: BLE001 - surface a clean message to the caller
        raise SystemExit(f"ERROR: could not read input CSV '{args.input}': {e}")

    clean, kpi_name, warnings = preprocess(
        raw, args.kpi, args.cost_col, args.date_col, args.channel_source
    )
    clean.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = build_summary(len(raw), clean, kpi_name, warnings)
    summary["output_file"] = args.output

    if args.train_cutoff:
        cutoff = args.train_cutoff
        train = clean[clean["Date"] <= cutoff]
        test = clean[clean["Date"] > cutoff]
        base = args.output.rsplit(".", 1)[0]
        train_path, test_path = f"{base}_train.csv", f"{base}_test.csv"
        train.to_csv(train_path, index=False, encoding="utf-8-sig")
        test.to_csv(test_path, index=False, encoding="utf-8-sig")

        train_channels = set(train["Channel"].unique())
        test_channels = set(test["Channel"].unique())
        unseen = sorted(test_channels - train_channels)
        if unseen:
            warnings.append(
                "Channels present in test but NOT in train (XGB LabelEncoder will fail "
                f"on these): {unseen}"
            )
        summary["split"] = {
            "train_cutoff": cutoff,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_file": train_path,
            "test_file": test_path,
            "test_only_channels": unseen,
        }
        summary["warnings"] = warnings

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
