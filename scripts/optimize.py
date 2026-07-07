#!/usr/bin/env python3
"""Optimize channel budget allocation with a trained DMMM model.

Self-contained port of DMMM.daily_allocation + utils.create_objective: given a
trained model, a total budget, and per-channel min/max limits, find the channel
allocation that MAXIMIZES predicted KPI (single planning period).

Per-channel bounds come from either:
  * a constraints CSV  (--constraints; columns: Channel, Lower_limit, Upper_limit), or
  * auto-derived from the data  (--data; historic mean daily spend +/- --bound-pct),
    matching the notebook's historic_mean*0.8 / *1.2.

If --data is given, also reports the baseline (historic-mean allocation) predicted
KPI so the improvement from optimizing can be stated in plain language.

Dependencies: pandas, numpy, optuna, xgboost, scikit-learn.
"""

import argparse
import json
import sys

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def predict(predictor, label_encoders, channels, costs, features,
            weeknum, weekday):
    """Predict KPI for each (channel, cost) pair."""
    df = pd.DataFrame({"Channel": list(channels), "Cost": list(costs),
                       "WeekNum": weeknum, "Weekday": weekday})
    df["Channel"] = label_encoders["Channel"].transform(df["Channel"])
    return predictor.predict(xgb.DMatrix(df[features], enable_categorical=True))


def make_objective(predictor, label_encoders, channels, features, low, high,
                   total_budget, weeknum, weekday, store):
    """Faithful port of utils.create_objective (maximize predicted KPI)."""
    def objective(trial):
        allocations = []
        for ch in channels:
            l = max(low[ch], 1)
            h = max(high[ch], 1)
            allocations.append(trial.suggest_int(f"{ch}", l, h, log=True))
        s = sum(allocations)
        adj = [a / s * total_budget for a in allocations] if s > 0 else allocations
        adj = [min(max(a, max(low[ch], 1)), high[ch]) for a, ch in zip(adj, channels)]

        total = sum(adj)
        while total != total_budget:
            scale = total_budget / total
            adj = [min(max(a * scale, max(low[ch], 1)), high[ch])
                   for a, ch in zip(adj, channels)]
            total = sum(adj)
            if total_budget - total < 1e-6:
                break
            if total > total_budget:
                break

        if total > total_budget:
            raise optuna.TrialPruned()
        if any(a < low[ch] or a > high[ch] for a, ch in zip(adj, channels)):
            raise optuna.TrialPruned()

        store[trial.number] = adj
        resp = predict(predictor, label_encoders, channels, adj, features,
                       weeknum, weekday)
        return float(resp.sum())
    return objective


def optimize_allocation(predictor, label_encoders, channels, features, low, high,
                        total_budget, alloc_time, n_trials, weeknum, weekday, seed):
    low_sum, high_sum = sum(low.values()), sum(high.values())
    if total_budget > high_sum:
        raise SystemExit(
            f"ERROR: total_budget ({total_budget:,.0f}) exceeds the sum of upper "
            f"limits ({high_sum:,.0f}). Raise channel upper limits or lower the budget.")
    if total_budget < low_sum:
        raise SystemExit(
            f"ERROR: total_budget ({total_budget:,.0f}) is below the sum of lower "
            f"limits ({low_sum:,.0f}). Lower channel lower limits or raise the budget.")

    if len(channels) < 2:
        # CMA-ES needs dim>=2; single channel allocation is trivial.
        best = [min(max(total_budget, low[channels[0]]), high[channels[0]])]
    else:
        sampler = optuna.samplers.CmaEsSampler(
            seed=seed, warn_independent_sampling=False)
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                    pruner=optuna.pruners.HyperbandPruner())
        store = {}
        obj = make_objective(predictor, label_encoders, channels, features, low, high,
                             total_budget, weeknum, weekday, store)
        study.optimize(obj, n_trials=n_trials, timeout=alloc_time,
                       n_jobs=1, show_progress_bar=False)
        best = store[study.best_trial.number]

    kpi = predict(predictor, label_encoders, channels, best, features, weeknum, weekday)
    return best, kpi


def main():
    p = argparse.ArgumentParser(description="DMMM channel budget optimizer")
    p.add_argument("--model", "-m", required=True, help="Trained model pickle (train.py)")
    p.add_argument("--output", "-o", default="dmmm_allocation.csv",
                   help="Output CSV for the allocation")
    p.add_argument("--total-budget", type=float, default=None,
                   help="Total budget to allocate. If omitted and --data is given, "
                        "defaults to the sum of historic mean daily spend.")
    p.add_argument("--data", default=None,
                   help="Preprocessed CSV used to auto-derive bounds and baseline")
    p.add_argument("--constraints", default=None,
                   help="CSV with columns Channel, Lower_limit, Upper_limit")
    p.add_argument("--bound-pct", type=float, default=0.2,
                   help="Auto bounds = historic mean +/- this fraction (default 0.2)")
    p.add_argument("--kpi", default="Install", help="KPI column name in --data")
    p.add_argument("--numerical-features", default="Cost")
    p.add_argument("--categorical-features", default="Channel")
    p.add_argument("--alloc-time", type=int, default=15,
                   help="Optuna timeout in seconds (default: 15)")
    p.add_argument("--n-trials", type=int, default=3000)
    p.add_argument("--weeknum", type=int, default=1, help="WeekNum feature value")
    p.add_argument("--weekday", type=int, default=0, help="Weekday feature value")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    numerical = [c.strip() for c in args.numerical_features.split(",") if c.strip()]
    categorical = [c.strip() for c in args.categorical_features.split(",") if c.strip()]
    features = numerical + categorical

    predictor, label_encoders, _ = pd.read_pickle(args.model)
    known_channels = list(label_encoders["Channel"].classes_)

    # Build per-channel bounds + (optional) historic-mean baseline.
    historic_mean = None
    if args.constraints:
        cons = pd.read_csv(args.constraints)
        cons = cons[cons["Channel"].isin(known_channels)]
        channels = cons["Channel"].tolist()
        low = dict(zip(cons["Channel"], cons["Lower_limit"]))
        high = dict(zip(cons["Channel"], cons["Upper_limit"]))
    elif args.data:
        data = pd.read_csv(args.data)
        # historic mean daily spend per channel (notebook's historic_mean)
        hm = (data.pivot_table(index="Date", columns="Channel", values="Cost",
                               aggfunc="sum").mean(axis=0))
        hm = hm[[c for c in hm.index if c in known_channels]]
        historic_mean = hm
        channels = hm.index.tolist()
        low = (hm * (1 - args.bound_pct)).to_dict()
        high = (hm * (1 + args.bound_pct)).to_dict()
    else:
        raise SystemExit("ERROR: provide --constraints or --data to define channel bounds.")

    if not channels:
        raise SystemExit("ERROR: no channels in common between bounds and the model.")

    total_budget = args.total_budget
    if total_budget is None:
        if historic_mean is None:
            raise SystemExit("ERROR: --total-budget is required when using --constraints.")
        total_budget = float(historic_mean.sum())

    best, kpi = optimize_allocation(
        predictor, label_encoders, channels, features, low, high, total_budget,
        args.alloc_time, args.n_trials, args.weeknum, args.weekday, args.seed)

    out = pd.DataFrame({"Channel": channels,
                        "Allocated_Budget": np.round(best, 2),
                        "Estimated_KPI": np.round(kpi, 2)})
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    def alloc_entry(c, b, k):
        entry = {"channel": c,
                 "lower_limit": round(float(low[c]), 2),
                 "upper_limit": round(float(high[c]), 2),
                 "allocated_budget": round(float(b), 2),
                 "estimated_kpi": round(float(k), 2)}
        if historic_mean is not None:
            entry["historic_mean_budget"] = round(float(historic_mean[c]), 2)
        return entry

    summary = {
        "output_file": args.output,
        "kpi": args.kpi,
        "total_budget": round(total_budget, 2),
        "budget_source": "user" if args.total_budget is not None else "historic_mean",
        "total_estimated_kpi": round(float(kpi.sum()), 2),
        "allocation": [alloc_entry(c, b, k) for c, b, k in zip(channels, best, kpi)],
    }
    if args.constraints:
        summary["bounds_source"] = "constraints_csv"
        summary["constraints_file"] = args.constraints
    else:
        summary["bounds_source"] = "historic_mean_pct"
        summary["bound_pct"] = args.bound_pct

    if historic_mean is not None:
        base_costs = historic_mean.values
        base_kpi = predict(predictor, label_encoders, channels, base_costs, features,
                           args.weeknum, args.weekday)
        base_budget = float(historic_mean.sum())
        base_kpi_total = float(base_kpi.sum())
        summary["baseline"] = {
            "description": "historic mean daily spend per channel",
            "total_budget": round(base_budget, 2),
            "total_estimated_kpi": round(base_kpi_total, 2),
            "kpi_improvement_pct": round(
                (float(kpi.sum()) - base_kpi_total) / base_kpi_total * 100, 2)
            if base_kpi_total else None,
            "budget_change_pct": round(
                (total_budget - base_budget) / base_budget * 100, 2)
            if base_budget else None,
        }

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
