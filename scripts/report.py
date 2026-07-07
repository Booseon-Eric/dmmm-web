#!/usr/bin/env python3
"""Phase 4 - assemble a human-readable DMMM report from the phase 1-3 JSON summaries.

preprocess.py / train.py / optimize.py each print a JSON summary to stdout. The
caller saves each of those to a file (e.g. `... > preprocess.json`) and passes them
here; this script merges whatever is available into a single Korean Markdown report
(with tables and a plain-language headline) and prints a short JSON summary on stdout.

Any subset of the three inputs may be given - missing phases are simply skipped.

Dependencies: none (stdlib only).
"""

import argparse
import json
import sys


def load(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 - surface a clean message to the caller
        raise SystemExit(f"ERROR: could not read JSON '{path}': {e}")


def fmt(n, nd=0):
    """Thousands-separated number; '-' for None."""
    if n is None:
        return "-"
    return f"{n:,.{nd}f}"


def section_preprocess(pp, lines):
    lines.append("## 1. 데이터 전처리\n")
    lines.append(f"- 입력 {fmt(pp.get('input_rows'))}행 → 정제 후 {fmt(pp.get('output_rows'))}행 "
                 f"(채널×일자 집계)")
    lines.append(f"- KPI: **{pp.get('kpi')}**")
    rng = pp.get("date_range") or ["-", "-"]
    lines.append(f"- 기간: {rng[0]} ~ {rng[1]}")
    lines.append(f"- 채널 {pp.get('n_channels')}개: {', '.join(pp.get('channels', []))}")

    rpc = pp.get("rows_per_channel") or {}
    if rpc:
        lines.append("\n| 채널 | 행 수 |\n|------|------:|")
        for ch, n in rpc.items():
            lines.append(f"| {ch} | {fmt(n)} |")

    split = pp.get("split")
    if split:
        lines.append(f"\n- 학습/테스트 분리 (기준일 {split.get('train_cutoff')}): "
                     f"train {fmt(split.get('train_rows'))}행 / test {fmt(split.get('test_rows'))}행")

    warnings = pp.get("warnings") or []
    if warnings:
        lines.append("\n**⚠️ 경고**")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("\n- 경고 없음 (NaN/음수/미지의 채널 없음)")
    lines.append("")


def section_train(tr, lines):
    lines.append("## 2. 모델 학습 (XGBoost)\n")
    lines.append(f"- 학습 행 {fmt(tr.get('n_train_rows'))} / 검증 행 {fmt(tr.get('n_valid_rows'))}")
    lines.append(f"- 학습 채널 {len(tr.get('channels', []))}개")
    lines.append(f"- Optuna 하이퍼파라미터 탐색: {fmt(tr.get('optuna_trials_run'))} trial")
    lines.append(f"- 검증 RMSE: **{tr.get('validation_rmse')}** "
                 f"(예측이 실제 {tr.get('kpi')} 값과 평균적으로 벗어나는 정도, 작을수록 좋음)")

    tm = tr.get("test_metrics")
    if tm:
        mape = tm.get("mape")
        bias = tm.get("bias")
        lines.append("\n**보류(test) 데이터 성능 — 일자 합산 기준**")
        if mape is not None:
            lines.append(f"- MAPE {mape * 100:.1f}% → 예측이 실제 {tr.get('kpi')} 수와 "
                         f"**평균 {mape * 100:.1f}% 정도 차이**")
        if bias is not None:
            direction = "과대" if bias > 0 else "과소"
            lines.append(f"- bias {bias * 100:+.1f}% → 전체적으로 실제보다 {abs(bias) * 100:.1f}% "
                         f"{direction}예측하는 경향")
        skipped = tm.get("skipped_unknown_channels") or []
        if skipped:
            lines.append(f"- ⚠️ 학습에 없던 채널이라 평가에서 제외됨: {', '.join(skipped)}")
    lines.append("")


def section_optimize(op, lines):
    lines.append("## 3. 예산 최적화\n")
    base = op.get("baseline")
    if base and base.get("kpi_improvement_pct") is not None:
        lines.append(f"> **핵심: 총예산 {fmt(op.get('total_budget'))}을 아래처럼 재배분하면, "
                     f"예상 {op.get('kpi')}가 약 {base['kpi_improvement_pct']:+.1f}% 늘어납니다.**")
        if base.get("budget_change_pct"):
            lines.append(f">\n> (과거 평균 대비 예산 {base['budget_change_pct']:+.1f}% 변경 기준)")
        lines.append("")

    budget_src = {"user": " (사용자 지정)",
                  "historic_mean": " (과거 평균 일일지출 합계)"}.get(op.get("budget_source"), "")
    lines.append(f"- 총예산: {fmt(op.get('total_budget'))}{budget_src}")

    bounds_src = op.get("bounds_source")
    if bounds_src == "historic_mean_pct" and op.get("bound_pct") is not None:
        lines.append(f"- 채널별 한도: 과거 평균 ±{op['bound_pct'] * 100:.0f}% 자동 설정")
    elif bounds_src == "constraints_csv":
        lines.append("- 채널별 한도: 사용자 지정 (constraints CSV)")

    lines.append(f"- 예상 총 {op.get('kpi')}: **{fmt(op.get('total_estimated_kpi'))}**")
    if base:
        lines.append(f"- (참고) 과거 평균 배분 시 예상 {op.get('kpi')}: "
                     f"{fmt(base.get('total_estimated_kpi'))}")

    alloc = op.get("allocation") or []
    # baseline per-channel KPI isn't in the summary, so we only show recommended here.
    if alloc:
        has_bounds = any(a.get("lower_limit") is not None for a in alloc)
        has_hist = any(a.get("historic_mean_budget") is not None for a in alloc)
        header = ["채널"]
        if has_hist:
            header.append("과거 평균")
        if has_bounds:
            header.append("하한")
        header.append("추천 예산")
        if has_bounds:
            header.append("상한")
        header.append("예상 " + str(op.get("kpi")))
        lines.append("\n| " + " | ".join(header) + " |")
        lines.append("|------|" + "----------:|" * (len(header) - 1))

        at_limit = False
        for a in alloc:
            budget = a.get("allocated_budget")
            budget_cell = fmt(budget)
            lo, hi = a.get("lower_limit"), a.get("upper_limit")
            if budget is not None and hi is not None and budget >= hi:
                budget_cell += " ⬆*"
                at_limit = True
            elif budget is not None and lo is not None and budget <= lo:
                budget_cell += " ⬇*"
                at_limit = True
            row = [str(a.get("channel"))]
            if has_hist:
                row.append(fmt(a.get("historic_mean_budget")))
            if has_bounds:
                row.append(fmt(lo))
            row.append(budget_cell)
            if has_bounds:
                row.append(fmt(hi))
            row.append(fmt(a.get("estimated_kpi")))
            lines.append("| " + " | ".join(row) + " |")

        if at_limit:
            lines.append("\n\\* 채널 한도(상한 ⬆ / 하한 ⬇)에 도달한 배분 — "
                         "한도를 조정하면 배분이 달라질 수 있음")
    lines.append("")


def main():
    p = argparse.ArgumentParser(description="DMMM phase-4 report builder (merges phase 1-3 JSON)")
    p.add_argument("--preprocess-json", default=None, help="preprocess.py stdout JSON")
    p.add_argument("--train-json", default=None, help="train.py stdout JSON")
    p.add_argument("--optimize-json", default=None, help="optimize.py stdout JSON")
    p.add_argument("--output", "-o", default="dmmm_report.md", help="Markdown report path")
    p.add_argument("--title", default="DMMM 마케팅 믹스 분석 리포트", help="Report title")
    args = p.parse_args()

    pp = load(args.preprocess_json)
    tr = load(args.train_json)
    op = load(args.optimize_json)

    if not any((pp, tr, op)):
        raise SystemExit("ERROR: provide at least one of "
                         "--preprocess-json / --train-json / --optimize-json")

    lines = [f"# {args.title}\n"]
    if pp:
        section_preprocess(pp, lines)
    if tr:
        section_train(tr, lines)
    if op:
        section_optimize(op, lines)

    report = "\n".join(lines).rstrip() + "\n"
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    summary = {
        "report_file": args.output,
        "phases_included": [name for name, v in
                            (("preprocess", pp), ("train", tr), ("optimize", op)) if v],
        "kpi": (op or tr or pp or {}).get("kpi"),
    }
    if op and op.get("baseline"):
        summary["kpi_improvement_pct"] = op["baseline"].get("kpi_improvement_pct")
        summary["total_estimated_kpi"] = op.get("total_estimated_kpi")
    if tr and tr.get("test_metrics"):
        summary["test_mape"] = tr["test_metrics"].get("mape")

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
