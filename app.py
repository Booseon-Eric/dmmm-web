#!/usr/bin/env python3
"""DMMM 마케팅 예산 최적화 — Streamlit 웹앱.

업로드한 마케팅 CSV를 4단계 스킬 파이프라인(전처리 → 학습 → 최적화 → 리포트)에
그대로 통과시키고, 결과 리포트와 다운로드를 제공한다.
각 단계는 scripts/ 의 CLI 스크립트를 subprocess로 호출한다 (스킬 로직 무수정 보존).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPTS = Path(__file__).parent / "scripts"
SAMPLE = Path(__file__).parent / "sample_data.csv"

st.set_page_config(page_title="DMMM 예산 최적화", page_icon="📊", layout="wide")

st.title("📊 DMMM 마케팅 예산 최적화")
st.caption(
    "마케팅 집행 데이터(CSV)를 올리면 XGBoost 반응 모델을 학습하고, "
    "채널별 예산을 재배분해 예상 KPI를 극대화하는 배분안과 리포트를 만들어 드립니다."
)

# ---------------------------------------------------------------- data format guide
with st.expander("📋 어떤 데이터를 넣어야 하나요? (CSV 양식 안내)", expanded=True):
    st.markdown(
        "아래 4개 컬럼이 있는 CSV가 필요합니다. **컬럼 이름이 달라도** 괜찮아요 — "
        "사이드바에서 실제 이름을 지정하면 됩니다. 하루에 채널별로 여러 캠페인 행이 "
        "있어도 자동으로 채널×일자 단위로 합쳐집니다."
    )

    fmt = pd.DataFrame({
        "역할": ["날짜", "채널(매체)", "비용", "KPI(목표 지표)"],
        "기본 컬럼명": ["Date_", "Media", "Cost_", "Install(Total)"],
        "설명": [
            "집행 일자 (YYYY-MM-DD)",
            "광고 매체/채널 이름 (예: naver, google, meta)",
            "그날 그 채널에 쓴 비용",
            "늘리고 싶은 성과 지표 (설치·구매·클릭 등)",
        ],
        "예시 값": ["2025-01-01", "naver", "150000", "42"],
    })
    st.table(fmt)

    st.markdown("**입력 예시** (이렇게 생긴 CSV):")
    example = pd.DataFrame({
        "Date_": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
        "Media": ["naver", "google", "meta", "naver", "google"],
        "Cost_": [150000, 230000, 180000, 145000, 240000],
        "Install(Total)": [42, 55, 47, 40, 58],
    })
    st.dataframe(example, use_container_width=True, hide_index=True)

    st.caption(
        "· KPI는 하나만 선택하면 됩니다 (설치·매출·가입 등 무엇이든 가능). \n"
        "· 정확한 분석을 위해 채널별로 최소 몇 주 이상의 데이터가 있으면 좋습니다. \n"
        "· 형식이 헷갈리면 아래 **‘샘플 데이터로 체험해보기’**를 켜서 실제 결과를 먼저 보세요."
    )

    if SAMPLE.exists():
        st.download_button(
            "⬇️ 샘플 CSV 내려받아 형식 확인하기",
            SAMPLE.read_bytes(), "sample_data.csv", "text/csv",
        )

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ 설정")

    st.subheader("데이터 컬럼")
    date_col = st.text_input("날짜 컬럼", "Date_")
    cost_col = st.text_input("비용 컬럼", "Cost_")
    kpi_col = st.text_input("KPI 컬럼", "Install(Total)")
    channel_col = st.text_input("채널 컬럼", "Media")

    st.subheader("최적화")
    budget_mode = st.radio("총예산", ["과거 평균 지출 합계 (자동)", "직접 입력"])
    total_budget = None
    if budget_mode == "직접 입력":
        total_budget = st.number_input("총예산 (일일)", min_value=1.0, value=1_000_000.0, step=10_000.0)
    bound_pct = st.slider("채널별 한도 (과거 평균 ±%)", 5, 50, 20) / 100

    st.subheader("검증 (선택)")
    use_cutoff = st.checkbox("보류(test) 기간으로 정확도 검증", value=True)
    cutoff = None
    if use_cutoff:
        cutoff = st.text_input("학습/테스트 분리 기준일 (YYYY-MM-DD)", "",
                               help="비워두면 데이터 마지막 2주를 자동으로 테스트로 씁니다.")

    with st.expander("고급"):
        train_time = st.number_input("모델 학습 시간(초)", 5, 120, 30)
        alloc_time = st.number_input("최적화 시간(초)", 5, 60, 15)
        seed = st.number_input("랜덤 시드", 0, 9999, 1)

# ---------------------------------------------------------------- input
up = st.file_uploader("마케팅 데이터 CSV 업로드", type="csv")
use_sample = st.toggle("샘플 데이터로 체험해보기", value=False, disabled=up is not None)

raw_bytes = None
if up is not None:
    raw_bytes = up.getvalue()
elif use_sample and SAMPLE.exists():
    raw_bytes = SAMPLE.read_bytes()
    st.info("샘플 데이터(가상의 4개 채널, 4개월치)를 사용합니다.")

if raw_bytes:
    try:
        preview = pd.read_csv(pd.io.common.BytesIO(raw_bytes), nrows=5)
        st.write("**미리보기** (첫 5행)")
        st.dataframe(preview, use_container_width=True)
        missing = [c for c in (date_col, cost_col, kpi_col, channel_col)
                   if c not in preview.columns]
        if missing:
            st.error(f"CSV에 없는 컬럼: {missing} — 사이드바에서 컬럼명을 맞춰주세요. "
                     f"실제 컬럼: {list(preview.columns)}")
            st.stop()
    except Exception as e:
        st.error(f"CSV를 읽을 수 없습니다: {e}")
        st.stop()

# ---------------------------------------------------------------- pipeline
def run(step_name, args, cwd):
    """Run one skill script; return parsed JSON summary from stdout."""
    proc = subprocess.run([sys.executable, str(SCRIPTS / args[0]), *args[1:]],
                          cwd=cwd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        err = proc.stdout.strip() or proc.stderr.strip()
        raise RuntimeError(f"[{step_name}] {err.splitlines()[-1] if err else '알 수 없는 오류'}")
    return json.loads(proc.stdout)


if raw_bytes and st.button("🚀 분석 시작", type="primary", use_container_width=True):
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        (wd / "input.csv").write_bytes(raw_bytes)

        # auto cutoff = last 14 days if requested but blank
        auto_cutoff = None
        if use_cutoff and not cutoff:
            dates = pd.to_datetime(
                pd.read_csv(wd / "input.csv", usecols=[date_col])[date_col])
            auto_cutoff = (dates.max() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        eff_cutoff = cutoff or auto_cutoff

        try:
            with st.status("파이프라인 실행 중...", expanded=True) as status:
                st.write("1️⃣ 데이터 전처리...")
                pre_args = ["preprocess.py", "-i", "input.csv", "-o", "pre.csv",
                            "--kpi", kpi_col, "--cost-col", cost_col,
                            "--date-col", date_col, "--channel-source", channel_col]
                if eff_cutoff:
                    pre_args += ["--train-cutoff", eff_cutoff]
                pp = run("전처리", pre_args, wd)
                (wd / "preprocess.json").write_text(json.dumps(pp, ensure_ascii=False))
                for w in pp.get("warnings", []):
                    st.warning(w)

                st.write(f"2️⃣ XGBoost 모델 학습 (최대 {train_time}초)...")
                train_input = "pre_train.csv" if eff_cutoff else "pre.csv"
                tr_args = ["train.py", "-i", train_input, "-o", "model.pkl",
                           "--kpi", pp["kpi"], "--train-time", str(train_time),
                           "--seed", str(seed)]
                if eff_cutoff:
                    tr_args += ["--test-input", "pre_test.csv"]
                tr = run("학습", tr_args, wd)
                (wd / "train.json").write_text(json.dumps(tr, ensure_ascii=False))

                st.write(f"3️⃣ 예산 배분 최적화 (최대 {alloc_time}초)...")
                op_args = ["optimize.py", "-m", "model.pkl", "--data", "pre.csv",
                           "-o", "alloc.csv", "--kpi", pp["kpi"],
                           "--bound-pct", str(bound_pct),
                           "--alloc-time", str(alloc_time), "--seed", str(seed)]
                if total_budget:
                    op_args += ["--total-budget", str(total_budget)]
                op = run("최적화", op_args, wd)
                (wd / "optimize.json").write_text(json.dumps(op, ensure_ascii=False))

                st.write("4️⃣ 리포트 생성...")
                run("리포트", ["report.py",
                               "--preprocess-json", "preprocess.json",
                               "--train-json", "train.json",
                               "--optimize-json", "optimize.json",
                               "-o", "report.md"], wd)
                status.update(label="✅ 분석 완료", state="complete")

            # ---------------------------------------------------- results
            base = op.get("baseline") or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("예상 KPI 개선", f"{base.get('kpi_improvement_pct', 0):+.1f}%")
            c2.metric(f"예상 총 {op['kpi']}", f"{op['total_estimated_kpi']:,.0f}")
            c3.metric("총예산", f"{op['total_budget']:,.0f}")

            alloc = pd.DataFrame(op["allocation"])
            chart = alloc.set_index("channel")[
                [c for c in ("historic_mean_budget", "allocated_budget") if c in alloc]]
            chart.columns = [{"historic_mean_budget": "과거 평균",
                              "allocated_budget": "추천 예산"}[c] for c in chart.columns]
            st.subheader("채널별 예산 배분")
            st.bar_chart(chart)

            st.subheader("📄 상세 리포트")
            report_md = (wd / "report.md").read_text(encoding="utf-8")
            st.markdown(report_md)

            st.divider()
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ 리포트 (Markdown)", report_md,
                               "dmmm_report.md", "text/markdown", use_container_width=True)
            d2.download_button("⬇️ 배분안 (CSV)", (wd / "alloc.csv").read_bytes(),
                               "dmmm_allocation.csv", "text/csv", use_container_width=True)
            d3.download_button("⬇️ 학습된 모델 (pkl)", (wd / "model.pkl").read_bytes(),
                               "dmmm_model.pkl", use_container_width=True)

        except RuntimeError as e:
            st.error(str(e))
        except subprocess.TimeoutExpired:
            st.error("실행 시간이 초과됐습니다. 데이터를 줄이거나 학습/최적화 시간을 낮춰보세요.")
