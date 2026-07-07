# DMMM 마케팅 예산 최적화 웹앱

마케팅 집행 CSV를 업로드하면 4단계 파이프라인이 자동으로 실행됩니다:

1. **전처리** (`scripts/preprocess.py`) — 채널×일자 집계, 날짜 피처 생성, 검증
2. **학습** (`scripts/train.py`) — XGBoost 반응 모델 + Optuna 하이퍼파라미터 탐색
3. **최적화** (`scripts/optimize.py`) — CMA-ES로 채널별 예산 재배분 (예상 KPI 최대화)
4. **리포트** (`scripts/report.py`) — 한국어 마크다운 리포트 생성

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포 (무료)

1. 이 폴더를 GitHub 저장소로 push
2. https://share.streamlit.io 접속 → GitHub 로그인
3. "New app" → 저장소 선택 → Main file path: `app.py` → Deploy
4. 몇 분 뒤 `https://<앱이름>.streamlit.app` URL 생성

## 입력 CSV 형식

기본 컬럼명 (사이드바에서 변경 가능):

| 컬럼 | 기본값 | 설명 |
|------|--------|------|
| 날짜 | `Date_` | YYYY-MM-DD |
| 비용 | `Cost_` | 일별 집행 비용 |
| KPI | `Install(Total)` | 최적화 목표 지표 |
| 채널 | `Media` | 매체/채널명 |

캠페인 단위 행이어도 됩니다 — 채널×일자로 자동 집계합니다.
`sample_data.csv` 로 형식을 확인하거나 앱에서 "샘플 데이터로 체험"을 켜보세요.

## 참고

- `scripts/setup_env.py` 는 로컬 스킬 실행용(Phase 0)입니다. 웹 배포에서는
  `requirements.txt` 가 그 역할을 대신하므로 앱에서 호출하지 않습니다.
- 무료 티어는 CPU가 약하므로 학습/최적화 시간을 기본값(30초/15초) 근처로 유지하세요.
