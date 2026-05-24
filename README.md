# AI Bubble Index

신영증권 김효진 박사의 4개 공식(주도주 압착 · 채권 자경단 · 사모 크레딧 · IPO 포화)을 기반으로
AI 자산 가격의 거품 정점·붕괴를 일별로 모니터링하는 0~100 지수.

## 구조

```
ai_bubble_index/
├── data/raw/             # 수집된 원본 (parquet)
├── data/processed/       # 정규화된 metric (parquet)
├── data/snapshots/       # 일자별 점수 (CSV)
├── src/
│   ├── common.py
│   ├── collectors/       # FRED·Yahoo·NY Fed·Ritter
│   ├── transforms/       # (Phase 2) 백분위·기둥 집계·점수
│   ├── validate/         # (Phase 3) 백테스트·민감도
│   ├── dashboard/        # (Phase 4) Streamlit
│   └── run_collect.py    # Phase 1 진입점
├── config/weights.yaml   # 이론 기반 가중치
├── requirements.txt
└── AI_Bubble_Index_Methodology.docx
```

## 가중치 (이론 기반)

| 기둥 | 가중 | 시간성 |
|---|---|---|
| 사모 크레딧 | 35% | 결정적 후행 |
| 채권 자경단 | 30% | 동행 |
| 주도주 압착 | 20% | 선행 |
| IPO 포화 | 15% | 동행~후행 |

## 설치

```bash
pip install -r requirements.txt
```

## Phase 1: 데이터 수집

```bash
# 전체 수집
python -m src.run_collect

# 일부만
python -m src.run_collect fred yahoo
```

수집 결과는 `data/raw/*.parquet` 에 저장된다.

## 데이터 소스

| Collector | 소스 | 항목 |
|---|---|---|
| `fred.py` | FRED CSV/API | DGS30, DGS10, DFII10, HY OAS, CCC OAS |
| `yahoo.py` | yfinance | ^GSPC, RSP, ^MOVE, BIZD, BKLN, IPO, SPY |
| `nyfed.py` | NY Fed 공개 | ACM Term Premium |
| `ritter.py` | UFL (수동) | IPO 첫날 수익률 |

### 수동 다운로드가 필요한 경우

- **Ritter IPO**: https://site.warrington.ufl.edu/ritter/ipo-data/ 에서 다운로드 후 `data/raw/ritter_ipo.xlsx` 에 저장
- **NY Fed ACM**: URL이 막혔다면 newyorkfed.org 에서 직접 다운로드 후 `data/raw/acm_termpremium.xlsx` 에 저장

### FRED API 키 (선택)

CSV 엔드포인트로도 동작하지만, 안정성을 위해 무료 API 키 사용 권장:
```bash
export FRED_API_KEY=your_key_here
```
발급: https://fred.stlouisfed.org/docs/api/api_key.html

## Roadmap

- [x] Phase 1 — 데이터 수집 모듈
- [ ] Phase 2 — 변환·점수 산출 파이프라인
- [ ] Phase 3 — 백테스트 (2000·2007·2021) + 민감도
- [ ] Phase 4 — Streamlit 대시보드
- [ ] Phase 5 — 자동화 (cron/GitHub Actions)

## 참고

방법론 상세는 `AI_Bubble_Index_Methodology.docx` 참조.
