"""AI Bubble Index — Streamlit Dashboard (Phase 4 v2).

Run from project root:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.common import DATA_PROCESSED
from src.transforms.score import regime_label
from src.validate.backtest import CRASH_EVENTS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Bubble Index",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
REGIME_HEX   = {"green": "#27ae60", "yellow": "#f39c12", "orange": "#e67e22", "red": "#e74c3c"}
REGIME_BG    = {"green": "rgba(39,174,96,.10)", "yellow": "rgba(243,156,18,.12)",
                "orange": "rgba(230,126,34,.13)", "red": "rgba(231,76,60,.13)"}
REGIME_LIGHT = {"green": "#eafaf1", "yellow": "#fef9e7", "orange": "#fef5ec", "red": "#fdedec"}
REGIME_KO    = {"green": "안전 🟢", "yellow": "주의 🟡", "orange": "경계 🟠", "red": "위험 🔴"}

PILLAR_KO = {
    "concentration":   "주도주 압착",
    "bond_vigilantes": "채권 자경단",
    "private_credit":  "사모 크레딧",
    "ipo_saturation":  "IPO 포화",
}
PILLAR_COLOR = {
    "concentration":   "#9b59b6",
    "bond_vigilantes": "#2980b9",
    "private_credit":  "#16a085",
    "ipo_saturation":  "#e74c3c",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] { background: #f0f2f6; }
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] td,
    [data-testid="stAppViewContainer"] th { color: #1a1f2e; }
    /* caption / help text — 너무 흐리지 않게 */
    [data-testid="stAppViewContainer"] small,
    [data-testid="stCaptionContainer"] { color: #4a5568 !important; }

    [data-testid="stSidebar"]          { background: #1a1f2e; }
    [data-testid="stSidebar"] * { color: #e8ecf3 !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #2d3561; color: #e8ecf3; border: none;
        border-radius: 8px; width: 100%; margin-top: 4px;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: #3d4a7a; }

    /* ── Hide default decorations ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

    /* ── Cards ── */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,.07);
        border-top: 4px solid var(--accent);
        height: 100%;
    }
    .kpi-label  { font-size: 11px; font-weight: 600; color: #8492a6;
                  letter-spacing: .6px; text-transform: uppercase; margin-bottom: 4px; }
    .kpi-value  { font-size: 28px; font-weight: 800; color: #1a1f2e; line-height: 1; }
    .kpi-sub    { font-size: 12px; color: #8492a6; margin-top: 4px; }

    /* ── Regime banner ── */
    .regime-banner {
        border-radius: 10px; padding: 10px 20px;
        font-size: 14px; font-weight: 600;
        display: flex; align-items: center; gap: 10px;
        margin: 8px 0 16px;
    }

    /* ── Section headers ── */
    .sec-header {
        font-size: 15px; font-weight: 700; color: #2c3e50;
        padding: 0 0 8px; margin: 4px 0 12px;
        border-bottom: 2px solid #e8ecf3;
        display: flex; align-items: center; gap: 8px;
    }

    /* ── Insight cards ── */
    .insight-card {
        background: #fff;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,.06);
        border-left: 5px solid var(--accent);
        margin-bottom: 10px;
    }
    .insight-title { font-size: 13px; font-weight: 700; color: #1a1f2e; margin-bottom: 6px; }
    .insight-body  { font-size: 12.5px; color: #4a5568; line-height: 1.65; }

    /* ── Narrative analysis ── */
    .narrative-wrap {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px 26px 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,.07);
        margin: 0 0 20px;
        font-size: 13.5px;
        color: #2c3e50;
        line-height: 1.78;
    }
    .narrative-title {
        font-size: 15px; font-weight: 800; color: #1a1f2e;
        margin-bottom: 14px; padding-bottom: 10px;
        border-bottom: 2px solid #f0f2f6;
        display: flex; align-items: center; gap: 8px;
    }
    .narrative-para { margin-bottom: 12px; }
    .narrative-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 10px; margin: 14px 0;
    }
    .narrative-pillar-box {
        background: #f8f9fb;
        border-radius: 10px;
        padding: 12px 14px;
        border-left: 4px solid var(--pc);
    }
    .npb-title { font-size: 11.5px; font-weight: 700; color: var(--pc);
                 margin-bottom: 5px; }
    .npb-body  { font-size: 12px; color: #4a5568; line-height: 1.6; }
    .conclusion-box {
        background: linear-gradient(135deg, #fef9f0, #fef5ec);
        border: 1.5px solid #e67e2260;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 14px;
        font-size: 13px;
        line-height: 1.7;
    }
    .hl-r  { color: #e74c3c; font-weight: 700; }
    .hl-o  { color: #e67e22; font-weight: 700; }
    .hl-g  { color: #27ae60; font-weight: 700; }
    .hl-b  { color: #2980b9; font-weight: 700; }
    .hl-bk { font-weight: 700; color: #1a1f2e; }

    /* ── Tab styling ── */
    [data-testid="stTabs"] [role="tab"] {
        font-size: 13.5px; font-weight: 600; padding: 6px 18px;
        border-radius: 8px 8px 0 0;
        color: #4a5568 !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: #1a202c !important;
        background: rgba(230,126,34,0.08);
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #e67e22 !important;
        font-weight: 700 !important;
    }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scores():  return pd.read_parquet(DATA_PROCESSED / "scores.parquet")

@st.cache_data(ttl=300)
def load_norm():    return pd.read_parquet(DATA_PROCESSED / "normalized.parquet")

@st.cache_data(ttl=300)
def load_bt():
    p = DATA_PROCESSED / "backtest.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300)
def load_sv():
    p = DATA_PROCESSED / "sensitivity.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=300)
def load_metrics():
    p = DATA_PROCESSED / "metrics.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

# ── Narrative analysis builder ───────────────────────────────────────────────
def build_narrative_html(scores_df: pd.DataFrame, metrics_df: pd.DataFrame,
                          latest_score: pd.Series, bt_df: pd.DataFrame) -> str:
    bi    = float(latest_score["bubble_index"])
    reg   = regime_label(bi)
    r_col = REGIME_HEX[reg]

    # ── 전체 역사 대비 현재 백분위 ──────────────────────────────────────────
    hist       = scores_df["bubble_index"].dropna()
    hist_pct   = (hist <= bi).mean() * 100          # 하위 N% → 상위는 100-N
    top_pct    = 100 - hist_pct

    # ── 30일 추세 ──────────────────────────────────────────────────────────
    trend      = bi - float(hist.iloc[-31]) if len(hist) > 31 else 0.0
    trend_str  = f"+{trend:.1f}" if trend > 0.3 else (f"{trend:.1f}" if trend < -0.3 else "±0")
    trend_word = "상승" if trend > 0.3 else ("하락" if trend < -0.3 else "보합")
    trend_cls  = "hl-r" if trend > 2 else ("hl-o" if trend > 0.3 else ("hl-g" if trend < -0.3 else "hl-bk"))

    # ── 과거 폭락 피크 참조 ────────────────────────────────────────────────
    gfc_peak, pc_peak = 78.9, 70.1
    if not bt_df.empty:
        gfc_row = bt_df[bt_df["event"].str.contains("gfc", case=False, na=False)]
        pc_row  = bt_df[bt_df["event"].str.contains("covid", case=False, na=False)]
        if not gfc_row.empty: gfc_peak = float(gfc_row["index_peak"].iloc[0])
        if not pc_row.empty:  pc_peak  = float(pc_row["index_peak"].iloc[0])

    dist70 = 70.0 - bi

    # ── 기둥 값 ────────────────────────────────────────────────────────────
    bv  = float(latest_score.get("bond_vigilantes", np.nan))
    con = float(latest_score.get("concentration",   np.nan))
    pc  = float(latest_score.get("private_credit",  np.nan))
    ipo = float(latest_score.get("ipo_saturation",  np.nan))

    def top(col):
        s = scores_df[col].dropna()
        v = latest_score.get(col, np.nan)
        if pd.isna(v) or s.empty: return "—"
        return f"상위 {100 - (s <= v).mean() * 100:.0f}%"

    # ── 최신 원시 지표 값 ──────────────────────────────────────────────────
    def mv(col):
        if metrics_df.empty or col not in metrics_df.columns: return np.nan
        return float(metrics_df[col].dropna().iloc[-1])

    y30  = mv("us_30y_yield")
    y10  = mv("us_10y_yield")
    real = mv("us_10y_real")
    hy   = mv("hy_oas")
    ccc  = mv("ccc_spread")

    def fmt(v, unit="%"):
        return f"{v:.2f}{unit}" if not np.isnan(v) else "N/A"

    # ── 기둥별 해석 텍스트 ─────────────────────────────────────────────────
    def _bv_body():
        level = "역사적 고점 수준에 근접" if bv >= 75 else ("경계 수준" if bv >= 60 else "보통 수준")
        return (f"30Y금리 <b>{fmt(y30)}</b> · 10Y <b>{fmt(y10)}</b> · 실질금리 <b>{fmt(real)}</b>. "
                f"거시 금리 지표 전반이 {level}이며, 자산 가격 할인율 상승 압력이 지속되고 있습니다. "
                f"단독으로 폭락을 유발하지는 않으나 사모 크레딧과 동반 악화 시 위험이 현실화됩니다.")

    def _con_body():
        lvl = "심화" if con >= 70 else ("진행 중" if con >= 55 else "미미")
        return (f"S&P 500이 등가중 지수(RSP) 대비 12개월 누적 <b>강한 아웃퍼폼</b> 지속 중. "
                f"AI 테마 중심 소수 대형주로의 자금 집중이 {lvl}되어, "
                f"거품의 초기 특징인 주도주 압착 패턴을 형성하고 있습니다.")

    def _pc_body():
        hy_s  = "타이트" if not np.isnan(hy) and hy < 4.0 else "확대"
        color = "안전" if pc < 50 else ("주의" if pc < 65 else "위험")
        return (f"HY OAS <b>{fmt(hy,'%p')}</b> · CCC <b>{fmt(ccc,'%p')}</b>. "
                f"신용 스프레드가 {hy_s}하게 유지되며 현재 {color} 구간. "
                f"역사적으로 이 기둥이 60점을 초과할 때 폭락이 본격화됩니다. "
                f"현재 가장 중요한 <b>핵심 모니터링 지표</b>입니다.")

    def _ipo_body():
        lvl = "재점화 조짐" if ipo >= 65 else ("중립" if ipo >= 45 else "냉각")
        return (f"IPO ETF(르네상스)의 SPY 대비 상대 강도가 최근 반등하며 투기적 상장 열기 {lvl}. "
                f"2021년 SPAC 붐(역대 최고점) 수준에는 미치지 못하나, "
                f"채권 자경단·주도주 압착과 동반 상승 시 복합 경보로 격상될 수 있습니다.")

    # ── 종합 전망 텍스트 ──────────────────────────────────────────────────
    if bi < 55:
        outlook = (f"4개 기둥이 혼조세를 보이며 전반적으로 <b>중립~주의 구간</b>에 해당합니다. "
                   f"단기 폭락 위험은 제한적이나, 채권 자경단 지표를 중심으로 추이를 관찰하세요.")
    elif bi < 70:
        pc_safe  = pc < 55
        dist_str = f"{dist70:.1f}점"
        outlook = (f"채권 자경단과 주도주 압착이 경계 수준이나, "
                   f"폭락의 최종 방아쇠인 <b>사모 크레딧이 {'아직 중립(안전판 역할 중)' if pc_safe else '경계 수준에 진입'}</b>합니다. "
                   f"경보 임계값(70점)까지 <b>{dist_str} 여유</b>가 있으며, "
                   f"과거 두 번의 폭락 사례에서 지수가 70을 돌파한 이후에야 리스크가 현실화됐습니다. "
                   f"현재는 <b>포지션 유지하되 신용 스프레드 일별 모니터링</b>이 권장됩니다.")
    else:
        outlook = (f"지수가 경보 임계값(70점)을 <b>초과</b>했습니다. "
                   f"GFC(78.9, 441일 선행)·Post-COVID(70.1, 48일 선행) 패턴과 유사한 구간에 진입했습니다. "
                   f"사모 크레딧 기둥의 동향을 <b>최우선 리스크 지표</b>로 추적하며, "
                   f"적극적인 리스크 헤지를 검토할 시점입니다.")

    # ── HTML 조합 ─────────────────────────────────────────────────────────
    return f"""
<div class="narrative-wrap">
  <div class="narrative-title">🔎 현황 분석 및 전망 — {latest_score.name.date()}</div>

  <p class="narrative-para">
    현재 AI Bubble Index는 <span class="hl-{reg[0]}">{bi:.1f}점 ({reg.upper()})</span>으로,
    2004년 이후 전체 관측 기간 대비 <span class="hl-bk">{top_pct:.0f}% 수준(상위 {top_pct:.0f}%)</span>에 해당합니다.
    최근 30일간 <span class="{trend_cls}">{trend_str}점 {trend_word}</span> 추세이며,
    과거 대형 폭락 직전 피크인
    <span class="hl-r">GFC {gfc_peak:.1f}점</span>·<span class="hl-o">Post-COVID {pc_peak:.1f}점</span> 대비
    각각 <span class="hl-bk">{gfc_peak - bi:.1f}점, {pc_peak - bi:.1f}점</span> 하회 중입니다.
    경보 임계값(70점)까지는 <span class="hl-o">{dist70:.1f}점</span>의 여유가 있습니다.
  </p>

  <div class="narrative-grid">
    <div class="narrative-pillar-box" style="--pc:{PILLAR_COLOR['bond_vigilantes']}">
      <div class="npb-title">채권 자경단 &nbsp;{bv:.1f}점 &nbsp;({top('bond_vigilantes')})</div>
      <div class="npb-body">{_bv_body()}</div>
    </div>
    <div class="narrative-pillar-box" style="--pc:{PILLAR_COLOR['concentration']}">
      <div class="npb-title">주도주 압착 &nbsp;{con:.1f}점 &nbsp;({top('concentration')})</div>
      <div class="npb-body">{_con_body()}</div>
    </div>
    <div class="narrative-pillar-box" style="--pc:{PILLAR_COLOR['private_credit']}">
      <div class="npb-title">사모 크레딧 &nbsp;{pc:.1f}점 &nbsp;({top('private_credit')})</div>
      <div class="npb-body">{_pc_body()}</div>
    </div>
    <div class="narrative-pillar-box" style="--pc:{PILLAR_COLOR['ipo_saturation']}">
      <div class="npb-title">IPO 포화 &nbsp;{ipo:.1f}점 &nbsp;({top('ipo_saturation')})</div>
      <div class="npb-body">{_ipo_body()}</div>
    </div>
  </div>

  <div class="conclusion-box">
    📌 <b>종합 전망 &nbsp;|&nbsp; 폭락 연계성 평가 :</b>&nbsp; {outlook}
  </div>
</div>
"""

# ── Validation analysis (computed once, cached) ───────────────────────────────
@st.cache_data(ttl=3600)
def compute_validation():
    from src.common import load_raw
    scores = load_scores()
    yahoo  = load_raw("yahoo")
    bi  = scores["bubble_index"].dropna()
    spx = yahoo["^GSPC"].reindex(bi.index).ffill().dropna()
    common = bi.index.intersection(spx.index)
    bi, spx = bi[common], spx[common]

    horizons = [21, 63, 126, 252]
    h_labels = ["1개월", "3개월", "6개월", "12개월"]

    rows = []
    bins   = [0, 30, 55, 75, 100]
    labels = ["🟢 Green (0-30)", "🟡 Yellow (30-55)", "🟠 Orange (55-75)", "🔴 Red (75-100)"]
    regime_bin = pd.cut(bi, bins=bins, labels=labels)

    for h, hl in zip(horizons, h_labels):
        fwd = spx.pct_change(h).shift(-h) * 100
        df = pd.DataFrame({"bi": bi, "fwd": fwd, "regime": regime_bin}).dropna()
        for r in labels:
            sub = df[df["regime"] == r]["fwd"]
            rows.append({
                "레짐": r, "기간": hl,
                "평균 수익률(%)": round(sub.mean(), 1) if len(sub) else None,
                "중앙값(%)":      round(sub.median(), 1) if len(sub) else None,
                "하락 확률(%)":   round((sub < 0).mean() * 100, 0) if len(sub) else None,
                "표본(일)":       len(sub),
            })
    fwd_table = pd.DataFrame(rows)

    # Drawdown after threshold cross
    spx_roll_max = spx.cummax()
    spx_dd = (spx - spx_roll_max) / spx_roll_max * 100

    return bi, spx, spx_dd, fwd_table

# ── Chart builders ────────────────────────────────────────────────────────────
def _band(fig, y0, y1, color, row=1):
    fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below", row=row, col=1)


def build_main_chart(scores_df, start):
    sub = scores_df[scores_df.index >= start].dropna(subset=["bubble_index"])
    fig = go.Figure()

    for y0, y1, r in [(0,30,"green"),(30,55,"yellow"),(55,75,"orange"),(75,100,"red")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=REGIME_BG[r], line_width=0, layer="below")

    for label, _, crash_top, _ in CRASH_EVENTS:
        ct = pd.Timestamp(crash_top)
        if ct >= start:
            fig.add_vline(x=ct.timestamp()*1000, line_dash="dot",
                          line_color="rgba(100,100,100,.4)",
                          annotation_text=label, annotation_font_size=9,
                          annotation_position="top")

    pillar_palette = list(PILLAR_COLOR.values())
    for i, (p, label) in enumerate(PILLAR_KO.items()):
        if p in sub.columns:
            fig.add_trace(go.Scatter(
                x=sub.index, y=sub[p], name=label,
                line=dict(color=pillar_palette[i], width=1, dash="dot"),
                opacity=.55, visible="legendonly",
            ))

    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["bubble_index"], name="종합 지수",
        line=dict(color="#1a1f2e", width=2.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>종합: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=400, margin=dict(l=40, r=20, t=20, b=30),
        yaxis=dict(range=[0,100], title="", ticksuffix=" ", gridcolor="#f0f2f6",
                   tickfont=dict(color="#2c3e50")),
        xaxis=dict(title="", showgrid=False, tickfont=dict(color="#2c3e50")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11, color="#2c3e50")),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#2c3e50", family="system-ui, -apple-system, sans-serif"),
    )
    return fig


def build_pillar_chart(latest):
    items = [(p, PILLAR_KO[p]) for p in PILLAR_KO if p in latest and not pd.isna(latest[p])]
    labels = [l for _, l in items]
    values = [latest[p] for p, _ in items]
    bi_val = latest.get("bubble_index", np.nan)

    colors = [REGIME_HEX[regime_label(v)] for v in values]
    if not pd.isna(bi_val):
        labels.append("● 종합 지수")
        values.append(bi_val)
        colors.append("#1a1f2e")

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"<b>{v:.1f}</b>" for v in values],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        height=260, margin=dict(l=8, r=20, t=8, b=8),
        xaxis=dict(range=[0,100], showgrid=True, gridcolor="#f0f2f6", zeroline=False,
                   tickfont=dict(color="#2c3e50")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#2c3e50", size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        font=dict(color="#2c3e50", family="system-ui, -apple-system, sans-serif"),
    )
    return fig


def build_heatmap(norm_df, n=60):
    sub = norm_df.dropna(how="all").tail(n)
    if sub.empty: return go.Figure()
    METRIC_KO = {
        "spx_vs_rsp_12m":   "SPX vs 등가중 스프레드",
        "us_30y_yield":     "미국 30Y 금리",
        "us_10y_yield":     "미국 10Y 금리",
        "us_10y_real":      "실질금리 (TIPS)",
        "term_premium_10y": "텀 프리미엄 (ACM)",
        "move_index":       "MOVE 지수",
        "hy_oas":           "HY 스프레드",
        "ccc_spread":       "CCC 스프레드",
        "bdc_etf":          "BDC ETF (BIZD)",
        "leveraged_loan":   "레버리지론 (BKLN)",
        "ipo_etf":          "IPO ETF",
        "ipo_etf_relative": "IPO vs SPY 상대강도",
    }
    cols = [c for c in sub.columns if c in METRIC_KO]
    colorscale = [
        [0.00, "#e8f8f0"], [0.30, "#fef9e7"],
        [0.55, "#fdebd0"], [1.00, "#fadbd8"],
    ]
    fig = go.Figure(go.Heatmap(
        z=sub[cols].T.values,
        x=[str(d.date()) for d in sub.index],
        y=[METRIC_KO[c] for c in cols],
        colorscale=colorscale, zmin=0, zmax=100,
        colorbar=dict(title="점수", thickness=10, len=0.9),
        hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=8, b=40),
        xaxis=dict(tickangle=-45, nticks=8, showgrid=False,
                   tickfont=dict(color="#2c3e50", size=9)),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#2c3e50", size=10)),
        coloraxis_colorbar=dict(title=dict(text="점수", font=dict(color="#2c3e50")),
                                tickfont=dict(color="#2c3e50")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#2c3e50", family="system-ui, -apple-system, sans-serif"),
    )
    return fig


def build_crash_chart(bi, spx, spx_dd, start="2004-01-01"):
    start_ts = pd.Timestamp(start)
    bi     = bi[bi.index >= start_ts]
    spx    = spx[spx.index >= start_ts]
    spx_dd = spx_dd[spx_dd.index >= start_ts]

    # 공통 인덱스 정렬 — customdata로 3개 값을 각 trace에 모두 담음
    common_idx = bi.index.intersection(spx.index).intersection(spx_dd.index)
    bi_a   = bi.reindex(common_idx).ffill()
    spx_a  = spx.reindex(common_idx).ffill()
    dd_a   = spx_dd.reindex(common_idx).ffill()
    # customdata columns: [0]=bi, [1]=spx, [2]=dd
    cdata  = np.column_stack([bi_a.values, spx_a.values, dd_a.values])

    # 모든 패널에서 호버 시 3개 값을 한 번에 표시하는 템플릿
    _htpl = (
        "<b>📅 %{x|%Y-%m-%d}</b><br>"
        "<span style='color:#1a1f2e'>🫧 Bubble Index : <b>%{customdata[0]:.1f}</b></span><br>"
        "<span style='color:#2980b9'>📈 S&amp;P 500 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>%{customdata[1]:,.0f}</b></span><br>"
        "<span style='color:#c0392b'>📉 SPX 낙폭 &nbsp;&nbsp;&nbsp;&nbsp;: <b>%{customdata[2]:.1f}%</b></span>"
        "<extra></extra>"
    )

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
        row_heights=[0.44, 0.28, 0.28],
        vertical_spacing=0.04,
        subplot_titles=["AI Bubble Index", "S&P 500 (로그 스케일)", "SPX 고점 대비 낙폭 (%)"],
    )

    # ── Panel 1: Bubble Index (메인) ──
    for y0, y1, r in [(0,30,"green"),(30,55,"yellow"),(55,75,"orange"),(75,100,"red")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=REGIME_BG[r], line_width=0, layer="below", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bi_a.index, y=bi_a.values, name="Bubble Index",
        customdata=cdata, line=dict(color="#1a1f2e", width=2.5), showlegend=True,
        hovertemplate=_htpl,
    ), row=1, col=1, secondary_y=False)
    fig.add_hline(y=70, line_dash="dash", line_color="#e67e22", line_width=1.2, row=1, col=1)

    # ── Panel 2: S&P 500 (메인) + Bubble Index (보조 우측 축, 반투명) ──
    fig.add_trace(go.Scatter(
        x=spx_a.index, y=spx_a.values, name="S&P 500",
        customdata=cdata, line=dict(color="#2980b9", width=1.8), showlegend=True,
        hovertemplate=_htpl,
    ), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=bi_a.index, y=bi_a.values, name="Bubble Index (참조)",
        line=dict(color="#1a1f2e", width=1.2, dash="dot"),
        opacity=0.45, showlegend=True, hoverinfo="skip",
        visible="legendonly",
    ), row=2, col=1, secondary_y=True)

    # ── Panel 3: SPX 낙폭 (메인) + Bubble Index (보조 우측 축, 반투명) ──
    fig.add_trace(go.Scatter(
        x=dd_a.index, y=dd_a.values, name="SPX 낙폭",
        customdata=cdata, fill="tozeroy", fillcolor="rgba(231,76,60,.25)",
        line=dict(color="#c0392b", width=1), showlegend=True,
        hovertemplate=_htpl,
    ), row=3, col=1, secondary_y=False)
    fig.add_hline(y=-20, line_dash="dot", line_color="#7f8c8d", line_width=1, row=3, col=1)

    fig.add_trace(go.Scatter(
        x=bi_a.index, y=bi_a.values, name="Bubble Index (참조)",
        line=dict(color="#1a1f2e", width=1.2, dash="dot"),
        opacity=0.45, showlegend=False, hoverinfo="skip",
        visible="legendonly",
    ), row=3, col=1, secondary_y=True)

    # 보조 y축(Bubble Index 참조선) 설정 — 0~100 고정, 눈금 오른쪽에 표시
    for row in [2, 3]:
        fig.update_yaxes(range=[0, 100], secondary_y=True, row=row, col=1,
                         tickfont=dict(color="#aaaaaa", size=9),
                         showgrid=False, zeroline=False,
                         title_text="Index", title_font=dict(color="#aaaaaa", size=9))

    # Crash event overlays
    event_colors = {"GFC": "#c0392b", "post_covid": "#8e44ad",
                    "Post-COVID 랠리": "#8e44ad", "post_covid": "#8e44ad"}
    for ev_label, _, crash_top, crash_bot, in CRASH_EVENTS:
        ec = "#c0392b" if "gfc" in ev_label.lower() else "#8e44ad"
        ct, cb = pd.Timestamp(crash_top), pd.Timestamp(crash_bot)
        for r in range(1, 4):
            fig.add_vrect(x0=ct, x1=cb, fillcolor=ec, opacity=0.08,
                          line_width=0, row=r, col=1)
            fig.add_vline(x=ct.timestamp()*1000, line_dash="dash",
                          line_color=ec, line_width=1.2, opacity=0.7, row=r, col=1)

        # First ≥70 crossing marker on row 1
        search_start = pd.Timestamp("2005-01-01") if "gfc" in ev_label.lower() else pd.Timestamp("2020-06-01")
        window = bi[(bi.index >= search_start) & (bi.index <= ct)]
        above = window[window >= 70]
        if not above.empty:
            f70 = above.index[0]
            fig.add_vline(x=f70.timestamp()*1000, line_dash="solid",
                          line_color=ec, line_width=2.2, row=1, col=1)
            fig.add_annotation(
                x=f70, y=93, text=f"<b>첫 경보</b>",
                showarrow=False, font=dict(size=9, color=ec),
                bgcolor="white", bordercolor=ec, borderwidth=1,
                borderpad=3, row=1, col=1,
            )

    # 주(primary) y축 — secondary_y=False 명시해야 보조축에 영향 안 줌
    fig.update_yaxes(range=[0, 100], secondary_y=False, row=1, col=1,
                     gridcolor="#e8ecf3", tickfont=dict(color="#2c3e50"))
    fig.update_yaxes(type="log",     secondary_y=False, row=2, col=1,
                     gridcolor="#e8ecf3", tickfont=dict(color="#2c3e50"))
    fig.update_yaxes(range=[-65, 5], secondary_y=False, row=3, col=1,
                     gridcolor="#e8ecf3", tickfont=dict(color="#2c3e50"))

    # 보조(secondary) y축 — Bubble Index 참조선 스케일 (0~100 선형)
    _sec = dict(range=[0, 100], secondary_y=True,
                showgrid=False, zeroline=False,
                tickfont=dict(color="#b0b8c8", size=8),
                title_text="Index", title_font=dict(color="#b0b8c8", size=8))
    fig.update_yaxes(**_sec, row=2, col=1)
    fig.update_yaxes(**_sec, row=3, col=1)
    # 패널1 보조축은 사용 안 하므로 숨김
    fig.update_yaxes(visible=False, secondary_y=True, row=1, col=1)

    # x축: 각 row 개별 적용
    _spike = dict(
        showgrid=False, tickfont=dict(color="#2c3e50"),
        showspikes=True, spikemode="across+toaxis",
        spikedash="dot", spikecolor="#444444",
        spikethickness=2, spikesnap="cursor",
    )
    fig.update_xaxes(**_spike, row=1, col=1)
    fig.update_xaxes(**_spike, row=2, col=1)
    fig.update_xaxes(**_spike, row=3, col=1)

    fig.update_layout(
        height=620,
        margin=dict(l=50, r=55, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#2c3e50")),
        hovermode="x",
        hoversubplots="axis",
        hoverdistance=200,
        spikedistance=1000,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#2c3e50", family="system-ui, -apple-system, sans-serif"),
    )
    return fig

# ── KPI card helper ───────────────────────────────────────────────────────────
def kpi(label, value, sub="", accent="#e67e22"):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def section(icon, title):
    st.markdown(f'<div class="sec-header"><span>{icon}</span>{title}</div>', unsafe_allow_html=True)


def insight(title, body, accent="#e67e22"):
    st.markdown(f"""
    <div class="insight-card" style="--accent:{accent}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    scores_df  = load_scores()
    norm_df    = load_norm()
    bt_df      = load_bt()
    sv_df      = load_sv()
    metrics_df = load_metrics()

    valid = scores_df.dropna(subset=["bubble_index"])
    latest = valid.iloc[-1]
    latest_date  = latest.name.date()
    bi_val  = latest["bubble_index"]
    regime  = regime_label(bi_val)
    r_color = REGIME_HEX[regime]
    r_light = REGIME_LIGHT[regime]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:40px">🫧</div>
            <div style="font-size:20px; font-weight:800; letter-spacing:.5px">AI Bubble Index</div>
            <div style="font-size:11px; color:#8d9db6; margin-top:4px">신영증권 김효진 박사 4-Pillar 모델</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{r_color}18; border:1.5px solid {r_color}; border-radius:12px;
                    padding:14px; text-align:center; margin:12px 0;">
            <div style="font-size:36px; font-weight:900; color:{r_color}">{bi_val:.1f}</div>
            <div style="font-size:13px; font-weight:700; color:{r_color}; letter-spacing:.4px">
                {REGIME_KO[regime]}</div>
            <div style="font-size:11px; color:#8d9db6; margin-top:4px">{latest_date}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="color:#8d9db6;font-size:11px;margin:16px 0 4px">📅 차트 시작일</div>',
                    unsafe_allow_html=True)
        min_d = valid.index.min().date()
        max_d = valid.index.max().date()
        start_date = st.date_input("", value=pd.Timestamp("2006-01-01").date(),
                                   min_value=min_d, max_value=max_d, label_visibility="collapsed")

        st.markdown('<hr style="border-color:#2d3561; margin:16px 0">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:#8d9db6">데이터: {min_d} ~ {max_d}</div>',
                    unsafe_allow_html=True)
        if st.button("🔄  데이터 새로고침"):
            st.cache_data.clear(); st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  메인 대시보드", "🔍  폭락 연관성 검증", "📖  방법론"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Main Dashboard
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # KPI row — 종합 지수 / 레짐 / 4개 필라 (점수 높은 순 정렬)
        PILLAR_WEIGHT = {
            "bond_vigilantes": "30%",
            "concentration":   "20%",
            "private_credit":  "35%",
            "ipo_saturation":  "15%",
        }
        pillar_order = sorted(
            PILLAR_WEIGHT.keys(),
            key=lambda p: latest.get(p, 0) or 0,
            reverse=True,
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi("종합 지수", f"{bi_val:.1f}", "/ 100", r_color)
        with c2: kpi("레짐", REGIME_KO[regime], regime.upper(), r_color)
        for col, p in zip([c3, c4, c5, c6], pillar_order):
            v = latest.get(p)
            with col:
                kpi(PILLAR_KO[p], f"{v:.0f}" if v and not pd.isna(v) else "—",
                    f"{PILLAR_WEIGHT[p]} 가중", PILLAR_COLOR[p])

        st.markdown("<br>", unsafe_allow_html=True)

        # Regime banner
        regime_emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}[regime]
        regime_msg   = {
            "green":  "거품 신호 없음 — 위험 수위가 낮습니다.",
            "yellow": "주의 구간 — 일부 지표에서 과열 신호가 나타나고 있습니다.",
            "orange": "경계 구간 — 복수 지표가 과열 수준에 근접했습니다. 리스크 관리를 권장합니다.",
            "red":    "위험 구간 — 역사적으로 대형 폭락이 발생한 수준입니다.",
        }[regime]
        st.markdown(f"""
        <div class="regime-banner" style="background:{r_light}; border:1.5px solid {r_color}40;">
            <span style="font-size:22px">{regime_emoji}</span>
            <span style="color:{r_color}"><b>{latest_date} &nbsp;|&nbsp; {bi_val:.1f} / 100
            &nbsp;|&nbsp; {regime.upper()}</b> &nbsp;— {regime_msg}</span>
        </div>""", unsafe_allow_html=True)

        # Narrative analysis
        latest_metrics = (
            metrics_df.dropna(how="all").iloc[-1]
            if not metrics_df.empty else pd.Series(dtype=float)
        )
        st.markdown(
            build_narrative_html(scores_df, metrics_df, latest, bt_df),
            unsafe_allow_html=True,
        )

        # Main chart
        section("📈", "종합 지수 추이")
        st.plotly_chart(build_main_chart(valid, pd.Timestamp(start_date)),
                        use_container_width=True)

        # Pillar + Heatmap
        col_l, col_r = st.columns(2)
        with col_l:
            section("🏛️", "기둥별 현재 점수")
            st.plotly_chart(build_pillar_chart(latest), use_container_width=True)
        with col_r:
            section("🌡️", "지표 히트맵 (최근 60일)")
            st.plotly_chart(build_heatmap(norm_df), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sensitivity
        section("🎚️", "가중치 민감도 분석")
        if not sv_df.empty:
            disp = sv_df.copy()
            disp["pillar"] = disp["pillar"].map(lambda x: PILLAR_KO.get(x, x))
            disp = disp.rename(columns={
                "pillar": "기둥", "weight_pct": "가중치(%)",
                "pillar_score": "기둥 점수", "composite_base": "기준 지수",
                "+5pp": "+5pp", "-5pp": "-5pp", "impact_range": "영향 범위",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption("±5pp 가중치 변화 시 종합 지수 변동폭. 영향 범위가 클수록 민감한 기둥.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Validation / Crash Correlation
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        bi_s, spx_s, spx_dd_s, fwd_table = compute_validation()

        # Key insight cards
        section("💡", "핵심 검증 결과")
        i1, i2, i3 = st.columns(3)
        with i1:
            insight("✅ GFC (2007~2009) — 선행 신호 확인",
                    "지수가 <b>70</b>을 넘은 시점이 시장 고점 <b>474일(약 1.3년) 전</b>이었습니다. "
                    "고점 당일 지수는 <b>76.2 (RED)</b>로 위험 구간을 유지했습니다.",
                    "#c0392b")
        with i2:
            insight("✅ Post-COVID 랠리 (2021~2022) — 선행 신호 확인",
                    "지수가 70을 넘은 시점이 시장 고점 <b>48일 전</b>이었습니다. "
                    "SPX 실제 낙폭 <b>-25.4%</b> 기간 동안 지수가 꾸준히 높은 수준을 유지했습니다.",
                    "#8e44ad")
        with i3:
            insight("⚠️ 한계: 완벽한 타이밍 예측기가 아닙니다",
                    "2014~2015년, 2017~2018년에도 지수가 높았지만 대형 폭락은 없었습니다. "
                    "12개월 상관계수 <b>r = -0.167</b> — 방향성은 있지만 신호 강도는 약합니다.",
                    "#7f8c8d")

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive 3-panel chart
        section("📉", "지수 vs S&P 500 — 전체 기간 비교")
        st.plotly_chart(build_crash_chart(bi_s, spx_s, spx_dd_s, str(start_date)),
                        use_container_width=True)
        st.caption("실선 수직선: 첫 경보(지수 70 돌파) | 점선 수직선: 시장 고점 | 음영: 폭락 구간")

        st.markdown("<br>", unsafe_allow_html=True)

        # Forward returns table
        col_a, col_b = st.columns([3, 2])
        with col_a:
            section("📊", "레짐별 향후 수익률 분포 (SPX)")
            if not fwd_table.empty:
                pivot = fwd_table.pivot(index="레짐", columns="기간",
                                        values="평균 수익률(%)")
                pivot = pivot[["1개월", "3개월", "6개월", "12개월"]]
                def color_cell(v):
                    if pd.isna(v): return ""
                    if v < -2: return "background-color:#fadbd8; color:#922b21"
                    if v < 0:  return "background-color:#fef0ed; color:#922b21"
                    if v > 5:  return "background-color:#d5f5e3; color:#1e8449"
                    return "background-color:#fef9e7; color:#7d6608"
                st.dataframe(pivot.style.map(color_cell).format("{:+.1f}%"),
                             use_container_width=True)
                st.caption("각 셀: 해당 레짐에서 향후 N개월 평균 SPX 수익률")

        with col_b:
            section("🗓️", "이벤트별 상세")
            if not bt_df.empty:
                disp = bt_df.rename(columns={
                    "event": "이벤트", "crash_top": "고점",
                    "index_peak": "피크 지수", "regime_at_peak": "레짐",
                    "lead_days": "선행일수", "spx_drawdown_pct": "SPX 낙폭",
                })
                st.dataframe(disp[["이벤트","고점","피크 지수","레짐","선행일수","SPX 낙폭"]],
                             use_container_width=True, hide_index=True)
                st.caption("선행일수: 지수 피크 → 시장 고점까지 일수")

        # Threshold 70 drawdown summary
        st.markdown("<br>", unsafe_allow_html=True)
        section("📐", "Bubble Index ≥ 70 진입 후 SPX 최대 낙폭 (평균)")
        dd_cols = st.columns(3)
        labels_h = ["6개월 이내", "12개월 이내", "24개월 이내"]
        avgs = [-5.7, -6.9, -13.9]
        worsts = [-21.1, -23.9, -55.6]
        for col, lbl, avg, worst in zip(dd_cols, labels_h, avgs, worsts):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:#e74c3c">
                    <div class="kpi-label">{lbl} 평균 낙폭</div>
                    <div class="kpi-value" style="color:#e74c3c">{avg:.1f}%</div>
                    <div class="kpi-sub">최악: {worst:.1f}%</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Methodology
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        section("🏗️", "4-Pillar 방법론")
        mc1, mc2 = st.columns(2)
        pillar_info = [
            ("사모 크레딧", "35%", "private_credit",
             "HY OAS 스프레드, CCC 스프레드, BDC ETF (BIZD), 레버리지론 (BKLN)",
             "결정적 후행 — Minsky 순간의 방아쇠로 작동. 신용시장이 무너지면 주식 폭락 확정."),
            ("채권 자경단", "30%", "bond_vigilantes",
             "미국 30Y/10Y 금리, 실질금리 (TIPS), 텀 프리미엄 (ACM), MOVE 지수",
             "동행 — 채권시장이 자산 가격의 과도한 낙관을 제어하는 역할."),
            ("주도주 압착", "20%", "concentration",
             "S&P 500 vs 등가중 지수 12개월 수익률 스프레드",
             "선행 — 소수 종목에 시장이 집중되면 거품의 초기 신호."),
            ("IPO 포화", "15%", "ipo_saturation",
             "Renaissance IPO ETF, IPO vs SPY 상대강도",
             "동행~후행 — 투기적 상장이 폭발적으로 증가하면 낙관 정점 신호."),
        ]
        for i, (name, weight, key, metrics, desc) in enumerate(pillar_info):
            col = mc1 if i % 2 == 0 else mc2
            color = PILLAR_COLOR[key]
            with col:
                st.markdown(f"""
                <div class="insight-card" style="--accent:{color}; margin-bottom:14px">
                    <div class="insight-title" style="color:{color}; font-size:15px">
                        {name} <span style="font-size:12px;color:#8492a6;font-weight:400">가중 {weight}</span>
                    </div>
                    <div class="insight-body">
                        <b>지표:</b> {metrics}<br><br>{desc}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("⚙️", "계산 방법")
        st.markdown("""
<style>
.calc-table { width:100%; border-collapse:collapse; margin-bottom:16px; }
.calc-table th {
    background:#2c3e50; color:#ecf0f1;
    padding:10px 14px; text-align:left;
    font-size:13px; font-weight:600; border-bottom:2px solid #e67e22;
}
.calc-table td {
    padding:10px 14px; border-bottom:1px solid #dde3ec;
    color:#1a1d27; font-size:13px; vertical-align:top;
    background:#ffffff;
}
.calc-table tr:nth-child(even) td { background:#f4f6fb; }
.calc-table tr:hover td { background:#eaf0fb; }
.calc-table .step-num {
    color:#e67e22; font-weight:700; font-size:15px; text-align:center;
}
.calc-table .step-name { font-weight:600; color:#2c3e50; }
.calc-table code {
    background:#f0f3f8; color:#c0392b;
    padding:1px 5px; border-radius:3px; font-size:12px;
}
.regime-bar {
    margin-top:8px; padding:10px 16px; border-radius:8px;
    background:#2c3e50; color:#ecf0f1; font-size:13px; font-weight:500;
}
.regime-bar span { font-weight:700; }
</style>
<table class="calc-table">
  <thead>
    <tr><th>단계</th><th>처리</th><th>설명</th></tr>
  </thead>
  <tbody>
    <tr>
      <td class="step-num">1</td>
      <td class="step-name">원시 지표 수집</td>
      <td>FRED, Yahoo Finance, NY Fed에서 일별 데이터 수집</td>
    </tr>
    <tr>
      <td class="step-num">2</td>
      <td class="step-name">롤링 백분위 정규화</td>
      <td>24개월(504거래일) 이동 창에서 현재 값의 백분위 순위 → 0~100</td>
    </tr>
    <tr>
      <td class="step-num">3</td>
      <td class="step-name">방향 반전</td>
      <td>스프레드 계열(HY OAS 등)은 좁을수록 거품 → <code>100 - 백분위</code></td>
    </tr>
    <tr>
      <td class="step-num">4</td>
      <td class="step-name">기둥 내 평균</td>
      <td>동일 기둥의 지표들을 산술평균</td>
    </tr>
    <tr>
      <td class="step-num">5</td>
      <td class="step-name">불투명성 페널티</td>
      <td>정보 공개 의무가 낮은 필라(사모 크레딧)에 상향 보정 배수 적용 — 현재 ×1.20</td>
    </tr>
    <tr>
      <td class="step-num">6</td>
      <td class="step-name">가중 합산</td>
      <td>이론 가중치로 기둥 점수를 가중 평균</td>
    </tr>
    <tr>
      <td class="step-num">7</td>
      <td class="step-name">EMA 스무딩</td>
      <td>28일 지수이동평균 적용 (노이즈 제거)</td>
    </tr>
  </tbody>
</table>
<div class="regime-bar">
  📊 레짐 기준 &nbsp;|&nbsp;
  <span style="color:#27ae60">Green ≤ 30</span> &nbsp;＜&nbsp;
  <span style="color:#f39c12">Yellow ≤ 55</span> &nbsp;＜&nbsp;
  <span style="color:#e67e22">Orange ≤ 75</span> &nbsp;＜&nbsp;
  <span style="color:#e74c3c">Red</span>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
