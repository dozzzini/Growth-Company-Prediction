import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# ✅ (오류 가능 지점 수정 1) wordcloud 미설치 환경 대비 (ModuleNotFoundError 방지)
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

import re
import os

# --- 1) 페이지 기본 설정 ---
st.set_page_config(page_title="Sesac Enterprise AI", layout="wide")

# --- 2) 초기 세션 상태 설정 ---
if "search_input" not in st.session_state:
    st.session_state.search_input = ""

# ✅ (추가) 허용 기업 리스트 및 정규화 로직
ALLOWED_COMPANIES = ["삼성전자", "SK하이닉스", "LG에너지솔루션", "현대자동차", "NAVER", "카카오"]

def _normalize_company_name(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    s = re.sub(r"[^0-9a-zA-Z가-힣]", "", s)
    s = s.upper()
    return s

NORMALIZED_TO_CANONICAL = {_normalize_company_name(name): name for name in ALLOWED_COMPANIES}

# --- 3) 화면별 커스텀 CSS 정의 ---
def load_css(is_result_page: bool) -> None:
    # ✅ 사이드바 공통 스타일 정의 (요청: 이미지와 유사한 톤 + 글씨 크기 키움)
    sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #a1b1bf !important;   /* 이미지처럼 회청색 */
        border-right: 1px solid rgba(0,0,0,0.08);
    }

    /* 사이드바 전체 글자(라디오/라벨/버튼) 크기 키우기 */
    [data-testid="stSidebar"] * {
        color: #1f1f1f !important;
        font-size: 1.12rem !important; /* 전체적으로 키움 */
    }

    /* 라디오 항목 텍스트(■ 기업분석 개요 등) 더 크게 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div p {
        font-size: 1.18rem !important;
        font-weight: 600 !important;
        line-height: 1.35 !important;
    }
    [data-testid="stPlotlyChart"] > div {
        background: transparent !important;
    }
    /* Menu(라디오 그룹 상단) 같은 라벨이 보일 때 대비 */
    [data-testid="stSidebar"] label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    /* 브랜드 */
    .brand-side {
        color: #1f1f1f !important;
        font-size: 26px !important;   /* 기존 24 -> 26 */
        font-weight: 900 !important;
        padding: 18px 0 8px 0;
        text-align: left;             /* 이미지처럼 좌측 정렬 느낌 */
    }

    /* 구분선 */
    [data-testid="stSidebar"] hr {
        border-color: rgba(0,0,0,0.18) !important;
    }

    /* '처음으로 돌아가기' 버튼: 이미지처럼 밝은 버튼 */
    [data-testid="stSidebar"] div.stButton > button {
        background: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid rgba(0,0,0,0.14) !important;
        border-radius: 10px !important;
        height: 44px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    </style>
    """
    st.markdown("""
    <style>
    .core-card hr {
    border: none;
    border-top: 1px solid rgba(0,0,0,0.08);
    margin: 16px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(sidebar_style, unsafe_allow_html=True)

    if not is_result_page:
        st.markdown(
            """
            <style>
            header {visibility: hidden;}
            .stApp {
                background-color: #ffffff;
                background-image: radial-gradient(at 100% 100%, rgba(220, 235, 255, 1) 0px, transparent 50%),
                                 radial-gradient(at 0% 0%, rgba(235, 230, 255, 1) 0px, transparent 50%);
            }
            .navbar {
                display: flex; justify-content: space-between; align-items: center;
                padding: 15px 60px; background: rgba(255, 255, 255, 0.95);
                border-bottom: 1px solid #f0f0f0; position: fixed; top: 0; left: 0; right: 0; z-index: 999;
            }
            .brand { color: #5c67f2; font-size: 26px; font-weight: bold; text-decoration: none; }
            .nav-items a { color: #333; text-decoration: none; margin-left: 30px; font-size: 15px; }
            .btn-login { background-color: #5c67f2 !important; color: white !important; padding: 8px 18px; border-radius: 6px; }
            .main-hero { text-align: center; padding-top: 180px; }
            .hero-subtitle { font-size: 18px; font-weight: 500; color: #000000; margin: 0 0 12px 0; }
            .hero-title-main { font-size: 80px; font-weight: 900; color: #5c67f2; margin: 0; }
            div.stButton > button {
                background-color: #5c67f2 !important; color: white !important;
                width: 100%; height: 65px; border-radius: 12px !important;
                font-size: 24px !important; font-weight: 800 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            header {visibility: hidden;}
            .stApp { background-color: #f8f9fc; }
            .dashboard-card {
                background-color: white; padding: 25px; border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px;
                border: 1px solid #f0f0f0;
            }
            .kpi-card {
                background-color: white; padding: 25px; border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px;
                text-align: center; border-top: 4px solid #5c67f2;
            }
            .kpi-title { font-size: 14px; color: #666; font-weight: 600; margin-bottom: 10px; }
            .kpi-value { font-size: 32px; font-weight: 800; color: #5c67f2; }
            .fin-btn > button {
                background-color: #f8f9fc !important; color: #5c67f2 !important;
                border: 1px solid #e0e0e0 !important; border-radius: 8px !important;
                height: 40px !important; font-size: 14px !important; font-weight: 600 !important;
                width: 100%;
            }
            .fin-btn-selected > button {
                background-color: #5c67f2 !important; color: white !important;
                border: 1px solid #5c67f2 !important; border-radius: 8px !important;
                height: 40px !important; font-size: 14px !important; font-weight: 600 !important;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# (그래프 함수 gauge_figure, build_wordcloud 등은 이전과 동일하게 유지)
def gauge_figure(value: float, title: str = "") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={"suffix": "%", "font": {"size": 24, "color": "#5c67f2"}},
        title={"text": title, "font": {"size": 14, "color": "#666"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#5c67f2"}, "bgcolor": "#f0f2f6"}
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def build_wordcloud(text: str):
    # ✅ (오류 가능 지점 수정 2) WordCloud 미설치 시에도 앱이 죽지 않도록 대체 그림 반환
    if WordCloud is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "wordcloud 패키지가 설치되어 있지 않습니다.", ha="center", va="center")
        ax.axis("off")
        fig.patch.set_facecolor('none')
        return fig

    font_path = None
    candidates = ["C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    for p in candidates:
        if os.path.exists(p): font_path = p; break
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text).strip()
    wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white", colormap='Blues').generate(cleaned if cleaned else " ")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor('none')
    return fig

# --- 4) 메인 로직 분기 ---
if not st.session_state.search_input:
    load_css(is_result_page=False)
    st.markdown('<div class="navbar"><div class="brand">Sesac</div><div class="nav-items"><a href="#">분석스튜디오</a><a href="#">서비스</a><a href="#">요금제</a><span style="margin-left:30px;">🔔</span><a href="#">회원가입</a><a href="#" class="btn-login">로그인</a></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-hero"><div class="hero-subtitle">핵심 기술의 가치를 발견하는 가장 앞선 감각</div><div class="hero-title-main">AI 기반 기술기업 성장 예측</div></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        # ✅ 기존 st.text_input과 st.expander를 제거하고 st.selectbox로 교체
        # index=None을 설정하면 처음에는 아무것도 선택되지 않은 빈 칸으로 보입니다.
        selected_company = st.selectbox(
            "기업명 입력",
            options=ALLOWED_COMPANIES,
            index=None,
            placeholder="분석할 기업명을 선택하거나 입력하세요",
            label_visibility="collapsed"
        )

        # ✅ '분석 시작하기' 버튼 로직
        if st.button("분석 시작하기"):
            if selected_company is None:
                st.warning("분석할 기업명을 선택해 주세요.")
            else:
                # 선택된 기업명(표준 명칭)을 세션에 저장
                st.session_state.search_input = selected_company
                st.rerun()
else:
    load_css(is_result_page=True)
    with st.sidebar:
        st.markdown('<div class="brand-side">Sesac AI</div>', unsafe_allow_html=True)
        menu_choice = st.radio("Menu", ["■ 기업분석 개요", "■ 뉴스", "■ GenAI 보고서", "■ 최근 본 기업", "■ 관심 기업", "■ 추천 기업"], label_visibility="collapsed")
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("← 처음으로 돌아가기"):
            st.session_state.search_input = ""
            st.rerun()

    # (이후 페이지별 섹션 코드는 이전과 동일하게 유지)
    search_term = st.session_state.search_input
    st.markdown(f"<div class='dashboard-card' style='text-align:center'><h2>{search_term} <span style='color:#5c67f2; font-size:18px'>000000</span></h2></div>", unsafe_allow_html=True)

    if menu_choice == "■ 뉴스":
        if 'recent_viewed_logs' not in st.session_state:
            st.session_state.recent_viewed_logs = []

        st.markdown("### 📰 뉴스 분석 센터")

        categories = ["반도체", "이차전지", "디스플레이"]
        selected_category = st.radio(
            "산업 카테고리 선택", 
            categories, 
            horizontal=True
        )
        
        st.markdown("---")

        news_data = [
            {"title": f"[{selected_category}] 차세대 기술 확보를 위한 글로벌 경쟁 가속화", "date": "2026-01-26", "source": "경제뉴스"},
            {"title": f"{selected_category} 산업, 상반기 수출 실적 역대 최고치 경신 전망", "date": "2026-01-25", "source": "IT타임즈"},
            {"title": f"글로벌 공급망 재편에 따른 {selected_category} 기업의 대응 전략", "date": "2026-01-25", "source": "산업일보"},
            {"title": f"신규 시설 투자 공시: {selected_category} 생산 라인 대폭 증설", "date": "2026-01-24", "source": "금융신문"},
        ]

        col_news_left, col_news_right = st.columns([1.2, 1], gap="large")

        with col_news_left:
            st.markdown(f"#### 📢 {selected_category} 최근 뉴스")
            
            # 버튼 디자인을 텍스트처럼 만들기 위한 CSS
            st.markdown("""
                <style>
                div[data-testid="stButton"] > button[kind="tertiary"] {
                    padding: 0px;
                    border: none;
                    height: auto;
                    line-height: 1.5;
                    color: #1f1f1f;
                    background-color: transparent;
                    font-size: 1.05rem;
                    text-align: left;
                }
                div[data-testid="stButton"] > button[kind="tertiary"]:hover {
                    color: #5c67f2;
                    text-decoration: underline;
                }
                </style>
            """, unsafe_allow_html=True)

            with st.container(border=True):
                for i, news in enumerate(news_data):
                    # ✅ (오류 가능 지점 수정 3) 구버전 Streamlit에서 kind= 파라미터가 에러라 제거
                    if st.button(f"**{news['title']}**", key=f"news_text_btn_{i}"):
                        if news['title'] not in st.session_state.recent_viewed_logs:
                            st.session_state.recent_viewed_logs.insert(0, news['title'])
                            st.session_state.recent_viewed_logs = st.session_state.recent_viewed_logs[:5]
                        st.rerun()
                    
                    st.caption(f"📅 {news['date']} | 🏢 {news['source']}")
                    st.markdown('<hr style="margin:8px 0; border-top:1px solid #f8f9fa;">', unsafe_allow_html=True)
                
                st.button(f"{selected_category} 뉴스 더보기", key="more_news_footer")

        with col_news_right:
            st.markdown("#### ☁️ 뉴스 키워드 한눈에 보기")
            with st.container(border=True):
                news_titles_combined = " ".join([n['title'] for n in news_data])
                fig_wc_news = build_wordcloud(news_titles_combined)
                st.pyplot(fig_wc_news, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### 🕒 최근 본 뉴스")
            with st.container(border=True):
                if st.session_state.recent_viewed_logs:
                    for title in st.session_state.recent_viewed_logs:
                        st.markdown(f"• <span style='font-size:0.85rem;'>{title}</span>", unsafe_allow_html=True)
                else:
                    st.caption("클릭한 뉴스가 여기에 표시됩니다.")


    elif menu_choice == "■ GenAI 보고서":
        st.markdown("### 🤖 AI 분석 리포트")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="kpi-card"><div class="kpi-title">종합 투자 등급</div><div class="kpi-value">Strong Buy</div></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="kpi-card"><div class="kpi-title">AI 예측 성장성</div><div class="kpi-value">매우 높음</div></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="kpi-card"><div class="kpi-title">리스크 수준</div><div class="kpi-value" style="color:#f25c5c">보통</div></div>', unsafe_allow_html=True)

        col_report_left, col_report_right = st.columns([1.5, 1])
        with col_report_left:
            with st.container(border=True):
                st.subheader("📋 핵심 분석 의견")

                # 문단 사이에 빈 줄을 넣어 확실하게 구분합니다.
                st.info(f"""
        **1. 재무 건전성 및 수익성 분석**
        - {search_term}은 최근 3개년 동안 영업이익률이 업종 평균 대비 5%p 상회하고 있습니다.
        - 특히 부채비율이 감소하며 재무 구조가 개선되고 있는 점이 긍정적입니다.

        **2. 기술적 경쟁력 (LSTM 모델 기반)**
        - 당사 AI 모델 예측 결과, 향후 12개월 내 매출액이 약 15~18% 추가 성장할 것으로 전망됩니다.
        - R&D 투자 비중이 지속적으로 상승하고 있어 장기적 성장 동력이 확보되었습니다.
                """)

                st.subheader("💡 전략적 제언")
                st.success("""
        - 반도체 공정 자동화 솔루션의 글로벌 점유율 확대를 위해 북미 시장 마케팅 강화가 필요합니다.
        - 고정비 절감을 위한 공정 디지털 트랜스포메이션(DT) 가속화를 권고합니다.
                """)
        with col_report_right:
            with st.container(border=True):
                st.subheader("📊 부문별 점수")
                fig_radar = px.line_polar(pd.DataFrame({"항목": ["성장","수익","안정","기술","시장"], "점수": [92,85,78,95,88]}), r='점수', theta='항목', line_close=True)
                st.plotly_chart(fig_radar, use_container_width=True)

    elif menu_choice == "■ 기업분석 개요":
        # =========================
        # ✅ 넘버1 반영: 기업분석 개요 섹션 구조/레이아웃 적용
        # =========================

        # 상단 타이틀 및 핵심 KPI
        st.markdown("### 📈 기업 성장 가능성 요약")

        # 상단 6개 카드 섹션
        k_cols = st.columns(6)
        with k_cols[0]:
            st.markdown('<div class="kpi-card"><div class="kpi-title">2024 성장률</div><div class="kpi-value">31.9%</div></div>', unsafe_allow_html=True)
        with k_cols[1]:
            st.markdown('<div class="kpi-card"><div class="kpi-title">2025 예상치</div><div class="kpi-value">31.9%</div></div>', unsafe_allow_html=True)
        with k_cols[2]:
            st.markdown('<div class="kpi-card"><div class="kpi-title">업종 평균</div><div class="kpi-value" style="color:#666">24.2%</div></div>', unsafe_allow_html=True)
        with k_cols[3]:
            st.plotly_chart(gauge_figure(31.9, "안정성 지표"), use_container_width=True)
        with k_cols[4]:
            st.plotly_chart(gauge_figure(24.2, "성장성 지표"), use_container_width=True)
        with k_cols[5]:
            st.plotly_chart(gauge_figure(24.2, "수익성 지표"), use_container_width=True)

        st.markdown("---")

        # 메인 분석 영역 (좌: 재무 / 우: 산업 및 역량)
        main_left, main_right = st.columns([1.2, 1], gap="large")

        with main_left:
            st.markdown("### 📊 연도별 재무 트렌드")

            # --- PL/BS 통합 인터페이스 (넘버1: 탭 구조) ---
            tab1, tab2 = st.tabs(["손익계산서 (P/L)", "재무상태표 (B/S)"])

            with tab1:
                # =========================
                # 📊 2019~2024년 재무정보 (PL) - 버튼 강조/비강조 + 3개 막대 동시표시
                # =========================

                # 버튼 상태 (기본: 매출액)
                if "fin_metric" not in st.session_state:
                    st.session_state.fin_metric = "매출액"

                # 버튼 UI (기존 CSS 클래스 fin-btn / fin-btn-selected 그대로 사용)
                m_cols = st.columns(3)
                for i, m in enumerate(["매출액", "영업이익", "당기순이익"]):
                    with m_cols[i]:
                        cls = "fin-btn-selected" if st.session_state.fin_metric == m else "fin-btn"
                        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                        if st.button(m, key=f"pl_btn_{i}"):
                            st.session_state.fin_metric = m
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                # 데이터
                years = list(range(2019, 2025))
                revenue = [820, 900, 880, 980, 1030, 1100]
                op_profit = [60, 75, 70, 90, 110, 130]
                net_profit = [40, 55, 50, 65, 80, 95]

                # 색상 (요청값 고정)
                COL_REVENUE = "#632bf3"
                COL_OP      = "#00c1a0"
                COL_NET     = "#ffb000"

                # 버튼 선택에 따른 선명도(Opacity)
                selected = st.session_state.fin_metric
                opacity_map = {
                    "매출액":     1.0 if selected == "매출액" else 0.25,
                    "영업이익":   1.0 if selected == "영업이익" else 0.25,
                    "당기순이익": 1.0 if selected == "당기순이익" else 0.25,
                }

                # 막대 얇게 (한 해에 3개 막대가 나란히 서도록)
                bar_width = 0.18

                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=years, y=revenue, name="매출액",
                    marker=dict(color=COL_REVENUE, opacity=opacity_map["매출액"]),
                    width=bar_width,
                    text=[f"{v:,}" for v in revenue],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig1.add_trace(go.Bar(
                    x=years, y=op_profit, name="영업이익",
                    marker=dict(color=COL_OP, opacity=opacity_map["영업이익"]),
                    width=bar_width,
                    text=[f"{v:,}" for v in op_profit],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig1.add_trace(go.Bar(
                    x=years, y=net_profit, name="당기순이익",
                    marker=dict(color=COL_NET, opacity=opacity_map["당기순이익"]),
                    width=bar_width,
                    text=[f"{v:,}" for v in net_profit],
                    textposition="outside",
                    cliponaxis=False
                ))

                fig1.update_layout(
                    barmode="group",
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        tickmode="array",
                        tickvals=years,
                        ticktext=[str(y) for y in years],
                        showgrid=False
                    ),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                )
            
                st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
                
            with tab2:
                # =========================
                # ✅ BS 파트 (자산/부채/자본) - 버튼 강조/비강조 + 3개 막대 동시표시
                # =========================

                # 버튼 상태 (기본: 자산)
                if "bs_metric" not in st.session_state:
                    st.session_state.bs_metric = "자산"

                # 버튼 UI (기존 CSS 클래스 fin-btn / fin-btn-selected 그대로 사용)
                b_cols = st.columns(3)
                for i, m in enumerate(["자산", "부채", "자본"]):
                    with b_cols[i]:
                        cls = "fin-btn-selected" if st.session_state.bs_metric == m else "fin-btn"
                        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                        if st.button(m, key=f"bs_btn_{i}"):
                            st.session_state.bs_metric = m
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

                # 데이터
                years = list(range(2019, 2025))
                assets = [1500, 1600, 1700, 1850, 1950, 2100]
                liab   = [700,  720,  740,  760,  780,  800]
                equity = [800,  880,  960, 1090, 1170, 1300]

                # ✅ 색상 (PL과 동일 팔레트로 재사용: 요청대로 고정)
                COL_ASSET  = "#632bf3"
                COL_LIAB   = "#00c1a0"
                COL_EQUITY = "#ffb000"

                # 버튼 선택에 따른 선명도(Opacity)
                selected_bs = st.session_state.bs_metric
                opacity_map_bs = {
                    "자산":  1.0 if selected_bs == "자산" else 0.25,
                    "부채":  1.0 if selected_bs == "부채" else 0.25,
                    "자본":  1.0 if selected_bs == "자본" else 0.25,
                }

                # 막대 얇게 (한 해에 3개 막대가 나란히)
                bar_width_bs = 0.18

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=years, y=assets, name="자산",
                    marker=dict(color=COL_ASSET, opacity=opacity_map_bs["자산"]),
                    width=bar_width_bs,
                    text=[f"{v:,}" for v in assets],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig2.add_trace(go.Bar(
                    x=years, y=liab, name="부채",
                    marker=dict(color=COL_LIAB, opacity=opacity_map_bs["부채"]),
                    width=bar_width_bs,
                    text=[f"{v:,}" for v in liab],
                    textposition="outside",
                    cliponaxis=False
                ))
                fig2.add_trace(go.Bar(
                    x=years, y=equity, name="자본",
                    marker=dict(color=COL_EQUITY, opacity=opacity_map_bs["자본"]),
                    width=bar_width_bs,
                    text=[f"{v:,}" for v in equity],
                    textposition="outside",
                    cliponaxis=False
                ))

                fig2.update_layout(
                    barmode="group",
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        tickmode="array",
                        tickvals=years,
                        ticktext=[str(y) for y in years],
                        showgrid=False
                    ),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                )
                
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="small-note">※ 원하는 지표를 클릭하면 해당 데이터가 강조되어 표시됩니다. </div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- [임시 데이터 생성] ---
            fin_data = {
                "수익성": {"영업이익률": 15.2, "ROE": 12.5, "ROA": 8.4},
                "안정성": {"부채비율": 85.0, "자기자본비율": 54.0, "유동비율": 120.5},
                "성장성": {"매출액증가율": 22.4, "영업이익증가율": 18.2, "순이익증가율": 15.5}
            }
            st.markdown("---")
            # --- 기업 핵심 역량 진단 (넘버1: 카드 2개 묶음) ---
            # --- [디자인 가이드: 기업 핵심 역량 진단 섹션] ---
            st.markdown("### 🔍 기업 핵심 역량 진단")

            # 전용 CSS: 카드 스타일 및 라벨 디자인
            st.markdown("""
                <style>
                .capability-card {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 15px;
                    border: 1px solid #f0f2f6;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                    margin-bottom: 25px;
                }
                .capability-title {
                    font-size: 18px;
                    font-weight: 700;
                    color: #1f1f1f;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                </style>
            """, unsafe_allow_html=True)

            # 1. 안정성 섹션 (Stability)
            st.markdown('<div class="capability-title">🛡️ 안정성 (Stability)</div>', unsafe_allow_html=True)
            
            stab_cols = st.columns([1, 1.2])
            with stab_cols[0]:
                debt_ratio = fin_data["안정성"]["부채비율"]
                gauge_color = "#ffb000" if debt_ratio > 150 else "#00c1a0"
                fig_debt = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = debt_ratio,
                    title = {'text': "부채비율", 'font': {'size': 15, 'color': '#666'}},
                    number = {'suffix': "%", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 300], 'tickwidth': 1},
                        'bar': {'color': gauge_color},
                        'bgcolor': "#f8f9fa",
                        'steps': [
                            {'range': [0, 150], 'color': "#e9ecef"},
                            {'range': [150, 300], 'color': "#fff3e0"}
                        ],
                        'threshold': {'line': {'color': "#ff5252", 'width': 3}, 'thickness': 0.75, 'value': 200}
                    }
                ))
                fig_debt.update_layout(height=180, margin=dict(l=25, r=25, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_debt, use_container_width=True, config={'displayModeBar': False})

            with stab_cols[1]:
                stability_df = pd.DataFrame({
                    "항목": ["자기자본비율", "유동비율"],
                    "값": [fin_data["안정성"]["자기자본비율"], fin_data["안정성"]["유동비율"]]
                })
                fig_stab = go.Figure(go.Bar(
                    x=stability_df["값"], y=stability_df["항목"], orientation='h',
                    # ✅ (오류 가능 지점 수정 4) plotly에서 cornerradius 미지원/버전차로 ValueError 방지
                    marker=dict(color=['#5c67f2', '#8e99f3']),
                    text=stability_df["값"].map(lambda x: f"{x}%"), textposition='auto',
                    width=0.5
                ))
                fig_stab.update_layout(
                    height=180, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_stab, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # 2. 성장성 섹션 (Growth)
            st.markdown("---")
            st.markdown('<div class="capability-title">🚀 성장성 (Growth)</div>', unsafe_allow_html=True)
            
            growth_metrics = fin_data["성장성"]
            gr_cols = st.columns(len(growth_metrics))
            
            for i, (label, val) in enumerate(growth_metrics.items()):
                with gr_cols[i]:
                    fig_gr = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=val,
                        number={'suffix': "%", 'font': {'size': 28, 'color': '#1f1f1f'}, 'valueformat': '.1f'},
                        title={'text': label, 'font': {'size': 14, 'color': '#666'}},
                        delta={'reference': 5.0, 'position': "bottom", 'increasing': {'color': '#ff4b4b'}, 'decreasing': {'color': '#0366d6'}}
                    ))
                    fig_gr.update_layout(height=140, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gr, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # 3. 수익성 섹션 (Profitability)
            st.markdown("---")
            st.markdown('<div class="capability-title">💰 수익성 (Profitability)</div>', unsafe_allow_html=True)
            
            profit_metrics = fin_data["수익성"]
            pr_cols = st.columns(len(profit_metrics))
            
            for i, (label, val) in enumerate(profit_metrics.items()):
                with pr_cols[i]:
                    fig_pr = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=val,
                        number={'suffix': "%", 'font': {'size': 28, 'color': '#1f1f1f'}, 'valueformat': '.1f'},
                        title={'text': label, 'font': {'size': 14, 'color': '#666'}},
                        delta={'reference': 10.0, 'position': "bottom", 'increasing': {'color': '#ff4b4b'}}
                    ))
                    fig_pr.update_layout(height=140, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pr, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with main_right:
            st.markdown("### 🔍 산업 전방 현황")
            insight_text = "분석된 기업의 특허 경쟁력이 높고 반도체 공정 장비 분야 기술 장벽이 강합니다."
            fig_wc = build_wordcloud(insight_text)

            # 워드클라우드를 카드 내부에 배치 (넘버1)
            st.pyplot(fig_wc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 산업 연관 지표 및 감성 분석")

            def sparkline_figure(series: pd.Series, line_color: str):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(series)),
                    y=series.values,
                    mode="lines",
                    line=dict(width=2, color=line_color),
                    hoverinfo="skip"
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=52,
                    width=120,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False, fixedrange=True),
                    yaxis=dict(visible=False, fixedrange=True),
                    showlegend=False
                )
                fig.layout.template = "plotly_white"
                return fig

            # ✅ 수정: columns(좌/우) 중첩을 제거하여 Streamlit "nesting" 예외 방지
            # (카드 내부는 스파크라인(상단) + 텍스트(하단) 구성으로 유지)
            def render_ticker_card(title: str, price: float, prev_price: float, series: pd.Series):
                diff = price - prev_price
                pct = (diff / prev_price) * 100 if prev_price != 0 else 0.0

                up = diff >= 0
                change_color = "#d32f2f" if up else "#1976d2"  # 상승=빨강, 하락=파랑(국내 관행)
                spark_color = change_color

                # 표기 포맷
                price_str = f"{price:,.2f}"
                diff_str = f"{diff:+,.2f}"
                pct_str  = f"({pct:+.2f}%)"

                st.markdown('<div class="ticker-card">', unsafe_allow_html=True)

                # 스파크라인(상단)
                fig = sparkline_figure(series, spark_color)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

                # 텍스트(하단)
                st.markdown(f"""
                <div class="ticker-inner">
                  <div>
                    <p class="ticker-title">{title}</p>
                    <p class="ticker-price">{price_str}
                      <span class="ticker-change" style="color:{change_color}">{diff_str} {pct_str}</span>
                    </p>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            np.random.seed(7)

            def fake_series(start=1000, n=40, drift=0.0, vol=8.0):
                steps = np.random.normal(drift, vol, n)
                vals = start + np.cumsum(steps)
                return pd.Series(vals)

            data = [
                {
                    "title": "필라델피아 반도체 지수",
                    "series": fake_series(start=1450, n=45, drift=-0.6, vol=2.5),
                },
                {
                    "title": "반도체 및 전자부품 제조업 생산자물가지수(PPI)",
                    "series": fake_series(start=4990, n=45, drift=-0.8, vol=7.0),
                },
                {
                    "title": "반도체 수출가격지수",
                    "series": fake_series(start=995, n=45, drift=+1.2, vol=5.0),
                }
            ]

            for d in data:
                d["prev"] = float(d["series"].iloc[-2])
                d["price"] = float(d["series"].iloc[-1])

            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                render_ticker_card(data[0]["title"], data[0]["price"], data[0]["prev"], data[0]["series"])
            with c2:
                render_ticker_card(data[1]["title"], data[1]["price"], data[1]["prev"], data[1]["series"])
            with c3:
                render_ticker_card(data[2]["title"], data[2]["price"], data[2]["prev"], data[2]["series"])

            st.markdown('<div class="small-note">※ 출처 : https://fred.stlouisfed.org/series/IR21320 </div>', unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 긍부정 속성 분석")

            months = ['2022.11', '2022.12', '2023.01', '2023.02', '2023.03', '2023.04', '2023.05']
            pos = [50000, 45000, 52000, 60000, 58000, 65000, 48000]
            neu = [8000, 7000, 9000, 10000, 9500, 11000, 8500]
            neg = [12000, 10000, 15000, 18000, 17000, 20000, 14000]

            colors = {
                '긍정': '#632BF3',
                '중립': '#00C1A0',
                '부정': '#FFB000'
            }

            c_trend, c_ratio = st.columns([1.6, 1], gap="large")

            with c_trend:
                st.markdown('<div class="dashboard-card"><b>긍·부정 추이</b>', unsafe_allow_html=True)

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=months, y=pos, name='긍정', marker_color=colors['긍정']))
                fig_bar.add_trace(go.Bar(x=months, y=neu, name='중립', marker_color=colors['중립']))
                fig_bar.add_trace(go.Bar(x=months, y=neg, name='부정', marker_color=colors['부정']))

                fig_bar.update_layout(
                    barmode='stack',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=10, b=10),
                    xaxis=dict(
                        type="category",
                        categoryorder="array",
                        categoryarray=months,
                        showgrid=False
                    ),
                    yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with c_ratio:
                st.markdown('<div class="dashboard-card"><b>긍·부정 비율</b>', unsafe_allow_html=True)

                labels = ['긍정', '중립', '부정']
                values = [sum(pos), sum(neu), sum(neg)]

                max_idx = int(np.argmax(values))
                center_label = labels[max_idx]
                center_value = f"{values[max_idx]:,}건"

                fig_pie = go.Figure(
                    data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.62,
                        marker=dict(colors=[colors['긍정'], colors['중립'], colors['부정']]),
                        textinfo='none'
                    )]
                )

                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=10, b=10),
                    annotations=[
                        dict(
                            text=f"{center_label}<br><b>{center_value}</b>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=16)
                        )
                    ],
                )

                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

    elif menu_choice == "■ 최근 본 기업":
        # --- [0] 세션 상태 초기화 (데이터 구조 통합 및 안전한 생성) ---
        if 'viewed_history' not in st.session_state:
            st.session_state.viewed_history = []
        
        # 모든 피드백(하트, 메모)을 하나의 딕셔너리로 관리
        if 'company_feedback' not in st.session_state:
            st.session_state.company_feedback = {}

        st.markdown("### 🕒 최근 본 기업 상세 기록")
        st.markdown("---")

        # 분석 기록 저장 로직
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 첫 화면에서 선택된 기업(search_term)이 있고, 최신 기록과 다를 때만 저장
        if search_term and (not st.session_state.viewed_history or st.session_state.viewed_history[0]['name'] != search_term):
            st.session_state.viewed_history.insert(0, {"name": search_term, "time": current_time})
            st.session_state.viewed_history = st.session_state.viewed_history[:10]

        # --- [1] 3컬럼 레이아웃 구성 ---
        col_list, col_combined, col_action = st.columns(3, gap="medium")

        # 1. 최근 본 기업 목록
        with col_list:
            st.markdown("""<div style="background-color:#f8f9fc; padding:10px; border-radius:8px; border:1px solid #e0e0e0; text-align:center; font-weight:bold;">최근 본 기업 목록</div>""", unsafe_allow_html=True)
            with st.container(border=True):
                if st.session_state.viewed_history:
                    for entry in st.session_state.viewed_history:
                        name = entry['name']
                        # 하트가 등록된 기업은 아이콘 변경
                        feedback = st.session_state.company_feedback.get(name, {"heart": False, "note": ""})
                        icon = "❤️" if feedback.get("heart") else "🏢"
                        st.write(f"{icon} **{name}**")
                else:
                    st.caption("기록이 없습니다.")

        # 2. 열람 시간 및 피드백 (시간 + 하트 + 메모 요약)
        with col_combined:
            st.markdown("""<div style="background-color:#f8f9fc; padding:10px; border-radius:8px; border:1px solid #e0e0e0; text-align:center; font-weight:bold;">열람 시간 및 피드백</div>""", unsafe_allow_html=True)
            with st.container(border=True):
                if st.session_state.viewed_history:
                    for entry in st.session_state.viewed_history:
                        name = entry['name']
                        feedback = st.session_state.company_feedback.get(name, {"heart": False, "note": ""})
                        
                        # 하트 여부와 메모 요약 생성
                        heart_status = "❤️" if feedback.get("heart") else "🤍"
                        raw_note = feedback.get("note", "")
                        note_preview = f" | 📝 {raw_note[:10]}" if raw_note else ""
                        
                        st.write(f"⏰ {entry['time']} {heart_status}{note_preview}")
                else:
                    st.caption("기록이 없습니다.")

        # 3. 나의 피드백 (현재 선택된 기업에 대해 입력)
        with col_action:
            st.markdown("""<div style="background-color:#f8f9fc; padding:10px; border-radius:8px; border:1px solid #e0e0e0; text-align:center; font-weight:bold;">나의 피드백</div>""", unsafe_allow_html=True)
            with st.container(border=True):
                if search_term:
                    st.write(f"📍 **{search_term}** 관리")
                    
                    # 현재 기업의 피드백 데이터가 없으면 초기화
                    if search_term not in st.session_state.company_feedback:
                        st.session_state.company_feedback[search_term] = {"heart": False, "note": ""}
                    
                    # (A) 하트 버튼
                    is_hearted = st.session_state.company_feedback[search_term].get("heart", False)
                    btn_label = "❤️ 관심기업 등록" if not is_hearted else "💔 등록 해제"
                    if st.button(btn_label, key="heart_btn_action"):
                        st.session_state.company_feedback[search_term]["heart"] = not is_hearted
                        st.rerun()
                    
                    # (B) 메모 입력창
                    current_note_val = st.session_state.company_feedback[search_term].get("note", "")
                    # key값에 기업명을 넣어 중복 방지
                    typed_note = st.text_area("기업 메모", value=current_note_val, placeholder="메모를 입력하세요.", key=f"input_note_{search_term}")
                    
                    if st.button("메모 저장", key="save_note_btn"):
                        st.session_state.company_feedback[search_term]["note"] = typed_note
                        st.success("피드백이 저장되었습니다.")
                        st.rerun()
                else:
                    st.info("메인 섹션에서 기업을 먼저 선택해주세요.")

    elif menu_choice == "■ 관심 기업":
        st.markdown("### ⭐ 관심 기업 관리 센터")


        # --- [0] 세션 상태 초기화 ---
        if 'my_interests' not in st.session_state:
            st.session_state.my_interests = {
                "그룹 1": [],
                "그룹 2": [],
                "그룹 3": []
            }
        
        # 칸 개수 상태 관리 (기본 3개)
        if 'num_cols' not in st.session_state:
            st.session_state.num_cols = len(st.session_state.my_interests)

        # --- [1] 상단 설정 영역 (버튼으로 개수 조절) ---
        with st.expander("⚙️ 관심 기업 설정", expanded=True):
            st.write("**표시할 칸 개수 조절**")
            
            # +, - 버튼 배치
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 8])
            with btn_col1:
                if st.button("➖", use_container_width=True):
                    if st.session_state.num_cols > 1:
                        st.session_state.num_cols -= 1
                        st.rerun()
            with btn_col2:
                if st.button("➕", use_container_width=True):
                    if st.session_state.num_cols < 5: # 최대 5개 제한
                        st.session_state.num_cols += 1
                        st.rerun()
            with btn_col3:
                st.info(f"현재 {st.session_state.num_cols}개의 칸이 표시되고 있습니다. (최대 5개)")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # 설정 공간 생성
            current_groups = list(st.session_state.my_interests.keys())
            setting_cols = st.columns(st.session_state.num_cols)
            new_interests = {}

            for i in range(st.session_state.num_cols):
                with setting_cols[i]:
                    # 1. 칸 이름 수정
                    default_name = current_groups[i] if i < len(current_groups) else f"그룹 {i+1}"
                    new_name = st.text_input(f"칸 {i+1} 이름", value=default_name, key=f"col_name_{i}")
                    
                    # 2. 기업 선택/수정
                    current_items = st.session_state.my_interests.get(default_name, [])
                    selected_items = st.multiselect(
                        f"{new_name} 기업 편집",
                        options=ALLOWED_COMPANIES,
                        default=[item for item in current_items if item in ALLOWED_COMPANIES],
                        key=f"col_select_{i}"
                    )
                    new_interests[new_name] = selected_items

            # 변경사항 저장 버튼
            if st.button("보드 업데이트 저장", type="primary", use_container_width=True):
                st.session_state.my_interests = new_interests
                st.success("관심 기업 보드가 업데이트되었습니다.")
                st.rerun()

        st.markdown("---")

        # --- [2] 메인 레이아웃 (이미지 구조 반영) ---
        display_groups = list(st.session_state.my_interests.keys())
        # 현재 설정된 개수만큼 컬럼 생성
        main_cols = st.columns(len(display_groups), gap="medium")

        for i, group_name in enumerate(display_groups):
            with main_cols[i]:
                # 타이틀 박스 (상단)
                st.markdown(f"""
                    <div style="
                        background-color: #f8f9fc;
                        padding: 10px;
                        border-radius: 8px 8px 0 0;
                        border: 1px solid #e0e0e0;
                        text-align: center;
                        font-weight: bold;
                        border-bottom: 2px solid #5c67f2;
                    ">
                        {group_name}
                    </div>
                """, unsafe_allow_html=True)

                # 기업 리스트 박스 (하단)
                with st.container(border=True):
                    items = st.session_state.my_interests[group_name]
                    if items:
                        for item in items:
                            # 기업명을 클릭하면 해당 기업 분석으로 넘어가는 등의 확장 가능
                            st.markdown(f"• **{item}**")
                    else:
                        st.caption("등록된 기업이 없습니다.")
                        st.markdown("<br>" * 2, unsafe_allow_html=True)

    elif menu_choice == "■ 추천 기업":
        st.markdown("### 🎯 맞춤형 추천 기업 탐색")
        
        # --- [1] 상단 필터 영역 (이미지 레이아웃 반영) ---
        with st.container(border=True):
            # 행별로 구분된 필터 구성
            
            # 1. 업종 선택
            f_col1_left, f_col1_right = st.columns([1, 2])
            with f_col1_left:
                st.markdown("<div style='padding:10px; background-color:#e9ecef; font-weight:bold; border-radius:5px;'>업종 선택</div>", unsafe_allow_html=True)
            with f_col1_right:
                industry_choice = st.selectbox(
                    "업종 구분", 
                    ["업종구분없음", "반도체", "디스플레이", "이차전지"], 
                    label_visibility="collapsed"
                )

            # 2. 업종 특성 선호도 (중복 선택 가능)
            f_col2_left, f_col2_right = st.columns([1, 2])
            with f_col2_left:
                st.markdown("<div style='padding:10px; background-color:#e9ecef; font-weight:bold; border-radius:5px;'>업종 특성 선호도</div>", unsafe_allow_html=True)
            with f_col2_right:
                traits = st.multiselect(
                    "특성 선택", 
                    ["안정성", "성장성", "수익성"], 
                    default=["안정성"],
                    label_visibility="collapsed"
                )

            # 3. 기업 규모 선호도 (중복 선택 가능)
            f_col3_left, f_col3_right = st.columns([1, 2])
            with f_col3_left:
                st.markdown("<div style='padding:10px; background-color:#e9ecef; font-weight:bold; border-radius:5px;'>기업 규모 선호도</div>", unsafe_allow_html=True)
            with f_col3_right:
                size_traits = st.multiselect(
                    "규모 선택", 
                    ["매출액 상위 순", "자산 규모 상위 순"], 
                    default=["매출액 상위 순"],
                    label_visibility="collapsed"
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- [2] 결과 표 영역 (Top 10 추천 리스트) ---
        st.markdown(f"#### 🏆 {industry_choice} 분야 Top 10 추천 기업")
        
        # 임시 추천 데이터 (실제 서비스 시에는 필터 조건에 따라 데이터프레임 필터링 로직 추가)
        recommend_data = pd.DataFrame({
            "순위": range(1, 11),
            "기업명": ["한미반도체", "HPSP", "리노공업", "주성엔지니어링", "원익IPS", "티씨케이", "이오테크닉스", "하나마이크론", "파크시스템스", "넥스틴"],
            "매출액": [3200, 1800, 2500, 4100, 9500, 2200, 3100, 8800, 1200, 1100],
            "영업이익": [520, 900, 1100, 350, 150, 800, 450, 200, 320, 480],
            "당기순이익": [410, 720, 950, 280, 110, 650, 380, 150, 260, 400],
            "자산": [8500, 4200, 6800, 7500, 15000, 3800, 5500, 9200, 2100, 1900],
            "부채": [1200, 500, 800, 2100, 6500, 400, 1500, 4800, 300, 250],
            "자본": [7300, 3700, 6000, 5400, 8500, 3400, 4000, 4400, 1800, 1650]
        })

        # 표 디자인 최적화 (가로 스크롤 가능하도록 표시)
        st.dataframe(
            recommend_data.set_index("순위"), 
            use_container_width=True,
            column_config={
                "매출액": st.column_config.NumberColumn(format="%d 억"),
                "영업이익": st.column_config.NumberColumn(format="%d 억"),
                "자산": st.column_config.NumberColumn(format="%d 억")
            }
        )
        
        st.caption("※ 위 리스트는 선택하신 안정성, 성장성, 수익성 지표 및 규모 선호도를 종합 분석한 AI 추천 결과입니다.")

