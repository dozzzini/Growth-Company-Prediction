import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os

# --- 1) 페이지 기본 설정 (반드시 Streamlit 호출 중 가장 먼저) ---
st.set_page_config(page_title="Sesac Enterprise AI", layout="wide")

# --- 2) 초기 세션 상태 설정 ---
if "search_input" not in st.session_state:
    st.session_state.search_input = ""

# --- 3) 화면별 커스텀 CSS 정의 ---
def load_css(is_result_page: bool) -> None:
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

            /* 상단 네비게이션 바 */
            .navbar {
                display: flex; justify-content: space-between; align-items: center;
                padding: 15px 60px; background: rgba(255, 255, 255, 0.95);
                border-bottom: 1px solid #f0f0f0; position: fixed; top: 0; left: 0; right: 0; z-index: 999;
            }
            .brand { color: #5c67f2; font-size: 26px; font-weight: bold; text-decoration: none; }
            .nav-items a { color: #333; text-decoration: none; margin-left: 30px; font-size: 15px; }
            .btn-login { background-color: #5c67f2 !important; color: white !important; padding: 8px 18px; border-radius: 6px; }

            /* 히어로 영역 */
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
            .stApp { background-color: #f1f3f6; }

            /* 사이드바 */
            [data-testid="stSidebar"] { background-color: #a1b1bf !important; }
            [data-testid="stSidebar"] * { color: #2b2b2b !important; }

            /* 사이드바 폰트 크기 */
            [data-testid="stWidgetLabel"] p { font-size: 1.1rem !important; font-weight: bold !important; }
            div[data-baseweb="select"] > div { font-size: 1.0rem !important; }

            .dashboard-card {
                background-color: white; padding: 20px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
            }
            .brand-side { color: #2b2b2b; font-size: 24px; font-weight: bold; padding: 20px 0; }

            .kpi-card {
                background-color: white; padding: 22px 18px; border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
                text-align: center; min-height: 120px; display: flex;
                flex-direction: column; justify-content: center;
            }
            .kpi-year { font-size: 13px; color: #666; margin-bottom: 10px; }
            .kpi-percent { font-size: 26px; font-weight: 900; color: #5c67f2; margin: 0; line-height: 1.1; }
            </style>
            """,
            unsafe_allow_html=True,
        )

def gauge_figure(value: float, title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": "", "font": {"size": 26}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e0e0e0"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(92,103,242,0.15)"},
                    {"range": [40, 70], "color": "rgba(92,103,242,0.25)"},
                    {"range": [70, 100], "color": "rgba(92,103,242,0.35)"},
                ],
            },
        )
    )
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=35, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# -------- WordCloud helpers (환경별 폰트 에러 방지) --------
def _pick_korean_font_path() -> str | None:
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",                          # Windows
        "/System/Library/Fonts/AppleGothic.ttf",               # macOS
        "/Library/Fonts/AppleGothic.ttf",                      # macOS alt
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",     # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def build_wordcloud(text: str):
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    font_path = _pick_korean_font_path()

    # font_path가 None이면 WordCloud가 한글을 깨먹을 수 있으나, 최소한 앱이 죽지는 않게 처리
    wc = WordCloud(
        font_path=font_path,
        width=900,
        height=520,
        background_color="white",
        collocations=False,
        prefer_horizontal=0.9,
    ).generate(cleaned if cleaned else " ")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# --- 4) 메인 로직 분기 ---
if not st.session_state.search_input:
    load_css(is_result_page=False)

    st.markdown(
        """
        <div class="navbar">
            <div class="brand">Sesac</div>
            <div class="nav-items">
                <a href="#">분석스튜디오</a><a href="#">서비스</a><a href="#">요금제</a>
                <span style="margin-left:30px;">🔔</span><a href="#">회원가입</a>
                <a href="#" class="btn-login">로그인</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-hero">
            <div class="hero-subtitle">핵심 기술의 가치를 발견하는 가장 앞선 감각</div>
            <div class="hero-title-main">AI 기반 기술기업 성장 예측</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        temp_input = st.text_input("기업명 입력", placeholder="분석할 기업명을 입력하세요", label_visibility="collapsed")
        if st.button("분석 시작하기") and temp_input.strip():
            st.session_state.search_input = temp_input.strip()
            st.rerun()

else:
    load_css(is_result_page=True)

    # ✅ 사이드바
    with st.sidebar:
        st.markdown('<div class="brand-side">Sesac Enterprise AI</div>', unsafe_allow_html=True)
        st.markdown("### Menu")
        st.write("■ 기업검색")
        st.write("■ 성장지수 예측")
        st.write("■ XAI_SHAP")
        st.write("■ GenAI 보고서")
        st.write("■ 유사기업")
        st.markdown("---")
        st.markdown("### Controls")
        sel_year = st.selectbox("Select Year", [2021, 2022, 2023, 2024], index=3)
        st.markdown("---")
        st.markdown("### Connect")
        st.write("Twitter | LinkedIn")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("Created by: Serena Purslow")
        if st.button("← 처음으로 돌아가기"):
            st.session_state.search_input = ""
            st.rerun()

    # ✅ 메인
    search_term = st.session_state.search_input
    stock_code = "000000"
    st.markdown(f"<h2 style='text-align:center;'>{search_term} / {stock_code}</h2>", unsafe_allow_html=True)

    growth_pct = {2021: 12.3, 2022: 18.7, 2023: 25.4, 2024: 31.9}

    kpi_cols = st.columns(4)
    for i, yr in enumerate([2021, 2022, 2023, 2024]):
        with kpi_cols[i]:
            pct = growth_pct.get(yr, 0.0)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-year">{yr}</div>
                    <div class="kpi-percent">{pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    main_left, main_right = st.columns([2.2, 1.5], gap="large")

    with main_left:
        c_left, c_mid = st.columns([2.3, 1.0], gap="large")

        with c_left:
            st.markdown('<div class="dashboard-card"><h4>2019~2024년 재무정보</h4>', unsafe_allow_html=True)
            years = list(range(2019, 2025))
            values = [820, 900, 880, 980, 1030, 1100]
            fig_bar = px.bar(x=years, y=values)
            fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_mid:
            st.markdown('<div class="dashboard-card"><h4>성장지수</h4>', unsafe_allow_html=True)
            st.plotly_chart(gauge_figure(73, f"{search_term}"), use_container_width=True)
            st.plotly_chart(gauge_figure(62, "업종평균"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ✅ (중요) main_right는 결과 페이지에서만 존재하므로, 아래도 else 블록 안에 둔다
    with main_right:
        st.markdown('<div class="dashboard-card" style="min-height: 520px;"><h3>GenAI 인사이트</h3>', unsafe_allow_html=True)

        insight_text = """
        분석된 기업의 특허 경쟁력이 매우 높습니다.
        반도체 공정 장비 분야에서 기술 장벽이 강합니다.
        고객사 다변화와 신규 수주가 기대됩니다.
        원가 구조 개선과 고부가 제품 믹스가 긍정적입니다.
        """
        fig_wc = build_wordcloud(insight_text)
        st.pyplot(fig_wc, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📊 긍부정 속성 분석")
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown('<div class="dashboard-card"><b>긍·부정 추이</b>', unsafe_allow_html=True)
            st.plotly_chart(px.bar(x=["1월", "2월", "3월"], y=[60, 80, 70], height=200), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="dashboard-card"><b>긍·부정 비율</b>', unsafe_allow_html=True)
            st.plotly_chart(px.pie(values=[75, 15, 10], names=["긍정", "중립", "부정"], hole=0.5, height=200), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with s3:
            st.markdown(
                """
                <div class="dashboard-card">
                    <b>긍부정 키워드</b><br>
                    <div style="text-align:center; padding:20px; color:#5c67f2; font-weight:bold;">
                        최고 기대 매력적 합리적 인기
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="dashboard-card"><h4>🧬 모델 예측 근거 (SHAP Force Plot)</h4>', unsafe_allow_html=True)
        shap_data = pd.DataFrame(np.random.randn(100, 2), columns=["SHAP Value", "Feature Impact"])
        st.plotly_chart(
            px.scatter(
                shap_data,
                x="SHAP Value",
                y="Feature Impact",
                color="SHAP Value",
                color_continuous_scale="RdBu_r",
                height=300,
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 👯 Top 10 Similar Manufacturers")
        similar_df = pd.DataFrame(
            {
                "Rank": range(1, 11),
                "Company": ["한미반도체", "에이치피에스피", "리노공업", "주성엔지니어링", "이오테크닉스", "원익IPS", "티씨케이", "파크시스템스", "피에스케이", "유진테크"],
                "Similarity": [0.98, 0.96, 0.94, 0.91, 0.89, 0.88, 0.87, 0.85, 0.84, 0.82],
                "Growth Score": [88, 85, 92, 80, 75, 71, 78, 95, 68, 74],
            }
        )
        st.table(similar_df)

