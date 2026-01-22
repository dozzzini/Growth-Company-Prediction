import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- 1. 초기 세션 상태 설정 ---
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""

# --- 2. 페이지 기본 설정 ---
# 첫 화면과 결과 화면의 레이아웃을 유연하게 처리하기 위해 와이드 모드로 시작
st.set_page_config(page_title="Sesac Enterprise AI", layout="wide")

# --- 3. 화면별 커스텀 CSS 정의 ---
def load_css(is_result_page):
    if not is_result_page:
        # [첫 화면용 CSS] - 기존 Sesac 스타일 유지
        st.markdown("""
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
            .hero-title-main { font-size: 80px; font-weight: 900; color: #5c67f2; margin: 15px 0; }
            div.stButton > button {
                background-color: #5c67f2 !important; color: white !important;
                width: 100%; height: 65px; border-radius: 12px !important;
                font-size: 24px !important; font-weight: 800 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        # [결과 화면용 CSS] - 짙은 회색 사이드바 및 옅은 회색 배경 적용
        st.markdown("""
            <style>
            header {visibility: hidden;}
            .stApp { background-color: #f1f3f6; }
            [data-testid="stSidebar"] { background-color: #343a40 !important; }
            [data-testid="stSidebar"] * { color: white !important; }
            .dashboard-card {
                background-color: white; padding: 20px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
            }
            .brand-side { color: #ffffff; font-size: 24px; font-weight: bold; padding: 20px 0; }
            thead tr th:first-child, tbody th { display: none; }
            </style>
        """, unsafe_allow_html=True)

# --- 4. 메인 로직 분기 ---

if not st.session_state.search_input:
    # ---------------------------------------------------------
    # [Case 1] 첫 화면 디자인 유지
    # ---------------------------------------------------------
    load_css(is_result_page=False)
    
    # 상단 내비게이션 바
    st.markdown("""
        <div class="navbar">
            <div class="brand">Sesac</div>
            <div class="nav-items">
                <a href="#">분석스튜디오</a><a href="#">서비스</a><a href="#">요금제</a>
                <span style="margin-left:30px;">🔔</span><a href="#">회원가입</a>
                <a href="#" class="btn-login">로그인</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 중앙 메인 섹션
    st.markdown("""
        <div class="main-hero">
            <div style="font-size: 32px; font-weight: 700; color: #333;">핵심 기술의 가치를 발견하는 가장 앞선 감각</div>
            <div class="hero-title-main">AI 기반 기술기업 성장 예측 시스템</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        temp_input = st.text_input("기업명 입력", placeholder="분석할 기업명을 입력하세요", label_visibility="collapsed")
        if st.button("분석 시작하기") and temp_input:
            st.session_state.search_input = temp_input
            st.rerun()

else:
    # ---------------------------------------------------------
    # [Case 2] 분석 결과 화면 - 사이드바 기반 레이아웃
    # ---------------------------------------------------------
    load_css(is_result_page=True)
    
    # 사이드바 구성 (짙은 회색)
    with st.sidebar:
        st.markdown('<div class="brand-side">#Sesac Enterprise AI</div>', unsafe_allow_html=True)
        st.markdown("### Menu")
        st.write("■ Overview")
        st.write("■ Order Details")
        st.markdown("---")
        st.markdown("### Controls")
        # Select Year를 사이드바로 이동
        sel_year = st.selectbox("Select Year", [2021, 2022, 2023], index=2)
        st.markdown("---")
        st.markdown("### Connect")
        st.write("🐦 Twitter | 🔗 LinkedIn")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("Created by: Serena Purslow")
        if st.button("← 처음으로 돌아가기"):
            st.session_state.search_input = ""
            st.rerun()

    # 메인 결과 영역
    search_term = st.session_state.search_input
    
    # 상단 타이틀 및 버튼 영역
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(f"## 기업명/종목코드 : {search_term}")
    with t_col2:
        st.markdown('<div style="text-align:right;"><button style="padding:10px 20px; border-radius:8px; border:1px solid #ddd;">📄 리포트 출력</button></div>', unsafe_allow_html=True)

    # [섹션 1] 연도별 성장률 미니 그래프
    m_cols = st.columns(4)
    for i, (yr, val) in enumerate(m_data.items()):
        with m_cols[i]:
            st.markdown(f'<div class="dashboard-card" style="text-align:center;"><small>{yr}</small><br><b style="font-size:22px; color:#5c67f2;">{val}</b></div>', unsafe_allow_html=True)

    # [섹션 2] 매출 추이 및 인사이트
    c_left, c_right = st.columns([2.5, 1])
    with c_left:
        st.markdown('<div class="dashboard-card"><h4>2019~2024년 재무정보</h4>', unsafe_allow_html=True)
        fig = px.area(y=[20, 45, 30, 85, 60, 95], color_discrete_sequence=['#e0e0e0'])
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c_right:
        st.markdown(f"""
            <div class="dashboard-card" style="height:355px;">
                <h4>🤖 GenAI 인사이트</h4>
                <p style="font-size:0.9rem; line-height:1.6;"><b>심사역 종합 의견:</b><br>
                {search_term}는 HBM 공급망 관련 시장 긍정 보도가 증가하고 있으며, 차년도 성장이 유력합니다.</p>
                <hr>
                <small>✅ R&D 비중 15% 초과<br>✅ 특허 기술 영향력 우수</small>
            </div>
        """, unsafe_allow_html=True)

    # [섹션 3] 긍부정 속성 분석 (Top 10 유사기업 위쪽)
    st.markdown("### 📊 긍부정 속성 분석")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<div class="dashboard-card"><b>긍·부정 추이</b>', unsafe_allow_html=True)
        st.plotly_chart(px.bar(x=['1월','2월','3월'], y=[60, 80, 70], height=200), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="dashboard-card"><b>긍·부정 비율</b>', unsafe_allow_html=True)
        st.plotly_chart(px.pie(values=[75, 15, 10], names=['긍정','중립','부정'], hole=0.5, height=200), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="dashboard-card"><b>긍부정 키워드</b><br><div style="text-align:center; padding:20px; color:#5c67f2; font-weight:bold;">최고 기대 매력적 합리적 인기</div></div>', unsafe_allow_html=True)

    # [섹션 4] SHAP 그래프 (하단 배치)
    st.markdown('<div class="dashboard-card"><h4>🧬 모델 예측 근거 (SHAP Force Plot)</h4>', unsafe_allow_html=True)
    # 이미지 파일 image_00e3eb.png와 유사한 시각적 효과를 위해 임의의 데이터 시각화 배치
    shap_data = pd.DataFrame(np.random.randn(100, 2), columns=['SHAP Value', 'Feature Impact'])
    st.plotly_chart(px.scatter(shap_data, x='SHAP Value', y='Feature Impact', color='SHAP Value', color_continuous_scale='RdBu_r', height=300), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # [섹션 5] 유사 기업 테이블 (최하단)
    st.markdown("### 👯 Top 10 Similar Manufacturers")
    similar_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Company": ["한미반도체", "에이치피에스피", "리노공업", "주성엔지니어링", "이오테크닉스", "원익IPS", "티씨케이", "파크시스템스", "피에스케이", "유진테크"],
        "Similarity": [0.98, 0.96, 0.94, 0.91, 0.89, 0.88, 0.87, 0.85, 0.84, 0.82],
        "Growth Score": [88, 85, 92, 80, 75, 71, 78, 95, 68, 74]
    })
    st.table(similar_df)