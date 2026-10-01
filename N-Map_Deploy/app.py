import streamlit as st
import pandas as pd
import os
import tempfile
import streamlit.components.v1 as components
from modules import preprocessing, mining, visualization, auth

# --- Page Configuration ---
st.set_page_config(
    page_title="N-Map: Nursing Association Mining",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"  # expanded on desktop, collapsed on phones
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except FileNotFoundError:
    st.warning("assets/style.css 를 찾을 수 없어 기본 스타일로 표시됩니다.")

# --- Font Configuration ---
visualization.configure_fonts()

# --- Authentication Gate ---
# 임상 데이터를 다루므로 로그인하지 않으면 여기서 실행이 중단된다.
auth.require_login()

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        '<div class="nmap-wordmark"><span class="dot"></span>N-Map</div>',
        unsafe_allow_html=True,
    )
    auth.render_user_box()
    st.markdown("---")
    
    uploaded_file = st.file_uploader("임상 데이터 업로드 (xlsx/csv)", type=['xlsx', 'csv'])
    
    st.markdown("### 분석 파라미터 설정")
    min_support = st.slider("최소 지지도 (Min Support)", 0.01, 0.5, 0.05, 0.01, help="아이템셋이 등장하는 최소 빈도 비율입니다.")
    min_confidence = st.slider("최소 신뢰도 (Min Confidence)", 0.1, 1.0, 0.3, 0.05, help="규칙의 신뢰성(A이면 B이다)을 나타냅니다.")
    min_lift = st.number_input("최소 향상도 (Min Lift)", 1.0, 10.0, 1.0, 0.1, help="연관성의 강도를 나타내며 1보다 커야 유의미합니다.")
    
    st.markdown("---")
    st.info("지원 컬럼: 연령, 수술시간, 간호중재")

# --- Main Content ---
st.markdown(
    """
    <div class="nmap-hero">
        <span class="nmap-eyebrow">Nursing Association Mining</span>
        <h1>임상 데이터 속<br>간호의 인과를 읽다</h1>
        <p class="nmap-sub">
            N-Map 은 연령 &middot; 수술시간 &middot; 간호중재의 연관 규칙을 찾아
            근거 기반 간호(EBN)를 위한 통찰로 바꿉니다.
            파일을 올리면 지지도 &middot; 신뢰도 &middot; 향상도까지 한 번에 계산됩니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    # 1. Load & Preprocess Data
    with st.spinner("데이터 전처리 중..."):
        raw_df = preprocessing.load_data(uploaded_file)
        
        if raw_df is not None:
            try:
                processed_df = preprocessing.preprocess_data(raw_df)
                
                # Show Data Overview
                col1, col2, col3 = st.columns(3)
                col1.metric("총 데이터 수", len(raw_df))
                col2.metric("처리된 트랜잭션", len(processed_df))
                col3.metric("고유 간호중재 수", processed_df['간호중재'].nunique() if '간호중재' in processed_df.columns else 0)
                
                with st.expander("📄 처리된 데이터 미리보기"):
                    st.dataframe(processed_df.head(), use_container_width=True)
                    
                # 2. Association Rule Mining
                st.subheader("🔍 연관 규칙 마이닝 (Association Rule Mining)")
                
                # Prepare transactions
                # We want to associate Age, Surgery Time, and Interventions
                cols_to_mine = ['연령대', '수술시간_범주', '간호중재']
                cols_present = [c for c in cols_to_mine if c in processed_df.columns]
                
                if len(cols_present) >= 2:
                    transactions = preprocessing.prepare_transaction_matrix(processed_df, cols_present)
                    
                    with st.spinner("연관 규칙 분석 중..."):
                        rules = mining.run_apriori_analysis(transactions, min_support, min_confidence, min_lift)
                    
                    if not rules.empty:
                        st.success(f"총 {len(rules)}개의 연관 규칙을 발견했습니다.")
                        
                        # Display Rules Table
                        display_rules = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
                        display_rules.columns = ['조건 (Antecedents)', '결과 (Consequents)', '지지도 (Support)', '신뢰도 (Confidence)', '향상도 (Lift)']
                        
                        st.dataframe(
                            display_rules.head(10).style.highlight_max(axis=0, color='#eeebff'),
                            use_container_width=True
                        )
                        
                        # Download Rules
                        csv_rules = rules.to_csv(index=False).encode('utf-8-sig') # BOM for Excel
                        st.download_button("연관 규칙 CSV 다운로드", csv_rules, "nmap_rules.csv", "text/csv")
                        
                        # 3. Visualizations
                        tab1, tab2, tab3 = st.tabs(["🕸️ 네트워크 그래프", "🌊 환자 흐름 분석 (Sankey)", "🔥 히트맵 분석"])
                        
                        with tab1:
                            st.markdown("#### 속성 간 의존성 네트워크")
                            net = visualization.create_network_graph(rules)
                            if net:
                                # Save to tmp file to render
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                                    net.save_graph(tmp.name)
                                    with open(tmp.name, 'r', encoding='utf-8') as f:
                                        html_string = f.read()
                                    components.html(html_string, height=750, scrolling=True)
                                os.unlink(tmp.name)
                            
                            st.info("""
                            **💡 그래프 해석 가이드**
                            - **점(Node)**: 각각의 간호중재, 연령대, 수술시간을 나타냅니다.
                            - **선(Edge)**: 두 항목 간의 연관성을 나타내며, **선이 굵을수록 연관성(Lift)이 강함**을 의미합니다.
                            """)
                                
                        with tab2:
                            st.markdown("#### 환자 특성 및 중재 흐름 (Sankey Diagram)")
                            fig_sankey = visualization.create_sankey_diagram(processed_df)
                            if fig_sankey:
                                st.plotly_chart(fig_sankey, use_container_width=True)
                            else:
                                st.warning("흐름 분석을 위한 데이터 컬럼이 부족합니다.")
                            
                            st.info("""
                            **💡 그래프 해석 가이드**
                            - **흐름(Flow)**: 왼쪽에서 오른쪽으로 이어지는 환자의 특성(연령 → 수술시간 → 간호중재)을 보여줍니다.
                            - **굵기(Width)**: 해당 경로에 속하는 **환자의 수**(빈도)를 의미합니다. 굵을수록 해당 케이스가 많다는 뜻입니다.
                            """)
                                
                        with tab3:
                            st.markdown("#### 수술 종류별 중재 빈도 히트맵")
                            fig_heatmap = visualization.create_heatmap(processed_df)
                            if fig_heatmap:
                                # Updated to use plotly_chart
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            else:
                                st.warning("'수술시간_범주'와 '간호중재' 컬럼이 필요합니다.")
                            
                            st.info("""
                            **💡 그래프 해석 가이드**
                            - **색상(Color)**: **색이 진할수록** 해당 수술 시간대(세로축)에서 그 간호중재(가로축)가 **자주 시행됨**을 의미합니다.
                            - 특정 수술군에서 집중적으로 수행되는 간호 활동을 한눈에 파악할 수 있습니다.
                            """)
                                
                    else:
                        st.warning("설정된 임계값 조건에 맞는 규칙을 찾지 못했습니다. 지지도(Support)나 신뢰도(Confidence)를 낮춰보세요.")
                else:
                    st.error("마이닝을 위한 컬럼이 부족합니다. 입력 파일을 확인해주세요.")
            
            except ValueError as e:
                st.error(f"데이터 처리 오류: {e}")
                st.markdown("### 📋 파일에 포함된 컬럼:")
                st.write(list(raw_df.columns))
                st.warning("엑셀/CSV 파일에 **'연령'**, **'수술시간'**, **'간호중재'** 컬럼이 정확히 포함되어 있는지 확인해주세요.")
                
        else:
            st.error("파일 로드 실패. 형식을 확인해주세요.")

else:
    # --- Landing Page State ------------------------------------------------
    st.markdown(
        """
        <div class="nmap-grid">
            <div class="nmap-card">
                <div class="idx">01</div>
                <h4>업로드</h4>
                <p>사이드바에서 엑셀(.xlsx) 또는 CSV 파일을 올립니다.
                   <strong>연령</strong>, <strong>수술시간</strong>,
                   <strong>간호중재</strong> 컬럼만 있으면 됩니다.</p>
            </div>
            <div class="nmap-card">
                <div class="idx">02</div>
                <h4>파라미터</h4>
                <p>지지도 &middot; 신뢰도 &middot; 향상도 임계값을 조절해
                   찾아낼 규칙의 범위를 정합니다.</p>
            </div>
            <div class="nmap-card">
                <div class="idx">03</div>
                <h4>해석</h4>
                <p>네트워크 &middot; Sankey &middot; 히트맵 세 가지 관점으로
                   같은 규칙을 교차 확인합니다.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("👈 사이드바에서 임상 데이터 파일(Excel/CSV)을 업로드하여 분석을 시작하세요.")

    st.markdown(
        """
        <div class="nmap-dark">
            <h3>파라미터가 뜻하는 것</h3>
            <p><strong>지지도 (Support)</strong> &mdash;
               해당 패턴이 전체 데이터에서 얼마나 자주 등장하는지.
               높을수록 흔한 패턴입니다.</p>
            <p><strong>신뢰도 (Confidence)</strong> &mdash;
               A가 발생했을 때 B가 발생할 확률.
               높을수록 믿을 수 있는 규칙입니다.</p>
            <p><strong>향상도 (Lift)</strong> &mdash;
               A와 B가 우연히 같이 일어난 것보다 얼마나 더 밀접한지.
               <code>1</code> 보다 크면 양의 상관관계입니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 시각화 읽는 법")

    st.markdown(
        """
        <div class="nmap-grid">
            <div class="nmap-card">
                <h4>네트워크 그래프</h4>
                <p>간호중재 사이의 연결 관계를 봅니다.
                   선이 굵을수록 연관성(Lift)이 강합니다.</p>
            </div>
            <div class="nmap-card">
                <h4>Sankey 다이어그램</h4>
                <p>연령 &rarr; 수술시간 &rarr; 간호중재로 이어지는
                   환자의 흐름을 따라갑니다. 굵기는 환자 수입니다.</p>
            </div>
            <div class="nmap-card">
                <h4>히트맵</h4>
                <p>수술 시간대별로 자주 시행되는 중재를
                   색의 농도로 비교합니다.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 권장 데이터 형식")
    st.markdown(
        """
        두 가지 형식을 모두 지원합니다. 같은 의미의 컬럼이면 섞어 써도 됩니다.

        | 항목 | Type A — 기본 | Type B — 펼친 형식 |
        | --- | --- | --- |
        | 연령 | `연령` (숫자) | `나이` · `Age` 도 인식 · 없어도 분석 가능 |
        | 수술시간 | `수술시간` (분 단위 숫자) | `절개시간` + `봉합시간` (시각) → 자동 계산 |
        | 간호중재 | `간호중재` — `통증관리, 체위변경` 처럼 쉼표로 구분 | `간호중재1`, `간호중재2`, … 열마다 하나씩 |
        """
    )
