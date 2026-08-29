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
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except FileNotFoundError:
    st.warning("CS file not found. Styling might be missing.")

# --- Font Configuration ---
visualization.configure_fonts()

# --- Authentication Gate ---
# 임상 데이터를 다루므로 로그인하지 않으면 여기서 실행이 중단된다.
auth.require_login()

# --- Sidebar ---
with st.sidebar:
    auth.render_user_box()
    st.image("https://via.placeholder.com/150x50?text=N-Map", use_container_width=True) # Placeholder Logo
    st.title("N-Map 분석 도구")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("임상 데이터 업로드 (xlsx/csv)", type=['xlsx', 'csv'])
    
    st.markdown("### 분석 파라미터 설정")
    min_support = st.slider("최소 지지도 (Min Support)", 0.01, 0.5, 0.05, 0.01, help="아이템셋이 등장하는 최소 빈도 비율입니다.")
    min_confidence = st.slider("최소 신뢰도 (Min Confidence)", 0.1, 1.0, 0.3, 0.05, help="규칙의 신뢰성(A이면 B이다)을 나타냅니다.")
    min_lift = st.number_input("최소 향상도 (Min Lift)", 1.0, 10.0, 1.0, 0.1, help="연관성의 강도를 나타내며 1보다 커야 유의미합니다.")
    
    st.markdown("---")
    st.info("지원 컬럼: 연령, 수술시간, 간호중재")

# --- Main Content ---
st.title("🏥 N-Map: 간호 연관성 분석 및 시각화")
st.markdown("임상 간호 데이터 속의 숨겨진 패턴을 찾아 **근거 기반 간호(EBN)**를 위한 통찰력을 제공합니다.")

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
                            display_rules.head(10).style.highlight_max(axis=0, color='#d1e7dd'),
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
                            - **굵기(Width)**: 해당 경로에 속하는 **환자의 수(빈도)**를 의미합니다. 굵을수록 해당 케이스가 많다는 뜻입니다.
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
    # Landing Page State
    st.info("👈 사이드바에서 임상 데이터 파일(Excel/CSV)을 업로드하여 분석을 시작하세요.")
    st.markdown("""
    ### 권장 데이터 형식 (두 가지 모두 지원)
    
    **Type A: 기본 형식**
    * **연령**: 환자 나이
    * **수술시간**: 분 단위 숫자
    * **간호중재**: "중재1, 중재2" 형태의 문자열

    ### 📘 N-Map 사용 및 해석 가이드
    
    **1. 데이터 업로드**
    좌측 사이드바에서 엑셀(.xlsx) 또는 CSV 파일을 업로드하세요.
    
    **2. 파라미터 이해하기**
    * **지지도 (Support)**: 해당 패턴이 전체 데이터에서 얼마나 자주 등장하는지 (높을수록 흔한 패턴)
    * **신뢰도 (Confidence)**: A가 발생했을 때 B가 발생할 확률 (높을수록 믿을 수 있는 규칙)
    * **향상도 (Lift)**: A와 B가 우연히 같이 일어난 것보다 얼마나 더 밀접한지 (1보다 크면 양의 상관관계)

    **3. 시각화 해석**
    * **네트워크 그래프**: 간호중재 간의 복잡한 연결 관계를 파악합니다.
    * **Sankey 다이어그램**: 환자 특성에 따른 간호중재의 흐름을 봅니다.
    * **히트맵**: 수술 시간대별로 자주 하는 간호중재를 색상으로 비교합니다.
    """) 
