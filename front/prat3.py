import streamlit as st
import time
import random

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------------------
st.set_page_config(page_title="Yoonseul AI", layout="wide")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 시작 여부 확인
chat_started = len(st.session_state.messages) > 0

# -------------------------------------------------------------------------
# 2. CSS 스타일링
# -------------------------------------------------------------------------
st.markdown(f"""
    <style>
        /* [1] 윤슬 애니메이션 */
        @keyframes shimmer {{
            0% {{background-position: 0% 50%;}}
            50% {{background-position: 100% 50%;}}
            100% {{background-position: 0% 50%;}}
        }}

        /* [2] 메인 영역 상단 여백 */
        .block-container {{
            padding-top: {'2rem' if chat_started else '5rem'} !important;
            padding-bottom: 5rem !important;
        }}

        /* 헤더 투명화 */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            z-index: 10000 !important;
            height: auto;
        }}
        div[data-testid="stDecoration"] {{ display: none; }}

        /* [3] 커스텀 토글 디자인 */
        details.custom-toggle {{
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 12px;
            overflow: hidden;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        details.custom-toggle:hover {{
            border-color: #6a11cb;
            box-shadow: 0 4px 10px rgba(106, 17, 203, 0.1);
        }}
        details.custom-toggle[open] {{ border-color: #6a11cb; }}

        summary.toggle-header {{
            display: flex; align-items: center; padding: 14px 20px;
            cursor: pointer; list-style: none; background-color: white;
        }}
        summary.toggle-header::-webkit-details-marker {{ display: none; }}

        /* [4] 헤더 내부 요소 */
        .header-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            padding: 4px 10px; border-radius: 4px;
            color: white; font-weight: 800; font-size: 10px;
            text-transform: uppercase; margin-right: 12px; min-width: 45px;
        }}
        .badge-pdf {{ background-color: #ff4d4f; }}
        .badge-docx {{ background-color: #1890ff; }}
        .badge-pptx {{ background-color: #fa8c16; }}
        .badge-txt {{ background-color: #8c8c8c; }}
        
        .header-title {{
            font-weight: 600; font-size: 16px; color: #333;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}

        .header-score {{
            margin-left: auto;
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            box-shadow: 0 2px 5px rgba(106, 17, 203, 0.3);
            color: white; padding: 4px 12px; border-radius: 20px;
            font-size: 12px; font-weight: bold; white-space: nowrap;
        }}
        
        .toggle-arrow {{
            margin-left: 10px; font-size: 12px; color: #999; transition: transform 0.2s;
        }}
        details[open] .toggle-arrow {{ transform: rotate(180deg); }}

        /* [5] 내부 콘텐츠 */
        .toggle-content {{
            padding: 0 20px 20px 20px; border-top: 1px solid #f5f5f5;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        .summary-text {{
            font-size: 14px; color: #4a5568; line-height: 1.6;
            background-color: #f8f9fa; padding: 15px;
            border-radius: 8px; margin: 15px 0; border-left: 4px solid #e2e8f0;
        }}

        /* [6] 파일 경로 박스 */
        .path-box {{
            background-color: #262730; color: #00cec9;
            font-family: 'Courier New', monospace; font-size: 13px;
            padding: 12px 15px; border-radius: 6px;
            word-break: break-all; user-select: all; cursor: copy;
        }}
        .path-label {{
            font-size: 12px; font-weight: bold; color: #6a11cb;
            margin-bottom: 5px; display: block;
        }}
        
        /* [7] 배너 및 타이틀 */
        .animated-banner {{
            text-align: center; padding: 15px; color: white;
            font-weight: bold; font-size: 24px; margin-bottom: 20px;
            border-radius: 0 0 10px 10px;
            background: linear-gradient(90deg, #6a11cb, #2575fc, #6a11cb);
            background-size: 200% 200%;
            animation: shimmer 8s ease infinite;
        }}
        .intro-container {{
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 50vh; text-align: center; color: #444; margin-top: 30px;
        }}
        .intro-text {{
            font-size: 3.5rem; font-weight: 900; margin-bottom: 10px;
            background: linear-gradient(135deg, #6a11cb 20%, #2575fc 80%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .intro-sub {{ font-size: 1.1rem; color: #666; }}

        /* [8] 로딩 상태 위젯 색상 커스텀 (보라색 테마 적용) */
        div[data-testid="stStatusWidget"] {{
            border: 1px solid #e0e0e0;
            background-color: #fcfcfc;
        }}
        div[data-testid="stStatusWidget"] > div > svg {{
            fill: #6a11cb !important; /* 아이콘 색상 */
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 3. 사이드바 구성
# -------------------------------------------------------------------------
if chat_started:
    st.markdown('<div class="animated-banner">YOONSEUL AI</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("설정 및 필터")
    st.subheader("검색 옵션")
    sort_option = st.radio("정렬 기준", ["정확도순", "최신 날짜순"], horizontal=True)
    
    # [설정 1] 문서 개수 슬라이더 (Top K)
    top_k = st.slider("참고할 문서 개수 (Top K)", min_value=1, max_value=20, value=3)
    
    st.divider()
    
    st.subheader("상세 필터") 
    date_range = st.date_input("기간 선택", [])
    
    # [설정 2] 파일 확장자 필터
    file_types = st.multiselect(
        "파일 확장자", 
        ["PDF", "DOCX", "PPTX", "TXT"], 
        default=["PDF", "DOCX", "PPTX"]
    )

# -------------------------------------------------------------------------
# 4. 메인 로직
# -------------------------------------------------------------------------

if not chat_started:
    with st.empty().container():
        st.markdown("""
            <div class="intro-container">
                <div class="intro-text">Hello, Yoonseul AI</div>
                <div class="intro-sub">사내 문서 기반 지능형 검색 에이전트</div>
            </div>
        """, unsafe_allow_html=True)

# 이전 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# AI 응답 처리 (로딩 애니메이션 추가됨)
if chat_started and st.session_state.messages[-1]["role"] == "user":
    
    with st.chat_message("assistant"):
        
        # ---------------------------------------------------------
        # [NEW] 단계별 로딩 상태 표시 (st.status)
        # ---------------------------------------------------------
        with st.status("윤슬 AI가 답변을 생성하고 있습니다...", expanded=True) as status:
            st.write("📂 사내 문서 저장소 연결 및 검색 중...")
            time.sleep(1.0) # 실제 검색 시간 시뮬레이션
            
            st.write("🔍 관련도 분석 및 메타데이터 필터링...")
            time.sleep(0.8) # 필터링 시간 시뮬레이션
            
            st.write("✨ LLM 답변 요약 및 카드 생성 중...")
            time.sleep(0.5) # 생성 시간 시뮬레이션
            
            # 모든 작업 완료 시 상태 업데이트 (접힘)
            status.update(label="검색 및 분석 완료!", state="complete", expanded=False)

        # ---------------------------------------------------------
        # 기존 응답 출력 로직
        # ---------------------------------------------------------
        message_placeholder = st.empty()
        full_text = "네, 요청하신 내용을 바탕으로 관련 문서를 찾았습니다. 설정하신 필터 조건에 맞는 문서는 다음과 같습니다."
        
        # 타이핑 효과
        msg_buffer = ""
        for char in full_text:
            msg_buffer += char
            time.sleep(0.01)
            message_placeholder.markdown(msg_buffer + "▌")
        message_placeholder.markdown(full_text)
        
        # ---------------------------------------------------------
        # 더미 데이터 생성 및 필터링 적용
        # ---------------------------------------------------------
        
        # 1. 충분한 양의 가짜 데이터 생성 (20개)
        full_dummy_data = []
        types_pool = ["pdf", "docx", "pptx", "txt"]
        
        for i in range(1, 21):
            # 랜덤하게 타입 배정
            ftype = types_pool[i % 4] 
            full_dummy_data.append({
                "name": f"202{i%5}_프로젝트_문서_{i:02d}.{ftype}",
                "type": ftype,
                "score": 0.99 - (i * 0.02), # 점수 차등
                "summary": f"이 문서는 검색된 {i}번째 가상 문서입니다. '{ftype.upper()}' 포맷이며, 시각화 테스트를 위해 생성되었습니다.",
                "path": f"/Server/Docs/Project/Doc_{i:02d}.{ftype}"
            })

        # 2. 확장자 필터링 (사용자가 선택한 것만 남김)
        selected_extensions = [ext.lower() for ext in file_types]
        filtered_files = [f for f in full_dummy_data if f['type'] in selected_extensions]

        # 3. 개수 자르기 (Top K 만큼)
        final_files = filtered_files[:top_k]

        # ---------------------------------------------------------
        # [화면 렌더링] 필터링된 결과만 보여주기
        # ---------------------------------------------------------
        st.markdown(f"<br><h3>참고 문서 ({len(final_files)}건)</h3>", unsafe_allow_html=True)
        
        html_content = ""
        
        if not final_files:
             st.warning("선택하신 조건에 맞는 문서가 없습니다. 필터를 변경해 보세요.")
        else:
            for file in final_files:
                badge_class = f"badge-{file['type']}"
                score_pct = int(file['score'] * 100)
                
                card_html = f"""
                <details class="custom-toggle">
                    <summary class="toggle-header">
                        <span class="header-badge {badge_class}">{file['type'].upper()}</span>
                        <span class="header-title">{file['name']}</span>
                        <span class="header-score">{score_pct}% 일치</span>
                        <span class="toggle-arrow">▼</span>
                    </summary>
                    <div class="toggle-content">
                        <div class="summary-text">
                            <strong><h6> 문서 요약</h6></strong><br>
                            {file['summary']}
                        </div>
                        <span class="path-label">파일 경로 (클릭하여 전체 선택 후 Ctrl+C)</span>
                        <div class="path-box" title="클릭하면 전체가 선택됩니다.">{file['path']}</div>
                    </div>
                </details>
                """
                html_content += card_html
                
            st.markdown(html_content, unsafe_allow_html=True)

    # 대화 기록 저장
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_text
    })