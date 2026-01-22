import streamlit as st
import time
import sys
from pathlib import Path
from datetime import datetime, time as dt_time

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------------------
st.set_page_config(page_title="Yoonseul AI", layout="wide")

# -------------------------------------------------------------------------
# 2. 경로 설정 및 파이프라인 로드
# -------------------------------------------------------------------------
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import AIAgentPipeline

@st.cache_resource(show_spinner=False)
def load_pipeline():
    config_path = project_root / "config" / "config.yaml"
    return AIAgentPipeline(str(config_path))

# -------------------------------------------------------------------------
# 3. 세션 상태 초기화
# -------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

# 채팅 시작 여부 확인
chat_started = len(st.session_state.messages) > 0

# -------------------------------------------------------------------------
# 4. CSS 스타일링
# -------------------------------------------------------------------------
st.markdown(f"""
    <style>
       /* 사이드바 배경 및 폰트 */
        [data-testid="stSidebar"] {{
            background-color: #fcfcfc;
        }}

        /* 사이드바 제목/소제목 스타일 */
        [data-testid="stSidebar"] h1 {{
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            color: #1a202c !important;
            margin-bottom: 2rem !important;
        }}
        
        [data-testid="stSidebar"] h3 {{
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            color: #2d3748 !important;
            margin-top: 2.5rem !important;
            margin-bottom: 30px !important;
            border-bottom: none !important;
            padding-bottom: 0 !important;
        }}

        /* 위젯 라벨: 진하게 */
        div[data-testid="stWidgetLabel"] {{
            font-weight: 600 !important;
        }}
        
        /* 도움말 아이콘을 라벨 텍스트 바로 옆에 배치 */
        div[data-testid="stWidgetLabel"] > div {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
        }}
        
        /* 셀렉박스 컨테이너 전체 구조 조정 */
        div[data-testid="stSelectbox"] {{
            position: relative !important;
        }}
        
        /* 셀렉박스의 label 요소를 flex로 */
        div[data-testid="stSelectbox"] label[data-testid="stWidgetLabel"] {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            width: auto !important;
        }}
        
        /* 셀렉박스 도움말 아이콘 강제 재배치 */
        div[data-testid="stSelectbox"] div[data-testid="stTooltipIcon"] {{
            position: relative !important;
            right: auto !important;
            top: auto !important;
            transform: none !important;
            display: inline-block !important;
            margin-left: 0 !important;
        }}
        
        /* 셀렉박스 내 커서 제거 및 타이핑 무력화 */
        div[data-testid="stSelectbox"] input {{
            caret-color: transparent !important;
            cursor: pointer !important;
        }}
        
        /* 셀렉박스 라벨과 박스 사이 간격 확보 */
        div[data-testid="stSelectbox"] label {{
            margin-bottom: 15px !important; 
        }}
        
         /* 슬라이더 라벨과 트랙 사이 간격 확보 */
        div[data-testid="stSlider"] label {{
            margin-bottom: 20px !important;
        }}

        /* 나머지 위젯 간격 */
        div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            margin-top: 8px !important;
            margin-bottom: 8px !important;
        }}

        /* 메인 영역 스타일 */
        .animated-banner {{
            text-align: center; padding: 15px; color: white; font-weight: bold; font-size: 24px;
            background: linear-gradient(90deg, #6a11cb, #2575fc); border-radius: 0 0 10px 10px;
        }}
        
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
            color: white !important; font-weight: 800; font-size: 10px;
            text-transform: uppercase; margin-right: 12px; min-width: 45px;
        }}
        .badge-pdf {{ background-color: #ff4d4f; }}
        .badge-docx {{ background-color: #1890ff; }}
        .badge-pptx {{ background-color: #fa8c16; }}
        .badge-txt {{ background-color: #8c8c8c; }}
        .badge-xlsx {{ background-color: #27ae60; }}
        .badge-image {{ background-color: #9b59b6; }}
        
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

        /* [8] 로딩 상태 위젯 색상 커스텀 */
        div[data-testid="stStatusWidget"] {{
            border: 1px solid #e0e0e0;
            background-color: #fcfcfc;
        }}
        div[data-testid="stStatusWidget"] > div > svg {{
            fill: #6a11cb !important;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 5. 사이드바 구성 (첫 번째 코드 디자인 유지)
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("설정 및 필터")
    
    st.subheader("기능 설정")
    use_summary = st.checkbox("문서 요약 사용", value=True, help="검색된 문서의 핵심 내용을 요약합니다.")
    
    st.divider()
    
    st.subheader("검색 옵션")
    sort_option = st.radio("정렬 기준", ["관련도순", "최신순", "오래된순", "파일명순"], index=0)
    sort_map = {
        "관련도순": "relevance",
        "최신순": "date_desc",
        "오래된순": "date_asc",
        "파일명순": "name"
    }
    
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
    top_k = st.slider("참고할 문서 개수 (Top K)", min_value=1, max_value=20, value=5)
    
    st.divider()
    
    st.subheader("상세 필터") 
    
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    
    selected_ext = st.selectbox(
        "파일 확장자",
        ["전체", "pdf", "docx", "pptx", "xlsx", "txt", "image"],
        index=0,
        help="특정 파일 타입만 검색합니다."
    )

# -------------------------------------------------------------------------
# 6. 상단 배너 표시 (채팅 시작 시 출력)
# -------------------------------------------------------------------------
if chat_started:
    st.markdown('<div class="animated-banner">YOONSEUL AI</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 7. 메인 로직 - 인트로 화면
# -------------------------------------------------------------------------
if not chat_started:
    st.markdown("""
        <div class="intro-container">
            <div class="intro-text">Hello, Yoonseul AI</div>
            <div class="intro-sub">사내 문서 기반 지능형 검색 에이전트</div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 8. 이전 대화 기록 출력 (최종: 깔끔한 % 표시 + 자동 접기)
# -------------------------------------------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

        # 메시지에 저장된 'results'가 있으면 카드를 그려줍니다.
        if "results" in msg and msg["results"]:
            st.markdown(f"<br><h3>참고 문서 ({len(msg['results'])}건)</h3>", unsafe_allow_html=True)

            # 가장 최신 메시지만 펼치기
            is_last_message = (idx == len(st.session_state.messages) - 1)
            open_attr = "open" if is_last_message else ""
            
            # [상대 평가 기준점 잡기]
            # 검색된 문서 중 가장 높은 점수를 찾습니다.
            max_score_in_list = 0
            if msg["results"]:
                max_score_in_list = max([r.get("score", 0) for r in msg["results"]])

            for f in msg["results"]:
                meta = f.get("metadata", {})
                ftype = str(meta.get("file_type", "pdf")).lower()
                raw_score = f.get("score", 0)
                
                # [점수 계산]
                # 가장 높은 점수(1등)를 99%로 설정하고, 나머지는 비율대로 계산합니다.
                if max_score_in_list > 0:
                    score_pct = int((raw_score / max_score_in_list) * 99)
                else:
                    score_pct = 0 

                # 미리보기/요약 내용 결정
                summary_content = f.get("summary", "")
                if not summary_content or summary_content == "요약 미사용":
                    raw_text = f.get("text") or meta.get("text") or ""
                    summary_content = f"{raw_text[:300]}..." if raw_text else "내용 없음"

                badge_class = f"badge-{ftype}"

                # [최종 출력] 군더더기 없이 깔끔하게 %만 보여줍니다.
                st.markdown(f"""
                <details class="custom-toggle" {open_attr}>
                    <summary class="toggle-header">
                        <span class="header-badge {badge_class}">{ftype.upper()}</span>
                        <span class="header-title">{meta.get("file_name", "알 수 없는 파일")}</span>
                        <span class="header-score">{score_pct}% 일치</span>
                        <span class="toggle-arrow">▼</span>
                    </summary>
                    <div class="toggle-content">
                        <div class="summary-text">
                            <strong style="font-size: 16px;">문서 내용 ({'요약' if use_summary else '미리보기'})</strong><br>
                            {summary_content}
                        </div>
                        <span class="path-label">파일 경로 </span>
                        <div class="path-box" title="클릭하면 전체가 선택됩니다.">{meta.get("file_path", "경로 없음")}</div>
                    </div>
                </details>
                """, unsafe_allow_html=True)
# -------------------------------------------------------------------------
# 9. 사용자 입력 처리
# -------------------------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# -------------------------------------------------------------------------
# 10. AI 응답 처리 (백엔드 연동) - [최종: 높은 점수가 1등으로 정렬]
# -------------------------------------------------------------------------
if chat_started and st.session_state.messages[-1]["role"] == "user":
    
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()
                
    with st.chat_message("assistant"):
        # [단계별 로딩 상태 표시]
        with st.status("윤슬 AI가 답변을 생성하고 있습니다...", expanded=True) as status:
            st.write("📂 사내 문서 저장소 연결 및 검색 중...")
            time.sleep(0.5)
            
            pipeline = st.session_state.pipeline
            query = st.session_state.messages[-1]["content"]

            # 확장자 필터 전처리
            file_type_input = None if selected_ext == "전체" else selected_ext
            
            st.write("🔍 관련도 분석 및 메타데이터 필터링...")
            time.sleep(0.3)
            
            # 백엔드 검색 실행
            result_dict = pipeline.search_files(
                query=query,
                top_k=top_k,
                include_summary=use_summary,
                include_recommendations=False,
                file_type_filter=file_type_input,
                sort_by=sort_map[sort_option]
            )
            
            results = result_dict.get("results", [])

            # [정렬 보정] 사용자 피드백 반영: 점수가 높을수록 관련도가 높음
            # 따라서 '관련도순'일 때는 점수 내림차순(큰 게 위로) 정렬
            if sort_option == "관련도순":
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            st.write("✨ LLM 답변 요약 및 카드 생성 중...")
            time.sleep(0.3)
            
            st.session_state.last_results = results
            
            status.update(label="검색 및 분석 완료!", state="complete", expanded=False)

        # 기본 응답 메시지
        message_placeholder = st.empty()
        summary_status = "요약을 포함하여" if use_summary else "목록을"
        full_text = f"네, 요청하신 내용을 바탕으로 {summary_status} 관련 문서를 찾았습니다. 설정하신 필터 조건에 맞는 문서는 다음과 같습니다."
        
        # 타이핑 효과
        msg_buffer = ""
        for char in full_text:
            msg_buffer += char
            time.sleep(0.01)
            message_placeholder.markdown(msg_buffer + "▌")
        message_placeholder.markdown(full_text)
        
        # 대화 기록 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_text + f"<br><br>({len(results)}개의 문서가 검색되었습니다.)",
            "results": results
        })
        st.rerun()