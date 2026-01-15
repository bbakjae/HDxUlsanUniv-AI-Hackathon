import streamlit as st
import time
import sys
from pathlib import Path



# -------------------------------------------------------------------------
# 페이지 기본 설정
# -------------------------------------------------------------------------
st.set_page_config(page_title="Yoonseul AI", layout="wide")

# -------------------------------------------------------------------------
# 경로설정
# -------------------------------------------------------------------------
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import AIAgentPipeline
# -------------------------------------------------------------------------
# AI 파이프 라인 설정
# -------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    config_path = project_root / "config" / "config.yaml"
    return AIAgentPipeline(str(config_path))

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
# -------------------------------------------------------------------------
# 새션 상태 변수
# -------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

chat_started = len(st.session_state.messages) > 0

# -------------------------------------------------------------------------
# CSS 스타일링
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
        header[data-testid="stHeader"] {{ background-color: transparent !important; }}
        div[data-testid="stDecoration"] {{ display: none; }}

        /* [3] 인트로 컨테이너 스타일 (추가됨) */
        .intro-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 50vh;
            text-align: center;
        }}
        .intro-text {{
            font-size: 3.5rem;
            font-weight: 900;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #6a11cb 20%, #2575fc 80%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .intro-sub {{
            font-size: 1.2rem;
            color: #666;
            font-weight: 500;
        }}

        /* [4] 커스텀 토글 디자인 */
        details.custom-toggle {{
            background-color: white; border: 1px solid #e0e0e0;
            border-radius: 8px; margin-bottom: 12px; overflow: hidden;
        }}
        summary.toggle-header {{
            display: flex; align-items: center; padding: 14px 20px;
            cursor: pointer; list-style: none;
        }}

        .header-badge {{
            padding: 4px 10px; border-radius: 4px; color: white;
            font-size: 10px; font-weight: 800; margin-right: 12px;
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
            margin-left: auto; background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px;
        }}
        .toggle-arrow {{ margin-left: 10px; font-size: 12px; color: #999; }}

        .toggle-content {{ padding: 0 20px 20px 20px; }}
        .summary-text {{
            font-size: 14px; background-color: #f8f9fa; padding: 15px;
            border-radius: 8px; margin: 15px 0; border-left: 4px solid #e2e8f0;
        }}

        .path-label {{ font-size: 12px; font-weight: bold; color: #6a11cb; margin-bottom: 5px; display: block; }}
        .path-box {{
            background-color: #262730; color: #00cec9; font-family: monospace;
            font-size: 12px; padding: 10px; border-radius: 6px; word-break: break-all;
        }}

        .animated-banner {{
            text-align: center; padding: 15px; color: white;
            font-weight: bold; font-size: 24px; margin-bottom: 20px;
            border-radius: 0 0 10px 10px;
            background: linear-gradient(90deg, #6a11cb, #2575fc, #6a11cb);
            background-size: 200% 200%;
            animation: shimmer 8s ease infinite;
            position: sticky; top: 0; z-index: 999;
        }}

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
# 사이드바
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("설정 및 필터")

    # [추가] 요약 기능 ON/OFF 설정
    st.subheader("💡 기능 설정")
    use_summary = st.checkbox("문서 요약 사용", value=True, help="LLM을 사용하여 검색된 문서의 핵심 내용을 요약합니다.")
    # use_recommend = st.checkbox("연관 파일 추천 사용", value=False, help="검색된 문서와 유사한 다른 파일을 추천합니다.")
    st.divider()

    # 정렬 및 결과 개수
    sort_option = st.radio("정렬 기준", ["관련도순", "최신순", "오래된순", "파일명순"], index=0)
    sort_map = {
        "관련도순": "relevance",
        "최신순": "date_desc",
        "오래된순": "date_asc",
        "파일명순": "name"
    }
    top_k = st.slider("참고할 문서 개수 (Top K)", 1, 20, 5)

    st.divider()

    # 기간 및 확장자 필터
    st.subheader("⏰ 기간 필터")
    date_range = st.date_input("조회 기간", [])

    st.subheader("📁 확장자 필터")
    selected_ext = st.selectbox(
        "파일 확장자",
        ["전체", "pdf", "docx", "pptx", "xlsx", "image"],
        index=0,
        help="특정 파일 타입만 검색합니다."
    )

# -------------------------------------------------------------------------
# 상단 배너 표시 (채팅 시작 시 출력)
# -------------------------------------------------------------------------
if chat_started:
    st.markdown('<div class="animated-banner">YOONSEUL AI</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Intro (채팅 전 초기 화면)
# -------------------------------------------------------------------------
if not chat_started:
    st.markdown("""
        <div class="intro-container">
            <div class="intro-text">Hello, Yoonseul AI</div>
            <div class="intro-sub">사내 문서 기반 지능형 검색 에이전트</div>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 채팅 히스토리
# -------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------------------------
# 10. 사용자 입력 및 AI 프로세스
# -------------------------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if chat_started and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.status("📂 사내 문서 저장소 연결 및 검색 중...", expanded=True) as status:
            if st.session_state.pipeline is None:
                st.session_state.pipeline = load_pipeline()
            pipeline = st.session_state.pipeline
            query = st.session_state.messages[-1]["content"]

            # 날짜 및 확장자 전처리
            start_dt, end_dt = None, None
            if len(date_range) == 2:
                from datetime import datetime, time

                start_dt = datetime.combine(date_range[0], time.min)
                end_dt = datetime.combine(date_range[1], time.max)

            file_type_input = None if selected_ext == "전체" else selected_ext

            # [핵심] use_summary 값을 include_summary 인자에 전달
            result_dict = pipeline.search_files(
                query=query,
                top_k=top_k,
                include_summary=use_summary,
                include_recommendations=False, # 연관 파일 검색 필요시 수정
                file_type_filter=file_type_input,  # 수정된 변수 사용
                sort_by=sort_map[sort_option]
            )

            results = result_dict.get("results", [])

            # UI 날짜 필터 수동 적용
            if start_dt and end_dt:
                results = pipeline._apply_date_filter(results, {'start_date': start_dt, 'end_date': end_dt})

            st.session_state.last_results = results
            status.update(label="분석 완료!", state="complete", expanded=False)

        # 결과 텍스트 출력
        message_placeholder = st.empty()
        summary_status = "요약을 포함하여" if use_summary else "목록 중심으로"
        full_text = f"네, 요청하신 내용을 바탕으로 {summary_status} 관련 문서를 찾았습니다."

        msg_buffer = ""
        for char in full_text:
            msg_buffer += char
            time.sleep(0.02)
            message_placeholder.markdown(msg_buffer + "▌")

        message_placeholder.markdown(full_text)
        st.session_state.messages.append({"role": "assistant", "content": full_text})

# -------------------------------------------------------------------------
# 결과 카드
# -------------------------------------------------------------------------
if chat_started and st.session_state.last_results:
    st.markdown(f"<br><h3>참고 문서 ({len(st.session_state.last_results)}건)</h3>", unsafe_allow_html=True)

    for f in st.session_state.last_results:
        meta = f.get("metadata", {})
        ftype = str(meta.get("file_type", "pdf")).lower()
        raw_score = f.get("score", 0)
        score_pct = int(raw_score * 100) if raw_score <= 1 else int(raw_score)

        # --- [추가된 로직: 요약 대신 미리보기 생성] ---
        summary_content = f.get("summary", "")

        # 요약이 없거나 "요약 미사용"인 경우 미리보기 텍스트 생성
        if not summary_content or summary_content == "요약 미사용":
            # 1. f['text'] 확인 -> 2. meta['text'] 확인 -> 3. 없으면 안내 문구
            raw_text = f.get("text") or meta.get("text") or ""
            if raw_text:
                # 앞부분 300자 추출 (가독성을 위해 적당히 자름)
                summary_content = f"🔍 문서 미리보기: {raw_text[:300]}..."
            else:
                summary_content = "내용을 불러올 수 없는 문서입니다."
        # ------------------------------------------

        st.markdown(f"""
        <details class="custom-toggle" open>
            <summary class="toggle-header">
                <span class="header-badge badge-{ftype}">{ftype.upper()}</span>
                <span class="header-title">{meta.get("file_name")}</span>
                <span class="header-score">{score_pct}% 일치</span>
                <span class="toggle-arrow">▼</span>
            </summary>
            <div class="toggle-content">
                <div class="summary-text">
                    <strong style="font-size: 16px;">문서 내용 ({'요약' if use_summary else '미리보기'})</strong><br>
                    {summary_content}
                </div>
                <span class="path-label">파일 경로</span>
                <div class="path-box">{meta.get("file_path")}</div>
            </div>
        </details>
        """, unsafe_allow_html=True)