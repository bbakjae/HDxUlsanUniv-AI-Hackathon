import streamlit as st
import time
import sys
import base64
import os
import subprocess
import platform
import streamlit.components.v1 as components
from pathlib import Path

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Yoonseul AI", 
    layout="wide", 
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# 2. 경로 설정 및 파이프라인 로드
# -------------------------------------------------------------------------
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.main import AIAgentPipeline
except ImportError:
    class AIAgentPipeline:
        def __init__(self, config_path): pass
        def search_files(self, **kwargs): return {"results": []}

@st.cache_resource(show_spinner=False)
def load_pipeline():
    config_path = project_root / "config" / "config.yaml"
    return AIAgentPipeline(str(config_path))

# -------------------------------------------------------------------------
# 3. 유틸리티 함수
# -------------------------------------------------------------------------
@st.cache_data
def get_base64_image(image_path):
    try:
        if not image_path.exists():
            return "" 
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

def open_local_file(file_path):
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            st.toast(f"파일이 존재하지 않습니다: {file_path}")
            return

        system_name = platform.system()
        if system_name == "Windows":
            subprocess.Popen(f'explorer /select,"{abs_path}"')
        elif system_name == "Darwin":
            subprocess.call(["open", "-R", abs_path])
        else:
            subprocess.call(["xdg-open", os.path.dirname(abs_path)])
            
        st.toast(f"파일 탐색기를 실행했습니다.")
    except Exception as e:
        st.error(f"파일 열기 실패: {e}")

def scroll_to_anchor():
    js = """
    <script>
        function scrollToTarget() {
            var element = window.parent.document.getElementById('latest_user_question');
            if (element) {
                element.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }
        setTimeout(scrollToTarget, 100);
        setTimeout(scrollToTarget, 500);
    </script>
    """
    components.html(js, height=0)

# -------------------------------------------------------------------------
# 4. 세션 상태 및 봇 이미지 설정
# -------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_started = len(st.session_state.messages) > 0

bot_image_path = project_root / "lumi.png"
if bot_image_path.exists():
    bot_avatar = str(bot_image_path)
    bot_image_base64 = get_base64_image(bot_image_path)
else:
    bot_avatar = "🤖"
    bot_image_base64 = ""

# -------------------------------------------------------------------------
# 5. CSS 스타일링
# -------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {{
        font-family: 'Pretendard', sans-serif;
    }}
    
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        z-index: 1000;
    }}
    div[data-testid="stDecoration"] {{
        visibility: hidden;
    }}

    /* [핵심 수정] 사이드바 배경색만 #FAFAFA로 강제 변경 */
    section[data-testid="stSidebar"] {{
        background-color: #FAFAFA !important;
    }}

    .normal-header {{
        text-align: center; padding: 20px 0; margin-bottom: 20px;
        color: #A0AEC0; font-size: 14px; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase; width: 100%;
    }}

    /* 메인 컨텐츠 영역 너비: 화면의 90% 사용 */
    .block-container {{
        padding-top: 3rem !important; 
        padding-bottom: 150px !important;
        max-width: 90% !important; 
        margin: 0 auto !important;
    }}

    /* 봇 아바타 스타일링 */
    .bot-avatar-container {{
        width: 60px; height: 60px; min-width: 60px;
        border-radius: 50%; overflow: hidden; margin-right: 12px;
        border: 2px solid #C4C4FF; 
        box-shadow: 0 4px 12px rgba(92, 92, 255, 0.15);
        background-color: #f7fafc; display: flex; align-items: center; justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .bot-avatar-container:hover {{
        transform: scale(1.15); 
        box-shadow: 0 8px 24px rgba(92, 92, 255, 0.25);
    }}

    .bot-avatar-container img {{ width: 100%; height: 100%; object-fit: cover; }}

    div[data-testid="stStatusWidget"] {{
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 15px; background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }}

    .intro-container {{
        text-align: center; 
        padding: 80px 20px;
        animation: fadeIn 1.2s ease;
    }}
    .intro-title {{
        font-size: 3.5rem; font-weight: 900; margin-bottom: 16px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }}
    .intro-sub {{ font-size: 1.2rem; color: #718096; font-weight: 500; }}

    .chat-row {{ 
        display: flex; margin-bottom: 24px; width: 100%; align-items: flex-start; 
    }}
    .chat-row.user {{ justify-content: flex-end; }}
    .chat-row.bot {{ justify-content: flex-start; }}
    
    .chat-bubble {{
        padding: 14px 20px; border-radius: 18px; font-size: 15px; line-height: 1.6;
        max-width: 80%; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .chat-row.user .chat-bubble {{ 
        background-color: #5C5CFF; color: white; border-bottom-right-radius: 4px; font-weight: 500;
    }}
    .chat-row.bot .chat-bubble {{ 
        background-color: #F7FAFC; color: #1A202C; border: 1px solid #E2E8F0; border-bottom-left-radius: 4px; font-weight: 400;
    }}

    div[data-testid="stExpander"] {{
        background-color: #FFFFFF; border: 2px solid #D1D5DB; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px !important;
        transition: all 0.2s ease;
    }}
    div[data-testid="stExpander"]:hover {{
        border-color: #5C5CFF; box-shadow: 0 6px 20px rgba(92, 92, 255, 0.15); transform: translateY(-2px);
    }}
    div[data-testid="stExpander"] > details > summary {{
        padding: 14px 18px !important; 
        font-weight: 800 !important; 
        color: #111827 !important;
        background-color: #fff; 
        font-size: 17px !important;
        letter-spacing: -0.2px !important;
    }}
    div[data-testid="stExpander"] > details > summary:hover {{
        background-color: #F7FAFC !important; color: #5C5CFF !important;
    }}
    div[data-testid="stExpander"] > details > div {{
        padding: 16px 18px !important; border-top: 1px solid #E5E7EB; background-color: #FAFAFA;
    }}

    .badge-base {{
        font-size: 10px; font-weight: 800; padding: 4px 8px; border-radius: 6px;
        color: white; text-transform: uppercase; margin-right: 8px;
    }}
    .bg-pdf {{ background-color: #FF4B4B; }} 
    .bg-docx {{ background-color: #3182CE; }} 
    .bg-pptx {{ background-color: #DD6B20; }} 
    .bg-xlsx {{ background-color: #38A169; }} 
    .bg-txt {{ background-color: #718096; }} 
    .bg-image {{ background-color: #319795; }}

    .content-box {{
        background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px;
        padding: 12px 14px; margin-bottom: 12px; font-size: 14px; color: #374151; line-height: 1.6;
    }}
    .preview-box {{ border-left: 3px solid #A0AEC0; }}
    .ai-summary-box {{ border-left: 3px solid #5C5CFF; background-color: #FBFBFF; }}
    
    .box-label {{
        font-size: 11px; font-weight: 700; display: block; margin-bottom: 4px; text-transform: uppercase;
    }}
    .label-preview {{ color: #718096; }}
    .label-ai {{ color: #5C5CFF; }}

    /* stCode 스타일 */
    .stCode {{ margin: 0px !important; }}
    .stCode > div > div > pre {{
        background-color: #FFFFFF !important;
        border: 2px solid #A0AEC0 !important;
        border-radius: 8px !important;
        color: #2D3748 !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 12px !important;
        display: flex !important; align-items: center !important;
        overflow-x: auto;
    }}

    /* 버튼 스타일 */
    div[data-testid="stButton"] button {{
        width: 100%; border-radius: 8px; 
        font-size: 12px !important; 
        font-weight: 600 !important; 
        min-height: 42px !important; height: 42px !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB; 
        color: #1F2937; 
        background-color: #FFFFFF; 
        margin: 0px !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }}
    div[data-testid="stButton"] button:hover {{
        border-color: #5C5CFF; color: #5C5CFF; background-color: #EEF2FF;
    }}

    .section-title {{
        font-size: 15px; font-weight: 800; color: #9CA3AF; 
        margin-top: 40px !important; margin-bottom: 20px !important; margin-left: 4px !important;
        text-transform: uppercase; letter-spacing: 1px;
    }}

    /* [수정됨] 검색창 영역 너비: 화면의 90% 사용 */
    div[data-testid="stBottom"] > div {{
        width: 100% !important;
        max-width: 90% !important; 
        margin: 0 auto !important;
        left: 0 !important;
        right: 0 !important;
    }}

    .stChatInputContainer {{
        background: transparent !important;
        padding-bottom: 30px !important;
    }}
    
    /* 검색창 및 버튼 크기 확대 */
    .stChatInput > div {{
        border: 2px solid #E2E8F0 !important;
        border-radius: 18px !important;
        background: #FFFFFF !important;
        padding: 15px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }}
    .stChatInput > div:focus-within {{
        border-color: #5C5CFF !important;
        box-shadow: 0 4px 16px rgba(92, 92, 255, 0.2) !important;
    }}
    /* 입력 텍스트 크기 확대 */
    .stChatInput textarea {{
        border: none !important; background: transparent !important; color: #1F2937 !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }}
    /* 전송 버튼 크기 확대 */
    .stChatInput button {{
        background: #5C5CFF !important; border: none !important; border-radius: 14px !important;
        width: 45px !important;
        height: 45px !important;
    }}
    .stChatInput button svg {{
        width: 20px !important; height: 20px !important;
    }}
    
    #latest_user_question {{ scroll-margin-top: 100px; }}
</style>
""", unsafe_allow_html=True)

# CSS 추가: 토글 등 색상 강제 적용
st.markdown("""
<style>
    div[data-baseweb="checkbox"] div[aria-checked="true"] { background-color: #5C5CFF !important; border-color: #5C5CFF !important; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #5C5CFF !important; }
    div[data-baseweb="slider"] div[data-testid="stTickBar"] div { background-color: #5C5CFF !important; }
    div[role="radiogroup"] div[tabindex="0"] { background-color: #5C5CFF !important; border-color: #5C5CFF !important; }
    div[data-baseweb="select"] div[aria-selected="true"] { color: #5C5CFF !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. 사이드바
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Search Options")
    use_summary = st.toggle("AI 요약 활성화", value=True)
    st.markdown("---")
    top_k = st.slider("검색 문서 수", 1, 10, 4)
    sort_option = st.selectbox("정렬", ["관련도순", "최신순", "파일명순"])
    selected_ext = st.selectbox("파일 형식", ["전체", "pdf", "docx", "pptx", "xlsx", "txt"])
    sort_map = {"관련도순": "relevance", "최신순": "date_desc", "파일명순": "name"}

# -------------------------------------------------------------------------
# 7. 메인 함수 정의
# -------------------------------------------------------------------------
def render_doc_card(doc, max_score, unique_key, card_index, is_latest):
    meta = doc.get("metadata", {})
    file_name = meta.get("file_name", "제목 없음")
    file_path = meta.get("file_path", "")
    ftype = str(meta.get("file_type", "txt")).lower()
    
    raw_score = doc.get("score", 0)
    score_pct = int((raw_score / max_score) * 99) if max_score > 0 else 0
    
    raw_text = doc.get("text") or meta.get("text") or ""
    preview_text = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
    
    ai_summary_text = (
        "이 문서는 프로젝트의 핵심 요구사항과 기술 명세서를 포함합니다. "
        "주요 내용으로 시스템 아키텍처 설계 및 데이터 파이프라인 구축 방안이 기술되어 있으며, "
        "보안 프로토콜 준수 사항도 명시되어 있어 개발 시 참고가 필요합니다."
    )

    is_expanded = (card_index < 2) and is_latest

    with st.expander(f"{file_name}", expanded=is_expanded):
        st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:12px;">
                <span class="badge-base bg-{ftype}">{ftype}</span>
                <span style="font-size:11px; font-weight:700; color:#6B7280; margin-left:auto;">{score_pct}% 일치</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="content-box preview-box">
                <span class="box-label label-preview">Text Preview</span>
                {preview_text}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="content-box ai-summary-box">
                <span class="box-label label-ai">AI Summary</span>
                {ai_summary_text}
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1], gap="small")
        
        with col1:
            st.code(file_path, language=None)
                
        with col2:
            if st.button("폴더 열기", key=f"open_{unique_key}", use_container_width=True):
                open_local_file(file_path)

# -------------------------------------------------------------------------
# 8. 화면 렌더링 로직
# -------------------------------------------------------------------------
if chat_started:
    st.markdown('<div class="normal-header">Workplace Intelligence · Yoonseul AI</div>', unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="intro-container">
            <div class="intro-title">Hello, Yoonseul AI</div>
            <div class="intro-sub">사내 문서 기반 지능형 검색 에이전트</div>
        </div>
    """, unsafe_allow_html=True)

total_messages = len(st.session_state.messages)

for msg_idx, msg in enumerate(st.session_state.messages):
    is_latest_message = (msg_idx == total_messages - 1)

    if msg["role"] == "user":
        if msg_idx >= total_messages - 2:
             st.markdown("<div id='latest_user_question'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chat-row user">
            <div class="chat-bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    elif msg["role"] == "assistant":
        bot_img_tag = f'<img src="data:image/png;base64,{bot_image_base64}">' if bot_image_base64 else '<div style="font-size:24px;">🤖</div>'
        st.markdown(f"""
        <div class="chat-row bot">
            <div class="bot-avatar-container">{bot_img_tag}</div>
            <div class="chat-bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if "results" in msg and msg["results"]:
            results = msg["results"]
            max_score = max([r.get("score", 0) for r in results]) if results else 0
            
            st.markdown(f'<div class="section-title">REFERENCED DOCUMENTS ({len(results)})</div>', unsafe_allow_html=True)
            
            card_count = 0
            for i in range(0, len(results), 2):
                cols = st.columns(2)
                with cols[0]:
                    unique_key_left = f"btn_{msg_idx}_{i}"
                    render_doc_card(results[i], max_score, unique_key_left, card_count, is_latest_message)
                    card_count += 1
                if i + 1 < len(results):
                    with cols[1]:
                        unique_key_right = f"btn_{msg_idx}_{i+1}"
                        render_doc_card(results[i+1], max_score, unique_key_right, card_count, is_latest_message)
                        card_count += 1
            st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 9. 입력창 및 로직
# -------------------------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    scroll_to_anchor()

    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()

    with st.status("윤슬 AI가 답변을 생성하고 있습니다...", expanded=True) as status:
        st.write("📂 사내 문서 저장소 연결 및 검색 중...")
        time.sleep(0.5)
        
        pipeline = st.session_state.pipeline
        query = st.session_state.messages[-1]["content"]

        file_type_input = None if selected_ext == "전체" else selected_ext
        
        st.write("🔍 관련도 분석 및 메타데이터 필터링...")
        time.sleep(0.3)
        
        result_dict = pipeline.search_files(
            query=query,
            top_k=top_k,
            include_summary=use_summary,
            include_recommendations=False,
            file_type_filter=file_type_input,
            sort_by=sort_map[sort_option]
        )
        
        results = result_dict.get("results", [])
        if sort_option == "관련도순":
            results.sort(key=lambda x: x.get("score", 0), reverse=True)

        st.write("✨ 결과 카드 생성 중...")
        time.sleep(0.3)
        
        status.update(label="검색 및 분석 완료!", state="complete", expanded=False)

    response_text = f"탐색을 완료했습니다! 요청하신 내용과 가장 관련 있는 문서 {len(results)}건을 찾았습니다."

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "results": results
    })
    
    st.rerun()