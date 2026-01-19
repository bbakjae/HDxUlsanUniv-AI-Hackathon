# 사내 네트워크 드라이브 AI Agent 프로토타입

## 프로젝트 개요
사내 네트워크 드라이브의 파일들을 지능적으로 검색하고, 내용을 요약하며, 연관 파일을 추천하는 로컬 LLM 기반 AI Agent

## 주요 기능
1. **멀티모달 파일 검색**: ppt, xlsx, docx, pdf, jpg, png 파일 지원
2. **하이브리드 검색**: 파일명 + 내용 기반 검색
3. **자동 요약**: Qwen2.5-14B-Instruct 기반 문서 요약
4. **연관 파일 추천**: 벡터 유사도 + 메타데이터 기반 추천

## 프로젝트 구조
```
HD_AI_Hackathon/
├── data/                      # 데이터 저장소
│   ├── synthetic_files/       # 합성 데이터 생성 위치
│   └── network_drive/         # 가상 네트워크 드라이브
├── src/                       # 소스 코드
│   ├── parsers/              # 파일 파서 모듈
│   ├── embeddings/           # 임베딩 생성 모듈
│   ├── search/               # 검색 엔진 모듈
│   ├── llm/                  # LLM 추론 모듈
│   └── index/                # 간단한 인덱싱, 몽고 테스트용 폴더
│   └── front/                # streamlit 프론트 뷰
│   └── recommend/            # 추천 시스템 모듈

├── config/                    # 설정 파일
├── scripts/                   # 유틸리티 스크립트
├── tests/                     # 테스트 코드
├── cache/                     # 캐시 저장소
└── qdrant_storage/           # Qdrant 벡터 DB 저장소

## 기술 스택
- **문서 파싱**: PyMuPDF, python-pptx, openpyxl, python-docx, PaddleOCR
- **임베딩**: BAAI/bge-m3, dragonkue/BGE-m3-ko
- **벡터 DB**: Qdrant
- **검색**: BM25 + 시맨틱 검색 + Reranking
- **LLM**: Qwen2.5-14B-Instruct (vLLM)
- **백엔드**: FastAPI
- **프론트엔드**: Gradio, streamlit

## 설치 및 실행
1. 의존성 설치: `pip install -r requirements.txt`
2. 합성 데이터 생성: `python scripts/generate_synthetic_data.py`
3. 파일 인덱싱: `python scripts/index_files.py`
4. 서버 실행: `python src/main.py` (또는 streamlit run src/front/prat11.py)
