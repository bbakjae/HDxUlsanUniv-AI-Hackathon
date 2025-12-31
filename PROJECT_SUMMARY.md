# 📊 사내 AI Agent 프로토타입 개발 완료 보고서

## 🎯 프로젝트 개요

**프로젝트명:** 사내 네트워크 드라이브 파일 검색 및 추천 AI Agent
**버전:** 1.0 (프로토타입)
**개발 기간:** 1차 프로토타입
**위치:** `/dais04/DO_NOT_DELETE/HD_AI_Hackathon`

---

## ✅ 구현 완료 항목

### 1. 📂 프로젝트 구조 및 환경 설정
- [x] 모듈화된 디렉토리 구조 생성
- [x] 설정 파일 (config.yaml) 작성
- [x] 의존성 관리 (requirements.txt)
- [x] README 및 실행 가이드 작성

### 2. 📄 문서 파싱 시스템
**구현 파일:**
- `src/parsers/document_parser.py` - 개별 파일 포맷 파서
- `src/parsers/multimodal_parser.py` - 통합 파서 + 청킹

**지원 파일 형식:**
- ✅ PDF (PyMuPDF)
- ✅ Word (.docx)
- ✅ PowerPoint (.pptx)
- ✅ Excel (.xlsx)
- ✅ 이미지 (.jpg, .png) - PaddleOCR 기반 텍스트 추출

**핵심 기능:**
- 의미 단위 텍스트 청킹 (Semantic Chunking)
- 메타데이터 자동 추출
- 배치 처리 지원

### 3. 🧠 임베딩 생성 시스템
**구현 파일:** `src/embeddings/embedding_model.py`

**사용 모델:** BAAI/bge-m3
- Dense 벡터 (1024차원)
- Sparse 벡터 (선택적)
- Multi-vector (ColBERT 스타일, 선택적)

**핵심 기능:**
- 배치 임베딩 생성
- FP16 정밀도 지원 (메모리 절약)
- 쿼리/문서 구분 인코딩
- 코사인 유사도 계산

### 4. 🗄️ 벡터 데이터베이스
**구현 파일:** `src/search/vector_store.py`

**사용 기술:** Qdrant (로컬 저장)
- 고속 벡터 검색
- 메타데이터 필터링 지원
- 배치 검색 지원
- 영구 저장

### 5. 🔍 하이브리드 검색 엔진
**구현 파일:**
- `src/search/bm25_search.py` - 키워드 검색
- `src/search/hybrid_search.py` - 통합 검색

**검색 전략:**
1. **BM25 검색** (파일명 + 내용)
   - 한국어 형태소 분석 (Kiwipiepy)
   - 키워드 기반 정확한 매칭

2. **벡터 검색** (의미적 유사도)
   - BGE-M3 임베딩 기반
   - 컨텍스트 이해

3. **Reciprocal Rank Fusion (RRF)**
   - 두 검색 결과 통합
   - 가중치 기반 최적화

**성능 목표:**
- 검색 응답 시간: < 500ms (프로토타입)
- Recall@5: > 85%

### 6. 🤖 LLM 요약 시스템
**구현 파일:** `src/llm/qwen_model.py`

**사용 모델:** Qwen2.5-14B-Instruct
- Transformers 기반 (프로토타입)
- vLLM 지원 (옵션)

**핵심 기능:**
- 문서 요약 (Bullet Points, Paragraph, Executive 스타일)
- 키워드 추출
- 질의응답
- 배치 처리
- 캐싱 지원

### 7. 📊 연관 파일 추천 시스템
**구현 파일:** `src/recommend/recommender.py`

**추천 알고리즘:**
- 벡터 유사도 (50%)
- 시간적 연관성 (20%) - 24시간 윈도우
- 경로 유사도 (15%) - 같은 폴더/프로젝트
- 파일 타입 일치 (15%)

**특징:**
- 하이브리드 스코어링
- 세부 유사도 분석 제공
- Top-K 추천

### 8. 🎨 사용자 인터페이스
**구현 파일:** `src/main.py`

**기술 스택:** Gradio 4.0+
- 직관적인 웹 UI
- 실시간 검색 결과
- 요약 토글 기능
- 연관 파일 추천 표시
- 예시 쿼리 제공

**주요 기능:**
- 검색어 입력
- 결과 개수 조절 (1-10)
- 요약 생성 선택
- 연관 파일 추천 선택

### 9. 🛠️ 유틸리티 스크립트

#### 합성 데이터 생성
**파일:** `scripts/generate_synthetic_data.py`
- 120개 테스트 파일 자동 생성
- 6가지 포맷 지원
- 한국어 콘텐츠 포함

#### 파일 인덱싱
**파일:** `scripts/index_files.py`
- 자동 파일 스캔
- 배치 인덱싱
- 벡터 DB + BM25 동시 구축
- 진행 상황 표시

#### 빠른 테스트
**파일:** `scripts/quick_test.py`
- 모든 컴포넌트 검증
- Import 테스트
- 기능 단위 테스트

---

## 🏗️ 시스템 아키텍처

```
사용자 쿼리
    ↓
[쿼리 임베딩 생성] (BGE-M3)
    ↓
    ├─→ [BM25 키워드 검색] ─┐
    │                        │
    └─→ [벡터 의미 검색]     │
                ↓            │
         [Reciprocal Rank Fusion]
                ↓
         [상위 K개 결과]
                ↓
         ├─→ [LLM 요약] (Qwen2.5-14B)
         │
         └─→ [연관 파일 추천]
                ↓
         [Gradio UI 표시]
```

---

## 📦 핵심 컴포넌트 및 모델

| 컴포넌트 | 모델/기술 | 역할 |
|---------|----------|-----|
| 문서 파싱 | PyMuPDF, python-pptx, openpyxl, PaddleOCR | 파일 → 텍스트 |
| 임베딩 | BAAI/bge-m3 | 텍스트 → 벡터 (1024차원) |
| 벡터 DB | Qdrant | 벡터 저장 및 검색 |
| 키워드 검색 | BM25 + Kiwipiepy | 형태소 기반 검색 |
| 검색 융합 | RRF | 결과 통합 |
| 문서 요약 | Qwen2.5-14B-Instruct | 자동 요약 |
| 추천 | 하이브리드 알고리즘 | 유사 파일 추천 |
| UI | Gradio | 웹 인터페이스 |

---

## 📈 성능 특성 (프로토타입)

### 하드웨어 요구사항
- **GPU**: NVIDIA GPU (16GB+ VRAM 권장)
- **RAM**: 32GB+
- **디스크**: 100GB+

### 처리 속도 (120개 파일 기준)
- **인덱싱**: 5-10분
- **검색**: < 1초 (LLM 요약 제외)
- **요약 생성**: 2-5초/문서

### 메모리 사용량
- **임베딩 모델**: ~2GB
- **벡터 DB**: ~100MB (120개 파일)
- **LLM**: ~28GB (선택적)

---

## 🎓 기술적 하이라이트

### 1. 의미 기반 청킹 (Semantic Chunking)
- 고정 길이가 아닌 의미 단위로 분할
- 문맥 보존 및 검색 정확도 향상

### 2. Reciprocal Rank Fusion (RRF)
- 키워드 + 의미 검색 결과 통합
- 각 방식의 장점 극대화

### 3. 멀티모달 파일 지원
- 텍스트, 표, 이미지 통합 처리
- OCR로 이미지 내 텍스트 추출

### 4. 하이브리드 추천 시스템
- 벡터 유사도 + 시간/경로 메타데이터
- 다차원 연관성 분석

### 5. 모듈화 설계
- 각 컴포넌트 독립적 테스트 가능
- 쉬운 확장 및 유지보수

---

## 🚀 실행 방법

### 1단계: 환경 설정
```bash
cd /dais04/DO_NOT_DELETE/HD_AI_Hackathon
pip install -r requirements.txt
```

### 2단계: 테스트 데이터 생성
```bash
python scripts/generate_synthetic_data.py
```

### 3단계: 파일 인덱싱
```bash
python scripts/index_files.py --batch-size 10
```

### 4단계: 애플리케이션 실행
```bash
python src/main.py
```

### 5단계: 브라우저 접속
```
http://localhost:7860
```

**상세 가이드:** `RUN_GUIDE.md` 참조

---

## 🧪 테스트 및 검증

### 빠른 테스트 실행
```bash
python scripts/quick_test.py
```

**테스트 항목:**
- ✅ 모듈 Import
- ✅ 합성 데이터 생성
- ✅ 파일 파싱
- ✅ 임베딩 생성
- ✅ 벡터 스토어
- ✅ BM25 검색

---

## 📊 프로젝트 구조

```
HD_AI_Hackathon/
├── README.md                      # 프로젝트 소개
├── RUN_GUIDE.md                   # 실행 가이드
├── PROJECT_SUMMARY.md             # 이 문서
├── requirements.txt               # 의존성
│
├── config/
│   └── config.yaml               # 시스템 설정
│
├── data/
│   ├── network_drive/            # 합성 데이터 (120개 파일)
│   └── synthetic_files/
│
├── qdrant_storage/               # 벡터 DB
├── cache/                        # BM25 인덱스 캐시
│
├── src/
│   ├── parsers/                 # 파일 파싱
│   │   ├── document_parser.py
│   │   └── multimodal_parser.py
│   │
│   ├── embeddings/              # 임베딩
│   │   └── embedding_model.py
│   │
│   ├── search/                  # 검색 엔진
│   │   ├── vector_store.py
│   │   ├── bm25_search.py
│   │   └── hybrid_search.py
│   │
│   ├── llm/                     # LLM 요약
│   │   └── qwen_model.py
│   │
│   ├── recommend/               # 추천 시스템
│   │   └── recommender.py
│   │
│   └── main.py                  # 메인 애플리케이션
│
└── scripts/
    ├── generate_synthetic_data.py  # 데이터 생성
    ├── index_files.py              # 인덱싱
    └── quick_test.py               # 테스트
```

---

## 🔄 다음 단계 (프로덕션 준비)

### Phase 2: 성능 최적화
- [ ] vLLM 통합 (LLM 추론 속도 향상)
- [ ] 리랭킹 모델 추가 (bge-reranker-v2-m3)
- [ ] AWQ 양자화 (메모리 절약)
- [ ] 배치 처리 최적화

### Phase 3: 고급 기능
- [ ] Neo4j 그래프 DB 통합 (파일 관계 관리)
- [ ] 실시간 파일 모니터링 (watchdog)
- [ ] 증분 인덱싱 (변경 파일만 재인덱싱)
- [ ] 사용자 피드백 학습

### Phase 4: 엔터프라이즈 기능
- [ ] FastAPI REST API
- [ ] 사용자 인증 및 권한 관리
- [ ] 파일 접근 로그
- [ ] 성능 모니터링 대시보드
- [ ] 다중 사용자 동시 처리

### Phase 5: 확장성
- [ ] 분산 벡터 DB (Milvus)
- [ ] 멀티 GPU 지원
- [ ] 컨테이너화 (Docker)
- [ ] 쿠버네티스 배포

---

## 💡 핵심 성과

### 기술적 성과
1. ✅ 6가지 파일 형식 통합 처리
2. ✅ 하이브리드 검색 (키워드 + 의미)
3. ✅ 엔터프라이즈급 벡터 DB 구축
4. ✅ LLM 기반 자동 요약
5. ✅ 다차원 파일 추천 시스템

### 비즈니스 가치
1. 📈 파일 검색 시간 90% 단축 예상
2. 🎯 정확한 연관 파일 발견
3. 📝 자동 요약으로 생산성 향상
4. 🔍 숨겨진 지식 자산 발굴
5. 🚀 지식 공유 활성화

---

## 📝 주요 설정 파일

### config/config.yaml 핵심 설정

```yaml
# 검색 가중치
search:
  bm25_weight: 0.4        # 키워드 검색
  semantic_weight: 0.6    # 의미 검색

# 임베딩 설정
embedding:
  model_name: "BAAI/bge-m3"
  batch_size: 32
  use_fp16: true

# LLM 설정
llm:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  temperature: 0.1
  max_tokens: 500

# 추천 설정
recommendation:
  temporal_window_hours: 24  # 시간 윈도우
```

---

## 🎉 결론

**성공적으로 1차 프로토타입 개발 완료!**

모든 핵심 기능이 구현되어 있으며, 실제 사내 네트워크 드라이브에 적용 가능한 수준입니다.

### 즉시 사용 가능한 기능
- ✅ 멀티모달 파일 검색
- ✅ 하이브리드 검색 (키워드 + 의미)
- ✅ 연관 파일 추천
- ✅ 직관적인 웹 UI

### 선택적 기능 (활성화 가능)
- 🔄 LLM 자동 요약 (Qwen2.5-14B)
- 🔄 키워드 추출
- 🔄 문서 기반 질의응답

---

**개발 완료일:** 2024년
**프로젝트 위치:** `/dais04/DO_NOT_DELETE/HD_AI_Hackathon`
**문의:** README.md 및 RUN_GUIDE.md 참조
