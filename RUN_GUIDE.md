# 🚀 AI Agent 실행 가이드

## 📋 사전 요구사항

### 1. 하드웨어
- **GPU**: NVIDIA GPU (CUDA 지원) - 최소 16GB VRAM 권장
- **RAM**: 최소 32GB
- **디스크**: 100GB 이상 여유 공간

### 2. 소프트웨어
- Python 3.8 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Git

---

## 🔧 설치 단계

### Step 1: 가상환경 생성 (권장)
```bash
cd /dais04/DO_NOT_DELETE/HD_AI_Hackathon

python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### Step 2: 의존성 설치
```bash
pip install --upgrade pip

# PyTorch 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
pip install -r requirements.txt
```

**주의사항:**
- `paddlepaddle`: GPU 버전이 필요한 경우 별도 설치 필요
- `vllm`: 선택적 (프로토타입에서는 transformers 사용 가능)

---

## 📊 데이터 준비

### Step 3: 합성 데이터 생성
```bash
python scripts/generate_synthetic_data.py
```

이 스크립트는 다음을 생성합니다:
- PDF 문서 (20개)
- Word 문서 (20개)
- PowerPoint 문서 (20개)
- Excel 문서 (20개)
- 이미지 파일 (40개: PNG, JPG 각 20개)

**총 120개의 테스트 파일**이 `data/network_drive/`에 생성됩니다.

---

## 🗂️ 파일 인덱싱

### Step 4: 파일 인덱싱 실행
```bash
# 기본 인덱싱
python scripts/index_files.py

# 옵션과 함께 실행
python scripts/index_files.py --batch-size 5 --recreate
```

**인덱싱 옵션:**
- `--config`: 설정 파일 경로 (기본: config/config.yaml)
- `--directory`: 인덱싱할 디렉토리 (기본: config의 network_drive)
- `--batch-size`: 배치 크기 (기본: 10)
- `--recreate`: 벡터 DB 재생성 여부

**예상 소요 시간:**
- 120개 파일 기준: 약 5-10분 (GPU 성능에 따라 다름)

**인덱싱 결과:**
- Qdrant 벡터 DB: `qdrant_storage/`
- BM25 인덱스: `cache/bm25_index.pkl`

---

## 🎯 AI Agent 실행

### Step 5: 메인 애플리케이션 실행
```bash
python src/main.py
```

**실행 시 로딩되는 컴포넌트:**
1. BGE-M3 임베딩 모델 (~2GB)
2. Qdrant 벡터 데이터베이스
3. BM25 검색 인덱스
4. ~~Qwen2.5-14B LLM (~28GB)~~ - 선택적

**프로토타입 모드:**
- LLM 요약 기능은 기본적으로 비활성화
- 활성화하려면 `src/main.py`에서 LLM 로딩 부분 주석 해제

### Step 6: 웹 인터페이스 접속
```
http://localhost:7860
```

브라우저에서 위 주소로 접속하면 Gradio UI가 표시됩니다.

---

## 🔍 사용 방법

### 1. 기본 검색
1. 검색어 입력 (예: "매출 보고서")
2. 결과 개수 선택 (1-10)
3. "🔍 검색" 버튼 클릭

### 2. 요약 기능 (선택적)
- "요약 생성" 체크박스 활성화
- **주의**: LLM이 로드되어 있어야 함

### 3. 연관 파일 추천
- "연관 파일 추천" 체크박스 활성화 (기본 활성화)
- 검색 결과와 유사한 파일 추천

---

## 🛠️ 트러블슈팅

### 문제 1: CUDA Out of Memory
**증상:** GPU 메모리 부족 오류

**해결방법:**
```yaml
# config/config.yaml 수정
embedding:
  batch_size: 16  # 32 → 16으로 축소
  use_fp16: true  # 반드시 활성화
```

### 문제 2: PaddleOCR 에러
**증상:** OCR 모델 로드 실패

**해결방법:**
```bash
# CPU 버전 재설치
pip uninstall paddlepaddle
pip install paddlepaddle

# 또는 GPU 버전
pip install paddlepaddle-gpu
```

### 문제 3: 인덱싱 속도가 너무 느림
**해결방법:**
```bash
# 배치 크기 조정
python scripts/index_files.py --batch-size 3

# 또는 GPU 메모리 확인
nvidia-smi
```

### 문제 4: Qwen LLM 로딩 실패
**해결방법:**
- 프로토타입에서는 LLM 없이도 검색 가능
- 요약 기능만 비활성화됨
- 메모리 부족 시 LLM 로딩 건너뛰기

---

## 📈 성능 최적화 팁

### 1. GPU 메모리 최적화
```python
# src/main.py에서 설정 변경
llm_config = LLMConfig(
    use_vllm=False,  # transformers 사용
    # 또는 양자화 모델 사용
)
```

### 2. 인덱싱 속도 향상
- 배치 크기 증가 (GPU 메모리 허용 시)
- 멀티프로세싱 활용 (향후 구현)

### 3. 검색 속도 향상
- 결과 개수 제한 (top_k를 작게)
- 요약 기능 비활성화

---

## 📝 주요 디렉토리 구조

```
HD_AI_Hackathon/
├── data/
│   ├── network_drive/        # 합성 데이터 (120개 파일)
│   └── synthetic_files/       # (사용 안 함)
├── qdrant_storage/            # 벡터 DB (인덱싱 후 생성)
├── cache/
│   └── bm25_index.pkl        # BM25 인덱스 (인덱싱 후 생성)
├── src/
│   ├── parsers/              # 파일 파서
│   ├── embeddings/           # 임베딩 모델
│   ├── search/               # 검색 엔진
│   ├── llm/                  # LLM 요약
│   ├── recommend/            # 추천 시스템
│   └── main.py               # 메인 애플리케이션
├── scripts/
│   ├── generate_synthetic_data.py
│   └── index_files.py
└── config/
    └── config.yaml           # 설정 파일
```

---

## ⚙️ 설정 파일 커스터마이징

`config/config.yaml`에서 다음 설정 가능:

### 검색 설정
```yaml
search:
  top_k: 20              # 초기 검색 결과 수
  rerank_top_k: 5        # 최종 반환 결과 수
  bm25_weight: 0.4       # 키워드 검색 가중치
  semantic_weight: 0.6   # 의미 검색 가중치
```

### 임베딩 설정
```yaml
embedding:
  batch_size: 32         # 배치 크기 (GPU 메모리에 따라 조정)
  max_length: 8192       # 최대 입력 길이
  use_fp16: true         # FP16 사용 (메모리 절약)
```

### LLM 설정
```yaml
llm:
  temperature: 0.1       # 생성 다양성 (낮을수록 보수적)
  max_tokens: 500        # 최대 생성 토큰 수
```

---

## 🎉 다음 단계

프로토타입 검증 후:

1. **LLM 통합**: Qwen2.5-14B 전체 기능 활성화
2. **그래프 DB 추가**: Neo4j로 파일 관계 관리
3. **리랭킹 모델 추가**: bge-reranker-v2-m3
4. **실시간 모니터링**: 파일 변경 감지 및 자동 재인덱싱
5. **API 서버**: FastAPI 엔드포인트 추가
6. **권한 관리**: 사용자별 파일 접근 권한

---

## 📞 문제 발생 시

로그 확인:
```bash
# 인덱싱 로그
python scripts/index_files.py 2>&1 | tee indexing.log

# 애플리케이션 로그
python src/main.py 2>&1 | tee app.log
```

GPU 상태 확인:
```bash
nvidia-smi -l 1  # 1초마다 업데이트
```

---

**프로토타입 버전 1.0**
최종 업데이트: 2024년
