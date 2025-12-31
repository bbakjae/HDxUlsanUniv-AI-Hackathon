# 🚀 순차적 실행 가이드

**현재 위치:** `/dais04/DO_NOT_DELETE/HD_AI_Hackathon`

---

## 📊 현재 상태

✅ **완료됨:**
- 프로젝트 구조 생성
- 합성 데이터 120개 파일 생성
- 문서 파서 테스트 완료

🔄 **다음 단계:**
- 필수 패키지 설치
- 파일 인덱싱
- AI Agent 실행

---

## 🎯 실행 경로 선택

### **경로 A: 최소 기능 (빠른 테스트)** ⚡
- 소요 시간: ~10분
- 필요 용량: ~5GB
- GPU 메모리: ~8GB
- 기능: 문서 파싱 + 검색 (요약 제외)

### **경로 B: 전체 기능** 🔥
- 소요 시간: ~30분
- 필요 용량: ~30GB
- GPU 메모리: ~32GB
- 기능: 문서 파싱 + 검색 + LLM 요약

**→ 처음이라면 경로 A를 추천합니다!**

---

## 📍 경로 A: 최소 기능 실행 (권장)

### **Step 1: 필수 패키지 설치** (5-10분)

```bash
# 현재 디렉토리 확인
pwd  # /dais04/DO_NOT_DELETE/HD_AI_Hackathon

# 1. 임베딩 및 NLP 패키지
pip install sentence-transformers transformers torch accelerate

# 2. 벡터 데이터베이스
pip install qdrant-client

# 3. 검색 엔진
pip install rank-bm25 kiwipiepy

# 4. 유틸리티
pip install pydantic diskcache fastapi uvicorn gradio
```

**진행 상황 확인:**
```bash
python -c "
import torch
import sentence_transformers
import qdrant_client
print('✅ 모든 패키지 설치 완료!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

### **Step 2: 컴포넌트 테스트** (1-2분)

간단한 테스트로 모든 모듈이 정상 작동하는지 확인:

```bash
python << 'PYTEST'
import sys
sys.path.insert(0, '/dais04/DO_NOT_DELETE/HD_AI_Hackathon')

print("\n=== 컴포넌트 테스트 ===\n")

# 1. 파서 테스트
print("1. 문서 파서...", end=" ")
from src.parsers.document_parser import DocumentParser
parser = DocumentParser()
print("✅")

# 2. 임베딩 테스트
print("2. 임베딩 모델 로딩...", end=" ")
from src.embeddings.embedding_model import BGEM3Embedder
import torch
embedder = BGEM3Embedder(
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_fp16=torch.cuda.is_available()
)
print("✅")

# 3. 벡터 스토어 테스트
print("3. 벡터 데이터베이스...", end=" ")
from src.search.vector_store import QdrantVectorStore
print("✅")

# 4. BM25 검색 테스트
print("4. BM25 검색 엔진...", end=" ")
from src.search.bm25_search import BM25SearchEngine
print("✅")

print("\n✅ 모든 컴포넌트 정상 작동!\n")
PYTEST
```

---

### **Step 3: 파일 인덱싱** (5-10분)

120개 파일을 분석하고 벡터 데이터베이스에 저장:

```bash
# 소규모 테스트 (처음 10개 파일만)
python scripts/index_files.py --batch-size 2 --recreate

# 전체 인덱싱 (120개 모두)
# python scripts/index_files.py --batch-size 5 --recreate
```

**예상 출력:**
```
INFO - Loading BGE-M3 model...
INFO - Model loaded successfully
INFO - Found 120 files in /dais04/DO_NOT_DELETE/HD_AI_Hackathon/data/network_drive
Indexing files: 100%|██████████| 24/24 [00:XX<00:00]
INFO - Indexing complete!
  Successful: 120
  Failed: 0
  Vector DB count: 120
```

**인덱싱 확인:**
```bash
# 생성된 파일 확인
ls -lh qdrant_storage/
ls -lh cache/
```

---

### **Step 4: AI Agent 실행** (즉시)

```bash
# 메인 애플리케이션 실행
python src/main.py
```

**예상 출력:**
```
INFO - Loading config from config/config.yaml
INFO - Initializing AI Agent components...
INFO - Loading embedding model...
INFO - Model loaded successfully
INFO - Connecting to vector store...
INFO - Loading BM25 index...
INFO - All components initialized successfully!
INFO - Launching Gradio interface...
Running on local URL:  http://0.0.0.0:7860
```

**→ 브라우저에서 접속: `http://localhost:7860`**

---

### **Step 5: 검색 테스트**

웹 UI에서 테스트:

1. **검색어 입력:**
   - "매출 보고서"
   - "개발팀 계획서"
   - "2024년 분석자료"

2. **결과 확인:**
   - 파일 경로
   - 유사도 점수
   - 내용 미리보기

3. **연관 파일 추천:**
   - 체크박스 활성화
   - 추천된 파일 목록 확인

---

## 🔍 문제 해결 (경로 A)

### 문제 1: CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# config/config.yaml 수정
nano config/config.yaml

# 다음과 같이 변경:
embedding:
  batch_size: 8      # 32 → 8
  use_fp16: true     # 확인
  device: "cuda"     # 또는 "cpu"
```

재실행:
```bash
python scripts/index_files.py --batch-size 2 --recreate
```

---

### 문제 2: 모델 다운로드 실패

**증상:**
```
HTTPError: 404 Not Found
```

**해결:**
```bash
# 인터넷 연결 확인
ping huggingface.co

# 또는 수동 다운로드
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('BAAI/bge-m3')
print('모델 다운로드 완료')
"
```

---

### 문제 3: Kiwipiepy 에러

**증상:**
```
ModuleNotFoundError: No module named 'kiwipiepy'
```

**해결:**
```bash
pip install kiwipiepy
```

---

## 📍 경로 B: 전체 기능 (LLM 포함)

경로 A를 완료한 후 진행하세요.

### **Step B1: LLM 설정 수정**

```bash
# src/main.py 수정
nano src/main.py

# 85번째 줄 근처에서 주석 해제:
# try:
#     logger.info("Loading LLM (this may take a while)...")
#     llm_config = LLMConfig(...)
#     self.summarizer = QwenSummarizer(llm_config)
```

또는 자동으로:
```bash
sed -i 's/# try:/try:/g' src/main.py
sed -i 's/# logger.info("Loading LLM/logger.info("Loading LLM/g' src/main.py
```

---

### **Step B2: Qwen 모델 다운로드** (20-30분)

```bash
# 모델 다운로드 (약 28GB)
python << 'PYMODEL'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Qwen2.5-14B 모델 다운로드 중... (시간 소요)")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("✅ 모델 다운로드 완료!")
PYMODEL
```

---

### **Step B3: LLM 포함하여 실행**

```bash
python src/main.py
```

웹 UI에서 **"요약 생성" 체크박스 활성화**하여 테스트

---

## 🎯 빠른 명령어 체크리스트

### 초기 설정
```bash
# 1. 디렉토리 이동
cd /dais04/DO_NOT_DELETE/HD_AI_Hackathon

# 2. 패키지 설치
pip install sentence-transformers transformers torch accelerate qdrant-client rank-bm25 kiwipiepy pydantic diskcache fastapi uvicorn gradio

# 3. 테스트
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 인덱싱
```bash
# 소규모 테스트
python scripts/index_files.py --batch-size 2 --recreate

# 전체 인덱싱
python scripts/index_files.py --batch-size 5 --recreate
```

### 실행
```bash
# AI Agent 시작
python src/main.py

# 다른 터미널에서 로그 확인
tail -f app.log  # (로그 파일이 있다면)
```

### 확인
```bash
# 인덱싱된 파일 수
python -c "
import sys
sys.path.insert(0, '.')
from src.search.vector_store import QdrantVectorStore
store = QdrantVectorStore('qdrant_storage', 'company_files')
print(f'인덱싱된 문서: {store.count_documents()}개')
"
```

---

## 📊 진행 상황 체크리스트

### 경로 A (최소 기능)
- [ ] Step 1: 패키지 설치 완료
- [ ] Step 2: 컴포넌트 테스트 통과
- [ ] Step 3: 파일 인덱싱 완료 (120개)
- [ ] Step 4: AI Agent 실행 성공
- [ ] Step 5: 검색 테스트 완료

### 경로 B (전체 기능)
- [ ] 경로 A 완료
- [ ] Step B1: LLM 설정 수정
- [ ] Step B2: Qwen 모델 다운로드
- [ ] Step B3: 요약 기능 테스트

---

## 🎉 성공 기준

### 최소 기능 성공
- ✅ 검색 쿼리 입력 시 결과 반환
- ✅ 파일 경로 및 점수 표시
- ✅ 연관 파일 추천 작동

### 전체 기능 성공
- ✅ 위 모든 기능 +
- ✅ 문서 자동 요약 생성
- ✅ 키워드 추출

---

## 📞 도움말

### 로그 확인
```bash
# Python 스크립트 실행 시 로그 저장
python src/main.py 2>&1 | tee app.log
```

### GPU 상태 확인
```bash
# NVIDIA GPU 모니터링
watch -n 1 nvidia-smi

# 또는
nvidia-smi -l 1
```

### 프로세스 확인
```bash
# 실행 중인 Python 프로세스
ps aux | grep python

# 포트 사용 확인
lsof -i :7860
```

---

**다음 단계:** Step 1부터 순서대로 진행하세요! 🚀
