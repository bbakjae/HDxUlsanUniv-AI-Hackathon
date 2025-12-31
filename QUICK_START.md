# 🚀 빠른 시작 가이드

## ✅ 현재 상태

**프로토타입 개발 완료!**
- ✅ 합성 데이터 생성 완료 (120개 파일)
- ✅ 문서 파서 작동 확인 (PDF, DOCX, PPTX, XLSX)
- 🔄 임베딩 및 검색 엔진 (설치 필요)
- 🔄 LLM 요약 시스템 (선택적)

---

## 📊 이미 완료된 작업

### 1. 합성 데이터 생성 ✅
```bash
# 이미 실행 완료!
# 120개 파일이 data/network_drive/에 생성됨
ls -lh data/network_drive/ | wc -l  # 120개 파일 확인
```

### 2. 문서 파서 테스트 ✅
- PDF 파싱 ✅
- Word 파싱 ✅
- PowerPoint 파싱 ✅
- Excel 파싱 ✅
- 이미지 OCR (선택적 - PaddleOCR 설치 필요)

---

## 🔧 다음 단계

### Option 1: 최소 설치 (파서만 사용)

**이미 설치된 패키지:**
- ✅ python-docx
- ✅ python-pptx
- ✅ openpyxl
- ✅ PyMuPDF
- ✅ Pillow
- ✅ reportlab
- ✅ pyyaml

**이것만으로도 문서 파싱은 가능합니다!**

### Option 2: 검색 엔진 추가 (권장)

임베딩 및 검색 기능을 사용하려면 추가 설치:

```bash
# 1. 임베딩 모델 (필수)
pip install sentence-transformers FlagEmbedding transformers torch accelerate

# 2. 벡터 DB (필수)
pip install qdrant-client

# 3. BM25 검색 (필수)
pip install rank-bm25 kiwipiepy

# 4. 유틸리티
pip install pydantic diskcache
```

**예상 다운로드:** ~5GB (모델 포함)
**예상 시간:** 10-20분

### Option 3: 전체 기능 (LLM 포함)

LLM 요약까지 사용하려면:

```bash
# Option 2의 모든 패키지 +
pip install transformers>=4.36.0

# 선택적: vLLM (고속 추론)
pip install vllm
```

**주의:** LLM 사용 시 GPU 메모리 ~28GB 필요

---

## 🧪 간단한 테스트

### 1. 파서 테스트 (현재 가능!)

```python
from src.parsers.document_parser import DocumentParser
from pathlib import Path

parser = DocumentParser()

# PDF 파일 파싱
pdf_file = list(Path('data/network_drive').glob('*.pdf'))[0]
result = parser.parse_file(str(pdf_file))

print(f"File: {result['metadata']['file_name']}")
print(f"Text: {result['text'][:200]}...")
```

### 2. 전체 파이프라인 테스트 (Option 2 설치 후)

```bash
# 파일 인덱싱
python scripts/index_files.py --batch-size 5

# AI Agent 실행
python src/main.py
```

브라우저에서 `http://localhost:7860` 접속

---

## 📝 설치 옵션 요약

| 기능 | 필요 패키지 | 다운로드 크기 | 메모리 |
|-----|-----------|--------------|--------|
| **파서만** | 이미 설치됨 ✅ | 0 MB | < 1GB |
| **검색 엔진** | Option 2 | ~5 GB | ~8GB |
| **LLM 요약** | Option 3 | ~30 GB | ~32GB |

---

## 🎯 추천 경로

### 처음 시작하는 경우
1. ✅ 현재 상태에서 파서 테스트 해보기
2. Option 2 설치 (검색 엔진)
3. 인덱싱 실행
4. 검색 테스트
5. (선택적) LLM 추가

### GPU가 충분한 경우
- Option 3까지 모두 설치하여 전체 기능 사용

### GPU가 부족한 경우
- Option 2까지만 설치
- LLM 요약 기능은 비활성화 상태로 사용

---

## 🔍 현재 파일 구조

```
HD_AI_Hackathon/
├── data/
│   └── network_drive/          # ✅ 120개 파일 생성 완료
├── src/
│   ├── parsers/               # ✅ 작동 확인 완료
│   ├── embeddings/            # 🔄 Option 2 필요
│   ├── search/                # 🔄 Option 2 필요
│   ├── llm/                   # 🔄 Option 3 필요
│   └── recommend/             # 🔄 Option 2 필요
└── scripts/
    ├── generate_synthetic_data.py  # ✅ 실행 완료
    ├── index_files.py              # 🔄 Option 2 후 실행
    └── quick_test.py               # 🔄 Option 2 후 실행
```

---

## ⚡ 빠른 명령어

```bash
# 현재 위치 확인
pwd  # /dais04/DO_NOT_DELETE/HD_AI_Hackathon

# 생성된 파일 확인
ls -lh data/network_drive/ | head -20

# 파일 개수 확인
find data/network_drive -type f | wc -l  # 120

# 파일 형식별 개수
for ext in pdf docx pptx xlsx png jpg; do
  echo "$ext: $(find data/network_drive -name "*.$ext" | wc -l)"
done
```

---

## 💡 문제 해결

### 문제 1: ModuleNotFoundError
**해결:** 필요한 옵션의 패키지 설치

### 문제 2: CUDA 없음
**해결:** CPU 모드로 실행 가능 (느리지만 작동)
```python
# config/config.yaml에서
embedding:
  device: "cpu"  # cuda → cpu로 변경
```

### 문제 3: 메모리 부족
**해결:** 배치 크기 축소
```python
# config/config.yaml에서
embedding:
  batch_size: 8  # 32 → 8로 축소
```

---

## 📚 상세 문서

- **[README.md](README.md)** - 프로젝트 개요
- **[RUN_GUIDE.md](RUN_GUIDE.md)** - 전체 실행 가이드
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 개발 완료 보고서

---

## 🎉 다음 스텝

1. **Option 2 설치** (권장)
   ```bash
   pip install sentence-transformers FlagEmbedding transformers torch accelerate
   pip install qdrant-client rank-bm25 kiwipiepy pydantic diskcache
   ```

2. **인덱싱 실행**
   ```bash
   python scripts/index_files.py --batch-size 5
   ```

3. **AI Agent 실행**
   ```bash
   python src/main.py
   ```

**Good Luck! 🚀**
