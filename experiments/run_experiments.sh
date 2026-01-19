#!/bin/bash
#
# AI Agent Service 모델 비교 실험 실행 스크립트
# ============================================
#
# 사용법:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh [phase]
#
# phase:
#   all        - 전체 실험 실행 (기본값)
#   embedding  - 임베딩 모델 실험만
#   llm        - LLM 모델 실험만
#   combination - 조합 최적화 실험만
#

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 설정
PROJECT_ROOT="/dais04/DO_NOT_DELETE/HD_AI_Hackathon"
cd "$PROJECT_ROOT"

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 의존성 확인
check_dependencies() {
    log_info "의존성 확인 중..."

    # rouge-score 설치 확인
    python -c "import rouge_score" 2>/dev/null || {
        log_warning "rouge-score 설치 중..."
        pip install rouge-score
    }

    # FlagEmbedding 설치 확인
    python -c "from FlagEmbedding import BGEM3FlagModel" 2>/dev/null || {
        log_warning "FlagEmbedding 설치 중..."
        pip install FlagEmbedding
    }

    log_success "의존성 확인 완료"
}

# Phase 1: 테스트 데이터 확인
check_test_data() {
    log_info "테스트 데이터 확인 중..."

    if [ ! -f "$PROJECT_ROOT/experiments/test_queries.json" ]; then
        log_error "테스트 쿼리 파일이 없습니다: experiments/test_queries.json"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT/qdrant_storage_gdrive" ]; then
        log_error "Qdrant 스토리지가 없습니다: qdrant_storage_gdrive"
        exit 1
    fi

    log_success "테스트 데이터 확인 완료"
}

# Phase 2: 임베딩 모델 실험
run_embedding_experiment() {
    log_info "=========================================="
    log_info "Phase 2: 임베딩 모델 실험 시작"
    log_info "=========================================="

    echo ""
    echo "평가 대상 모델:"
    echo "  1. BAAI/bge-m3 (현재 사용)"
    echo "  2. dragonkue/BGE-m3-ko"
    echo "  3. dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    echo "  4. nlpai-lab/KURE-v1"
    echo "  5. Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    echo ""

    python -m experiments.eval_embedding \
        --test-queries experiments/test_queries.json \
        --qdrant-storage qdrant_storage_gdrive \
        --output experiments/results/embedding_results.json

    log_success "임베딩 모델 실험 완료"
    log_info "결과: experiments/results/embedding_results.json"
}

# Phase 3: LLM 모델 실험
run_llm_experiment() {
    log_info "=========================================="
    log_info "Phase 3: LLM 모델 실험 시작"
    log_info "=========================================="

    echo ""
    echo "평가 대상 모델:"
    echo "  1. Qwen/Qwen2.5-14B-Instruct (현재 사용)"
    echo "  2. Qwen/Qwen2.5-7B-Instruct"
    echo "  3. Qwen/Qwen2.5-3B-Instruct"
    echo "  4. microsoft/Phi-3.5-mini-instruct"
    echo "  5. google/gemma-2-9b-it"
    echo "  6. mistralai/Mistral-7B-Instruct-v0.3"
    echo ""

    python -m experiments.eval_llm \
        --test-queries experiments/test_queries.json \
        --qdrant-storage qdrant_storage_gdrive \
        --output experiments/results/llm_results.json \
        --num-samples 10

    log_success "LLM 모델 실험 완료"
    log_info "결과: experiments/results/llm_results.json"
}

# Phase 4: 조합 최적화 실험
run_combination_experiment() {
    log_info "=========================================="
    log_info "Phase 4: 조합 최적화 실험 시작"
    log_info "=========================================="

    echo ""
    echo "테스트 조합:"
    echo "  상위 임베딩 2개 × 상위 LLM 2개 = 4개 조합"
    echo ""

    python -m experiments.eval_combination \
        --test-queries experiments/test_queries.json \
        --qdrant-storage qdrant_storage_gdrive \
        --output experiments/results/combination_results.json \
        --num-queries 10

    log_success "조합 최적화 실험 완료"
    log_info "결과: experiments/results/combination_results.json"
}

# 결과 요약
print_summary() {
    log_info "=========================================="
    log_info "실험 결과 요약"
    log_info "=========================================="

    echo ""
    echo "결과 파일 위치:"
    echo "  - 임베딩 모델: experiments/results/embedding_results.json"
    echo "  - LLM 모델: experiments/results/llm_results.json"
    echo "  - 조합 최적화: experiments/results/combination_results.json"
    echo ""

    # 결과 파일 존재 확인
    for file in embedding_results.json llm_results.json combination_results.json; do
        if [ -f "$PROJECT_ROOT/experiments/results/$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (없음)"
        fi
    done

    echo ""
    log_info "상세 결과는 각 JSON 파일을 확인하세요."
}

# 메인 실행
main() {
    local phase="${1:-all}"

    echo ""
    echo "============================================"
    echo "  AI Agent Service 모델 비교 실험"
    echo "============================================"
    echo ""
    echo "실행 옵션: $phase"
    echo "시작 시간: $(date)"
    echo ""

    # 의존성 확인
    check_dependencies

    # 테스트 데이터 확인
    check_test_data

    # 결과 디렉토리 생성
    mkdir -p "$PROJECT_ROOT/experiments/results"

    case "$phase" in
        "embedding")
            run_embedding_experiment
            ;;
        "llm")
            run_llm_experiment
            ;;
        "combination")
            run_combination_experiment
            ;;
        "all")
            run_embedding_experiment
            echo ""
            run_llm_experiment
            echo ""
            run_combination_experiment
            ;;
        *)
            log_error "알 수 없는 옵션: $phase"
            echo "사용법: ./run_experiments.sh [all|embedding|llm|combination]"
            exit 1
            ;;
    esac

    echo ""
    print_summary

    echo ""
    echo "완료 시간: $(date)"
    log_success "실험 완료!"
}

# 실행
main "$@"
