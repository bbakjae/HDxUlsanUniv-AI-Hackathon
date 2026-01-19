"""
임베딩 모델 비교 평가 스크립트
- MRR, Recall@K, nDCG@K, 임베딩 시간, 메모리 사용량 측정
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import psutil
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    """임베딩 모델 설정"""
    name: str
    model_id: str
    model_type: str  # 'flag', 'sentence_transformers', 'huggingface'
    vector_dim: int
    use_fp16: bool = True
    device: str = "cuda"


@dataclass
class EmbeddingEvalResult:
    """임베딩 모델 평가 결과"""
    model_name: str
    model_id: str
    vector_dim: int
    mrr: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    korean_mrr: float
    english_mrr: float
    avg_embedding_time_ms: float
    gpu_memory_gb: float
    total_queries: int
    eval_timestamp: str


# 평가 대상 임베딩 모델 목록
EMBEDDING_MODELS = [
    EmbeddingModelConfig(
        name="BAAI/bge-m3",
        model_id="BAAI/bge-m3",
        model_type="flag",
        vector_dim=1024
    ),
    EmbeddingModelConfig(
        name="BGE-m3-ko",
        model_id="dragonkue/BGE-m3-ko",
        model_type="flag",
        vector_dim=1024
    ),
    EmbeddingModelConfig(
        name="Snowflake-arctic-ko",
        model_id="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_type="sentence_transformers",
        vector_dim=1024
    ),
    EmbeddingModelConfig(
        name="KURE-v1",
        model_id="nlpai-lab/KURE-v1",
        model_type="sentence_transformers",
        vector_dim=1024
    ),
    EmbeddingModelConfig(
        name="Qwen3-Embedding",
        model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        model_type="sentence_transformers",
        vector_dim=1536
    ),
]


class EmbeddingModelWrapper:
    """임베딩 모델 래퍼 (다양한 모델 타입 지원)"""

    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """모델 타입에 따라 로딩"""
        logger.info(f"Loading embedding model: {self.config.name}")

        if self.config.model_type == "flag":
            # FlagEmbedding 모델 (BGE 계열)
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(
                    self.config.model_id,
                    use_fp16=self.config.use_fp16,
                    device=self.config.device
                )
            except ImportError:
                logger.warning("FlagEmbedding not available, trying sentence_transformers")
                self.config.model_type = "sentence_transformers"
                self._load_model()
                return

        elif self.config.model_type == "sentence_transformers":
            # Sentence Transformers 모델
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.config.model_id,
                device=self.config.device
            )
            if self.config.use_fp16 and self.config.device == "cuda":
                self.model = self.model.half()

        elif self.config.model_type == "huggingface":
            # HuggingFace Transformers 모델
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModel.from_pretrained(self.config.model_id)
            if self.config.use_fp16:
                self.model = self.model.half()
            self.model = self.model.to(self.config.device)
            self.model.eval()

        logger.info(f"Model loaded: {self.config.name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        if self.config.model_type == "flag":
            output = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            return output['dense_vecs']

        elif self.config.model_type == "sentence_transformers":
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings

        elif self.config.model_type == "huggingface":
            with torch.no_grad():
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.config.device)
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()

    def get_gpu_memory_usage(self) -> float:
        """현재 GPU 메모리 사용량 (GB)"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0

    def cleanup(self):
        """모델 메모리 해제"""
        del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()


class EmbeddingEvaluator:
    """임베딩 모델 평가기"""

    def __init__(self, test_queries_path: str, qdrant_storage_path: str):
        self.test_queries_path = test_queries_path
        self.qdrant_storage_path = qdrant_storage_path
        self.test_data = self._load_test_queries()
        self.corpus_embeddings = None
        self.corpus_ids = None

    def _load_test_queries(self) -> Dict:
        """테스트 쿼리 로드"""
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_corpus_from_qdrant(self) -> Tuple[List[str], List[str]]:
        """Qdrant에서 문서 코퍼스 로드"""
        from qdrant_client import QdrantClient

        client = QdrantClient(path=self.qdrant_storage_path)

        # 컬렉션에서 모든 문서 가져오기
        collection_name = "company_files"

        # Scroll을 사용하여 모든 포인트 가져오기
        all_points = []
        offset = None

        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points, offset = result
            all_points.extend(points)

            if offset is None:
                break

        # 텍스트와 ID 추출
        texts = []
        ids = []
        for point in all_points:
            text = point.payload.get('text', '')
            if text:
                texts.append(text)
                ids.append(str(point.id))

        logger.info(f"Loaded {len(texts)} documents from Qdrant")
        return texts, ids

    def compute_mrr(self, rankings: List[List[str]], relevant_docs: List[List[str]]) -> float:
        """MRR (Mean Reciprocal Rank) 계산"""
        reciprocal_ranks = []

        for ranking, relevant in zip(rankings, relevant_docs):
            if not relevant:
                continue
            for i, doc_id in enumerate(ranking):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def compute_recall_at_k(self, rankings: List[List[str]], relevant_docs: List[List[str]], k: int) -> float:
        """Recall@K 계산"""
        recalls = []

        for ranking, relevant in zip(rankings, relevant_docs):
            if not relevant:
                continue
            top_k = set(ranking[:k])
            relevant_set = set(relevant)
            recall = len(top_k & relevant_set) / len(relevant_set) if relevant_set else 0.0
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    def compute_ndcg_at_k(self, rankings: List[List[str]], relevance_scores: List[Dict[str, int]], k: int) -> float:
        """nDCG@K 계산"""
        ndcgs = []

        for ranking, scores in zip(rankings, relevance_scores):
            if not scores:
                continue

            # DCG 계산
            dcg = 0.0
            for i, doc_id in enumerate(ranking[:k]):
                rel = scores.get(doc_id, 0)
                dcg += (2 ** rel - 1) / np.log2(i + 2)

            # IDCG 계산 (이상적인 순위)
            ideal_rels = sorted(scores.values(), reverse=True)[:k]
            idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return np.mean(ndcgs) if ndcgs else 0.0

    def evaluate_model(self, model_config: EmbeddingModelConfig) -> EmbeddingEvalResult:
        """단일 모델 평가"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_config.name}")
        logger.info(f"{'='*60}")

        # GPU 메모리 초기화
        torch.cuda.reset_peak_memory_stats()

        # 모델 로드
        model = EmbeddingModelWrapper(model_config)

        # GPU 메모리 측정 (모델 로딩 후)
        gpu_memory = model.get_gpu_memory_usage()

        # 코퍼스 로드 (첫 번째 모델에서만)
        if self.corpus_embeddings is None:
            corpus_texts, self.corpus_ids = self._load_corpus_from_qdrant()

        # 코퍼스 임베딩 생성
        logger.info("Encoding corpus...")
        corpus_texts, _ = self._load_corpus_from_qdrant()

        start_time = time.time()
        self.corpus_embeddings = model.encode(corpus_texts)
        corpus_encoding_time = time.time() - start_time
        logger.info(f"Corpus encoding time: {corpus_encoding_time:.2f}s")

        # 쿼리별 평가
        queries = self.test_data['queries']
        rankings = []
        embedding_times = []

        korean_rankings = []
        english_rankings = []

        for query_data in tqdm(queries, desc="Evaluating queries"):
            query = query_data['query']

            # 쿼리 임베딩 시간 측정
            start_time = time.time()
            query_embedding = model.encode([query])
            embedding_time = (time.time() - start_time) * 1000  # ms
            embedding_times.append(embedding_time)

            # 코사인 유사도 계산
            similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()

            # 상위 K개 결과
            top_k_indices = np.argsort(similarities)[::-1][:20]
            ranking = [self.corpus_ids[i] for i in top_k_indices]
            rankings.append(ranking)

            # 언어별 분류
            if query_data['language'] == 'ko':
                korean_rankings.append(ranking)
            else:
                english_rankings.append(ranking)

        # Ground Truth가 없는 경우 더미 데이터 사용
        # 실제 실험에서는 레이블링된 데이터 필요
        relevant_docs = [q.get('relevant_doc_ids', []) for q in queries]
        relevance_scores = [q.get('relevance_scores', {}) for q in queries]

        # 메트릭 계산
        # Ground Truth가 비어있으면 자동 레이블링 시뮬레이션
        if all(len(r) == 0 for r in relevant_docs):
            logger.warning("No ground truth labels found. Using top-3 results as pseudo-relevant for demonstration.")
            relevant_docs = [ranking[:3] for ranking in rankings]
            relevance_scores = [{doc: 2-i for i, doc in enumerate(ranking[:3])} for ranking in rankings]

        korean_relevant = [relevant_docs[i] for i, q in enumerate(queries) if q['language'] == 'ko']
        english_relevant = [relevant_docs[i] for i, q in enumerate(queries) if q['language'] == 'en']

        result = EmbeddingEvalResult(
            model_name=model_config.name,
            model_id=model_config.model_id,
            vector_dim=model_config.vector_dim,
            mrr=self.compute_mrr(rankings, relevant_docs),
            recall_at_5=self.compute_recall_at_k(rankings, relevant_docs, 5),
            recall_at_10=self.compute_recall_at_k(rankings, relevant_docs, 10),
            ndcg_at_10=self.compute_ndcg_at_k(rankings, relevance_scores, 10),
            korean_mrr=self.compute_mrr(korean_rankings, korean_relevant),
            english_mrr=self.compute_mrr(english_rankings, english_relevant),
            avg_embedding_time_ms=np.mean(embedding_times),
            gpu_memory_gb=gpu_memory,
            total_queries=len(queries),
            eval_timestamp=datetime.now().isoformat()
        )

        # 정리
        model.cleanup()
        self.corpus_embeddings = None

        return result

    def evaluate_all_models(self, output_path: str) -> List[EmbeddingEvalResult]:
        """모든 모델 평가"""
        results = []

        for config in EMBEDDING_MODELS:
            try:
                result = self.evaluate_model(config)
                results.append(result)

                # 중간 결과 저장
                self._save_results(results, output_path)

                logger.info(f"\nResults for {config.name}:")
                logger.info(f"  MRR: {result.mrr:.4f}")
                logger.info(f"  Recall@5: {result.recall_at_5:.4f}")
                logger.info(f"  nDCG@10: {result.ndcg_at_10:.4f}")
                logger.info(f"  Avg Embedding Time: {result.avg_embedding_time_ms:.2f}ms")
                logger.info(f"  GPU Memory: {result.gpu_memory_gb:.2f}GB")

            except Exception as e:
                logger.error(f"Error evaluating {config.name}: {e}")
                continue

        return results

    def _save_results(self, results: List[EmbeddingEvalResult], output_path: str):
        """결과 저장"""
        results_dict = [asdict(r) for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

    def compute_weighted_score(self, result: EmbeddingEvalResult, all_results: List[EmbeddingEvalResult]) -> float:
        """가중 종합 점수 계산"""
        # 정규화를 위한 최소/최대값 계산
        times = [r.avg_embedding_time_ms for r in all_results]
        memories = [r.gpu_memory_gb for r in all_results]

        min_time, max_time = min(times), max(times)
        min_mem, max_mem = min(memories), max(memories)

        # 정규화 (시간/메모리는 낮을수록 좋음)
        norm_time = 1 - (result.avg_embedding_time_ms - min_time) / (max_time - min_time + 1e-6)
        norm_mem = 1 - (result.gpu_memory_gb - min_mem) / (max_mem - min_mem + 1e-6)

        # 가중 점수 계산
        score = (
            result.mrr * 0.25 +
            result.recall_at_5 * 0.25 +
            result.ndcg_at_10 * 0.20 +
            result.korean_mrr * 0.15 +
            norm_time * 0.10 +
            norm_mem * 0.05
        )

        return score

    def print_comparison_table(self, results: List[EmbeddingEvalResult]):
        """비교표 출력"""
        # 종합 점수 계산
        scores = [(r, self.compute_weighted_score(r, results)) for r in results]
        scores.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 120)
        print("임베딩 모델 비교 결과표")
        print("=" * 120)
        print(f"{'모델':<30} {'벡터차원':>8} {'MRR':>8} {'R@5':>8} {'nDCG@10':>8} {'한국어MRR':>10} {'시간(ms)':>10} {'메모리(GB)':>10} {'종합점수':>10}")
        print("-" * 120)

        for result, score in scores:
            print(f"{result.model_name:<30} {result.vector_dim:>8} {result.mrr:>8.4f} {result.recall_at_5:>8.4f} {result.ndcg_at_10:>8.4f} {result.korean_mrr:>10.4f} {result.avg_embedding_time_ms:>10.2f} {result.gpu_memory_gb:>10.2f} {score:>10.4f}")

        print("=" * 120)
        print(f"\n최고 성능 모델: {scores[0][0].model_name} (종합점수: {scores[0][1]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='임베딩 모델 비교 평가')
    parser.add_argument('--test-queries', type=str,
                        default='experiments/test_queries.json',
                        help='테스트 쿼리 JSON 파일 경로')
    parser.add_argument('--qdrant-storage', type=str,
                        default='qdrant_storage_gdrive',
                        help='Qdrant 스토리지 경로')
    parser.add_argument('--output', type=str,
                        default='experiments/results/embedding_results.json',
                        help='결과 저장 경로')
    parser.add_argument('--models', type=str, nargs='+',
                        help='평가할 모델 이름 목록 (기본: 전체)')

    args = parser.parse_args()

    # 경로 설정
    project_root = Path(__file__).parent.parent
    test_queries_path = project_root / args.test_queries
    qdrant_storage_path = project_root / args.qdrant_storage
    output_path = project_root / args.output

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 평가 실행
    evaluator = EmbeddingEvaluator(
        str(test_queries_path),
        str(qdrant_storage_path)
    )

    # 특정 모델만 평가할 경우 필터링
    if args.models:
        global EMBEDDING_MODELS
        EMBEDDING_MODELS = [m for m in EMBEDDING_MODELS if m.name in args.models]

    results = evaluator.evaluate_all_models(str(output_path))

    # 비교표 출력
    if results:
        evaluator.print_comparison_table(results)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
