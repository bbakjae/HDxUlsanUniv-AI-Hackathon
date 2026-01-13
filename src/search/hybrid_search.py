"""
하이브리드 검색 엔진 - Multi-Vector + BM25 결합
Reciprocal Rank Fusion (RRF) 알고리즘 사용

Multi-Vector 모드:
- Qdrant 내장 RRF Fusion (Dense + Sparse)
- BM25는 fallback으로 사용

Legacy 모드:
- BM25 + Dense Vector RRF Fusion
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from .vector_store import QdrantVectorStore

# BM25 모듈 조건부 import (Multi-Vector 모드에서는 선택적)
try:
    from .bm25_search import BM25SearchEngine
    BM25_AVAILABLE = True
except ImportError:
    BM25SearchEngine = None
    BM25_AVAILABLE = False

if TYPE_CHECKING:
    from .bm25_search import BM25SearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    하이브리드 검색 엔진
    - Multi-Vector: Qdrant 내장 Dense + Sparse RRF Fusion
    - Legacy: BM25 + Dense Vector RRF Fusion
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedder=None,
        bm25_engine: Optional['BM25SearchEngine'] = None,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
        use_multi_vector: bool = True
    ):
        """
        Args:
            vector_store: Qdrant 벡터 스토어
            embedder: 임베딩 모델 (Multi-Vector 모드에서 쿼리 임베딩 생성용)
            bm25_engine: BM25 검색 엔진 (Legacy 모드용)
            bm25_weight: BM25/Sparse 검색 가중치
            vector_weight: Dense 벡터 검색 가중치
            rrf_k: RRF 파라미터
            use_multi_vector: Multi-Vector 모드 사용 (Qdrant 내장 RRF)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_engine = bm25_engine
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.use_multi_vector = use_multi_vector and vector_store.use_sparse

        # 가중치 정규화
        total_weight = bm25_weight + vector_weight
        self.bm25_weight /= total_weight
        self.vector_weight /= total_weight

        if self.use_multi_vector:
            logger.info("HybridSearchEngine initialized with Multi-Vector mode (Qdrant RRF)")
        else:
            logger.info("HybridSearchEngine initialized with Legacy mode (BM25 + Dense RRF)")

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        final_top_k: Optional[int] = None,
        use_rrf: bool = True,
        include_explanation: bool = True,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리 텍스트
            query_embedding: Dense 쿼리 임베딩 벡터
            top_k: 각 검색 방법에서 가져올 결과 수
            final_top_k: 최종 반환할 결과 수 (None이면 top_k 사용)
            use_rrf: RRF 사용 여부 (False면 가중 평균 사용)
            include_explanation: 검색 결과 설명 포함 여부
            sparse_vector: Sparse 쿼리 벡터 (Multi-Vector 모드용)

        Returns:
            검색 결과 리스트
        """
        final_top_k = final_top_k or top_k

        # Multi-Vector 모드: Qdrant 내장 RRF 사용
        if self.use_multi_vector and sparse_vector:
            return self._search_multi_vector(
                query=query,
                query_embedding=query_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
                final_top_k=final_top_k,
                include_explanation=include_explanation
            )

        # Legacy 모드: BM25 + Dense Vector RRF
        return self._search_legacy(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            final_top_k=final_top_k,
            use_rrf=use_rrf,
            include_explanation=include_explanation
        )

    def _search_multi_vector(
        self,
        query: str,
        query_embedding: np.ndarray,
        sparse_vector: Dict[int, float],
        top_k: int,
        final_top_k: int,
        include_explanation: bool
    ) -> List[Dict]:
        """
        Multi-Vector 검색 (Qdrant 내장 RRF)
        """
        logger.debug(f"Multi-Vector search with query: {query}")

        # Qdrant 하이브리드 검색 (Dense + Sparse RRF)
        results = self.vector_store.search(
            query_vector=query_embedding,
            sparse_vector=sparse_vector,
            top_k=final_top_k,
            use_hybrid=True,
            dense_weight=self.vector_weight,
            sparse_weight=self.bm25_weight
        )

        # 결과 포맷팅
        formatted_results = []
        for result in results:
            payload = result.get('payload', {})

            formatted = {
                'id': result['id'],
                'score': result['score'],
                'text': payload.get('text', ''),
                'metadata': payload,
                'payload': payload
            }

            if include_explanation:
                formatted['explanation'] = {
                    'search_type': ['multi_vector'],
                    'dense_score': result['score'],  # RRF 통합 점수
                    'sparse_score': result['score'],
                    'fusion_method': 'qdrant_rrf'
                }

            formatted_results.append(formatted)

        return formatted_results

    def _search_legacy(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
        final_top_k: int,
        use_rrf: bool,
        include_explanation: bool
    ) -> List[Dict]:
        """
        Legacy 검색 (BM25 + Dense Vector RRF)
        """
        # 쿼리 토큰 추출 (설명용)
        query_tokens = []
        if include_explanation and self.bm25_engine and self.bm25_engine.tokenizer:
            query_tokens = self.bm25_engine.tokenizer.tokenize(query)

        # 1. BM25 검색
        bm25_results = []
        if self.bm25_engine:
            logger.debug(f"BM25 search with query: {query}")
            bm25_results = self.bm25_engine.search(query, top_k=top_k)

        # 2. 벡터 검색
        logger.debug(f"Vector search")
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # BM25 점수와 벡터 점수 기록 (설명용)
        bm25_scores = {r['id']: r['score'] for r in bm25_results}
        vector_scores = {r['id']: r['score'] for r in vector_results}

        # 3. 결과 융합
        if use_rrf:
            fused_results = self._reciprocal_rank_fusion(
                bm25_results, vector_results, final_top_k
            )
        else:
            fused_results = self._weighted_fusion(
                bm25_results, vector_results, final_top_k
            )

        # 4. 검색 결과 설명 추가
        if include_explanation:
            for result in fused_results:
                doc_id = result['id']
                explanation = {
                    'query_keywords': query_tokens,
                    'bm25_score': bm25_scores.get(doc_id, 0.0),
                    'vector_score': vector_scores.get(doc_id, 0.0),
                    'matched_keywords': [],
                    'search_type': []
                }

                if doc_id in bm25_scores:
                    explanation['search_type'].append('keyword')
                if doc_id in vector_scores:
                    explanation['search_type'].append('semantic')

                doc_text = result.get('text', '') or result.get('payload', {}).get('text', '')
                if doc_text and query_tokens:
                    doc_text_lower = doc_text.lower()
                    for token in query_tokens:
                        if token.lower() in doc_text_lower:
                            explanation['matched_keywords'].append(token)

                result['explanation'] = explanation

        return fused_results

    def search_with_sparse(
        self,
        query: str,
        dense_vector: Optional[np.ndarray] = None,
        sparse_vector: Optional[Dict[int, float]] = None,
        top_k: int = 20,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Multi-Vector 검색 (Sparse 벡터 직접 전달 또는 자동 생성)

        Args:
            query: 검색 쿼리
            dense_vector: Dense 쿼리 벡터 (None이면 embedder로 자동 생성)
            sparse_vector: Sparse 쿼리 벡터 {token_id: weight} (None이면 embedder로 자동 생성)
            top_k: 반환할 결과 수
            filter_conditions: 필터 조건

        Returns:
            검색 결과
        """
        # embedder가 있고 벡터가 제공되지 않은 경우 자동 생성
        if (dense_vector is None or sparse_vector is None) and self.embedder is not None:
            query_result = self.embedder.encode_query_for_search(query)
            if dense_vector is None:
                dense_vector = query_result['dense_vec']
            if sparse_vector is None:
                sparse_vector = query_result['sparse_vec']

        if dense_vector is None:
            raise ValueError("dense_vector is required when embedder is not provided")

        if self.use_multi_vector and sparse_vector:
            results = self.vector_store.search(
                query_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter_conditions=filter_conditions,
                use_hybrid=True
            )
        else:
            # Fallback to legacy
            results = self.vector_store.search(
                query_vector=dense_vector,
                top_k=top_k,
                filter_conditions=filter_conditions
            )

        # 결과 포맷팅
        formatted_results = []
        for result in results:
            payload = result.get('payload', {})
            formatted = {
                'id': result['id'],
                'score': result['score'],
                'text': payload.get('text', ''),
                'metadata': payload,
                'payload': payload
            }
            formatted_results.append(formatted)

        return formatted_results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion 알고리즘

        RRF score = sum(1 / (k + rank))
        """
        rrf_scores = {}
        doc_info = {}

        # BM25 결과 처리
        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            doc_info[doc_id] = result

        # 벡터 검색 결과 처리
        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score

            if doc_id not in doc_info:
                doc_info[doc_id] = result
            else:
                existing_payload = doc_info[doc_id].get('payload', {})
                new_payload = result.get('payload', {})
                doc_info[doc_id]['payload'] = {**existing_payload, **new_payload}

                existing_metadata = doc_info[doc_id].get('metadata', {})
                doc_info[doc_id]['metadata'] = {**existing_metadata, **new_payload}

        # 점수 기준 정렬
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # 결과 포맷팅
        final_results = []
        for doc_id, score in sorted_docs:
            doc = doc_info[doc_id]
            payload = doc.get('payload', {})
            metadata = doc.get('metadata', {})

            if not metadata and payload:
                metadata = payload.copy()
            if not payload and metadata:
                payload = metadata.copy()

            text = doc.get('text', '') or payload.get('text', '') or metadata.get('text', '')

            result = {
                'id': doc_id,
                'score': score,
                'text': text,
                'metadata': metadata,
                'payload': payload
            }
            final_results.append(result)

        return final_results

    def _weighted_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """가중 평균 방식의 점수 융합"""
        bm25_scores_norm = self._normalize_scores([r['score'] for r in bm25_results])
        vector_scores_norm = self._normalize_scores([r['score'] for r in vector_results])

        combined_scores = {}
        doc_info = {}

        for result, norm_score in zip(bm25_results, bm25_scores_norm):
            doc_id = result['id']
            combined_scores[doc_id] = self.bm25_weight * norm_score
            doc_info[doc_id] = result

        for result, norm_score in zip(vector_results, vector_scores_norm):
            doc_id = result['id']
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + \
                                      self.vector_weight * norm_score

            if doc_id not in doc_info:
                doc_info[doc_id] = result
            else:
                doc_info[doc_id]['payload'] = result.get('payload', {})

        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        final_results = []
        for doc_id, score in sorted_docs:
            result = {
                'id': doc_id,
                'score': score,
                'text': doc_info[doc_id].get('text', ''),
                'metadata': doc_info[doc_id].get('metadata', {}),
                'payload': doc_info[doc_id].get('payload', {})
            }
            final_results.append(result)

        return final_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """점수를 0-1 범위로 정규화"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

        return normalized

    def search_with_filter(
        self,
        query: str,
        query_embedding: np.ndarray,
        filter_conditions: Dict,
        top_k: int = 20,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> List[Dict]:
        """
        필터 조건이 있는 검색
        """
        if self.use_multi_vector and sparse_vector:
            # Multi-Vector + 필터
            results = self.vector_store.search(
                query_vector=query_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter_conditions=filter_conditions,
                use_hybrid=True
            )

            formatted_results = []
            for result in results:
                payload = result.get('payload', {})
                formatted = {
                    'id': result['id'],
                    'score': result['score'],
                    'text': payload.get('text', ''),
                    'metadata': payload,
                    'payload': payload
                }
                formatted_results.append(formatted)
            return formatted_results

        # Legacy 모드
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_conditions=filter_conditions
        )

        if not self.bm25_engine:
            return vector_results

        bm25_results = self.bm25_engine.search(query, top_k=top_k * 2)

        filtered_bm25 = []
        for result in bm25_results:
            match = True
            for key, value in filter_conditions.items():
                if result.get('metadata', {}).get(key) != value:
                    match = False
                    break
            if match:
                filtered_bm25.append(result)

        fused_results = self._reciprocal_rank_fusion(
            filtered_bm25[:top_k],
            vector_results,
            top_k
        )

        return fused_results


def test_hybrid_search():
    """하이브리드 검색 테스트"""
    from ..embeddings.embedding_model import BGEM3Embedder
    import torch

    logger.info("Testing Multi-Vector Hybrid Search Engine")

    # 1. 컴포넌트 초기화 (Multi-Vector 모드)
    vector_store = QdrantVectorStore(
        storage_path="/tmp/test_hybrid_multi_vector",
        collection_name="test_hybrid",
        recreate_collection=True,
        use_sparse=True
    )

    bm25_engine = BM25SearchEngine(use_korean_tokenizer=True)

    embedder = BGEM3Embedder(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. 테스트 데이터
    documents = [
        "2024년 상반기 매출 실적 보고서입니다. 목표 대비 120% 달성하였습니다.",
        "고객 만족도 향상을 위한 전략 계획서입니다. 서비스 품질을 개선합니다.",
        "AI 기반 자동화 시스템 개발 프로젝트 제안서입니다.",
        "마케팅 캠페인 결과 분석 보고서입니다. ROI는 150%입니다.",
        "신입 사원 채용 계획 및 인재 육성 전략입니다."
    ]

    doc_ids = [f"doc_{i}" for i in range(len(documents))]

    metadata = [
        {"department": "재무팀", "file_type": "pdf"},
        {"department": "고객서비스팀", "file_type": "docx"},
        {"department": "개발팀", "file_type": "pptx"},
        {"department": "마케팅팀", "file_type": "pdf"},
        {"department": "인사팀", "file_type": "docx"}
    ]

    # 3. Multi-Vector 임베딩 생성 및 인덱싱
    logger.info("\nMulti-Vector 인덱싱 중...")
    embeddings = embedder.encode_for_indexing(documents)
    dense_vecs = embeddings['dense_vecs']
    sparse_vecs = embeddings['sparse_vecs']

    payloads = [
        {**meta, 'text': doc}
        for meta, doc in zip(metadata, documents)
    ]

    vector_store.add_documents(doc_ids, dense_vecs, payloads, sparse_vecs)

    # BM25 인덱싱 (fallback용)
    bm25_engine.index_documents(documents, doc_ids, metadata)

    # 4. 하이브리드 검색 엔진 생성 (Multi-Vector 모드)
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        bm25_engine=bm25_engine,
        bm25_weight=0.4,
        vector_weight=0.6,
        use_multi_vector=True
    )

    # 5. 검색 테스트
    test_query = "매출 보고서"
    logger.info(f"\n검색 쿼리: '{test_query}'")

    # 쿼리 임베딩 (Dense + Sparse)
    query_result = embedder.encode_query_for_search(test_query)
    query_dense = query_result['dense_vec']
    query_sparse = query_result['sparse_vec']

    results = hybrid_engine.search(
        query=test_query,
        query_embedding=query_dense,
        sparse_vector=query_sparse,
        top_k=20,
        final_top_k=3
    )

    logger.info(f"\n검색 결과 (Multi-Vector):")
    for i, result in enumerate(results):
        logger.info(f"{i+1}. [Score: {result['score']:.4f}]")
        logger.info(f"   Text: {result['payload'].get('text', '')[:60]}...")
        logger.info(f"   Type: {result.get('explanation', {}).get('search_type', [])}")


if __name__ == "__main__":
    test_hybrid_search()
