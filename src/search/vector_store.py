"""
Qdrant 벡터 데이터베이스 관리
Multi-Vector 지원 (Dense + Sparse)
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint,
    SparseVectorParams,
    SparseVector,
    NamedSparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Query
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant 벡터 데이터베이스 관리 클래스
    Multi-Vector 지원 (Dense + Sparse Named Vectors)
    """

    def __init__(
        self,
        storage_path: str,
        collection_name: str = "company_files",
        vector_size: int = 1024,
        distance: str = "Cosine",
        recreate_collection: bool = False,
        use_sparse: bool = True
    ):
        """
        Args:
            storage_path: Qdrant 저장 경로
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원
            distance: 거리 메트릭 (Cosine, Euclidean, Dot)
            recreate_collection: 컬렉션 재생성 여부
            use_sparse: Sparse 벡터 사용 여부 (Multi-Vector)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = self._parse_distance(distance)
        self.use_sparse = use_sparse

        # Qdrant 클라이언트 초기화
        logger.info(f"Initializing Qdrant client at {self.storage_path}")
        self.client = QdrantClient(path=str(self.storage_path))

        # 컬렉션 설정
        self._setup_collection(recreate_collection)

    def _parse_distance(self, distance: str) -> Distance:
        """거리 메트릭 파싱"""
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        return distance_map.get(distance, Distance.COSINE)

    def _setup_collection(self, recreate: bool = False):
        """컬렉션 설정 (Multi-Vector 지원)"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if recreate and collection_exists:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            logger.info(f"Creating collection: {self.collection_name} (use_sparse={self.use_sparse})")

            if self.use_sparse:
                # Multi-Vector: Dense + Sparse Named Vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_size,
                            distance=self.distance
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
                logger.info("Created Multi-Vector collection with Dense + Sparse vectors")
            else:
                # 기존 방식: 단일 Dense 벡터
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info("Created single Dense vector collection")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def _string_to_int_id(self, string_id: str) -> int:
        """문자열 ID를 정수 ID로 변환 (Qdrant 호환)"""
        hash_hex = hashlib.md5(string_id.encode()).hexdigest()[:16]
        return int(hash_hex, 16)

    def add_documents(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict],
        sparse_vectors: Optional[List[Dict[int, float]]] = None
    ) -> bool:
        """
        문서를 벡터 DB에 추가 (Multi-Vector 지원)

        Args:
            ids: 문서 ID 리스트
            vectors: Dense 임베딩 벡터 (n, dim)
            payloads: 메타데이터 리스트
            sparse_vectors: Sparse 벡터 리스트 [{token_id: weight}, ...]

        Returns:
            성공 여부
        """
        try:
            points = []
            for i, (id_, vector, payload) in enumerate(zip(ids, vectors, payloads)):
                int_id = self._string_to_int_id(id_)
                payload['original_id'] = id_

                # Dense 벡터 변환
                dense_vec = vector.tolist() if isinstance(vector, np.ndarray) else vector

                if self.use_sparse and sparse_vectors and i < len(sparse_vectors):
                    # Multi-Vector: Named Vectors 사용
                    sparse_dict = sparse_vectors[i]
                    if sparse_dict:
                        indices = list(sparse_dict.keys())
                        values = list(sparse_dict.values())
                        sparse_vec = SparseVector(indices=indices, values=values)

                        point = PointStruct(
                            id=int_id,
                            vector={
                                "dense": dense_vec,
                                "sparse": sparse_vec
                            },
                            payload=payload
                        )
                    else:
                        # Sparse가 비어있으면 Dense만
                        point = PointStruct(
                            id=int_id,
                            vector={"dense": dense_vec},
                            payload=payload
                        )
                else:
                    # 기존 방식: 단일 벡터
                    if self.use_sparse:
                        point = PointStruct(
                            id=int_id,
                            vector={"dense": dense_vec},
                            payload=payload
                        )
                    else:
                        point = PointStruct(
                            id=int_id,
                            vector=dense_vec,
                            payload=payload
                        )

                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(points)} documents to {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_document(
        self,
        doc_id: str,
        vector: Union[np.ndarray, List[float]],
        payload: Dict,
        sparse_vector: Optional[Dict[int, float]] = None
    ) -> bool:
        """단일 문서 추가"""
        sparse_vectors = [sparse_vector] if sparse_vector else None
        return self.add_documents(
            [doc_id],
            np.array([vector]) if isinstance(vector, list) else vector.reshape(1, -1),
            [payload],
            sparse_vectors
        )

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
        sparse_vector: Optional[Dict[int, float]] = None,
        use_hybrid: bool = True,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> List[Dict]:
        """
        벡터 유사도 검색 (Multi-Vector 하이브리드 검색 지원)

        Args:
            query_vector: Dense 쿼리 벡터
            top_k: 반환할 결과 수
            filter_conditions: 필터 조건
            score_threshold: 점수 임계값
            with_vectors: 결과에 벡터 포함 여부
            sparse_vector: Sparse 쿼리 벡터 (Multi-Vector)
            use_hybrid: Dense + Sparse 하이브리드 검색 사용
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse 검색 가중치

        Returns:
            검색 결과 리스트
        """
        try:
            # Dense 벡터 변환
            if isinstance(query_vector, np.ndarray):
                if len(query_vector.shape) > 1:
                    query_vector = query_vector.flatten()
                query_vector = query_vector.tolist()

            # 필터 생성
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)

            # Multi-Vector 하이브리드 검색
            if self.use_sparse and sparse_vector and use_hybrid:
                return self._hybrid_search(
                    dense_vector=query_vector,
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    query_filter=query_filter,
                    with_vectors=with_vectors,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )

            # 단일 Dense 검색
            if self.use_sparse:
                # Named Vector 사용
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense",
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=with_vectors
                ).points
            else:
                # 기존 방식
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=with_vectors
                ).points

            return self._format_results(search_result, with_vectors)

        except Exception as e:
            logger.error(f"Error searching: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        top_k: int = 10,
        query_filter: Optional[Filter] = None,
        with_vectors: bool = False,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> List[Dict]:
        """
        Qdrant 내장 하이브리드 검색 (RRF Fusion)

        Args:
            dense_vector: Dense 쿼리 벡터
            sparse_vector: Sparse 쿼리 벡터
            top_k: 반환할 결과 수
            query_filter: 필터 조건
            with_vectors: 벡터 포함 여부
            dense_weight: Dense 가중치
            sparse_weight: Sparse 가중치

        Returns:
            검색 결과 리스트
        """
        try:
            # Sparse 벡터 변환
            sparse_indices = list(sparse_vector.keys())
            sparse_values = list(sparse_vector.values())

            # Prefetch로 각 벡터 타입별 검색 후 RRF Fusion
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k * 2
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=top_k * 2
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=with_vectors
            ).points

            return self._format_results(search_result, with_vectors)

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            # Fallback to dense-only search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=with_vectors
            ).points
            return self._format_results(search_result, with_vectors)

    def search_sparse_only(
        self,
        sparse_vector: Dict[int, float],
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """Sparse 벡터만으로 검색 (키워드 검색)"""
        if not self.use_sparse:
            logger.warning("Sparse search not available (use_sparse=False)")
            return []

        try:
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)

            sparse_indices = list(sparse_vector.keys())
            sparse_values = list(sparse_vector.values())

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k,
                query_filter=query_filter,
                with_payload=True
            ).points

            return self._format_results(search_result)

        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    def _format_results(self, search_result, with_vectors: bool = False) -> List[Dict]:
        """검색 결과 포맷팅"""
        results = []
        for hit in search_result:
            original_id = hit.payload.get('original_id', str(hit.id))
            result = {
                'id': original_id,
                'score': hit.score,
                'payload': hit.payload
            }
            if with_vectors and hasattr(hit, 'vector') and hit.vector is not None:
                result['vector'] = hit.vector
            results.append(result)
        return results

    def batch_search(
        self,
        query_vectors: List[Union[np.ndarray, List[float]]],
        top_k: int = 10,
        sparse_vectors: Optional[List[Dict[int, float]]] = None
    ) -> List[List[Dict]]:
        """배치 검색"""
        all_results = []
        for i, query_vector in enumerate(query_vectors):
            sparse_vec = sparse_vectors[i] if sparse_vectors and i < len(sparse_vectors) else None
            results = self.search(query_vector, top_k=top_k, sparse_vector=sparse_vec)
            all_results.append(results)
        return all_results

    def _build_filter(self, conditions: Dict) -> Filter:
        """필터 조건 생성"""
        field_conditions = []
        for key, value in conditions.items():
            field_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        return Filter(must=field_conditions)

    def delete_documents(self, ids: List[str]) -> bool:
        """문서 삭제"""
        try:
            int_ids = [self._string_to_int_id(id_) for id_ in ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=int_ids
            )
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """문서 조회"""
        try:
            int_id = self._string_to_int_id(doc_id)
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[int_id]
            )
            if result:
                return {
                    'id': result[0].payload.get('original_id', str(result[0].id)),
                    'vector': result[0].vector,
                    'payload': result[0].payload
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None

    def count_documents(self) -> int:
        """컬렉션 내 문서 수 조회"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    def clear_collection(self) -> bool:
        """컬렉션 내 모든 문서 삭제 및 재생성"""
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection(recreate=False)
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'status': str(info.status),
                'use_sparse': self.use_sparse
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


def test_multi_vector_store():
    """Multi-Vector 스토어 테스트"""
    logger.info("Testing Multi-Vector QdrantVectorStore")

    # Multi-Vector 스토어 생성
    store = QdrantVectorStore(
        storage_path="/tmp/test_multi_vector_qdrant",
        collection_name="test_multi_vector",
        vector_size=1024,
        recreate_collection=True,
        use_sparse=True
    )

    # 테스트 데이터
    test_ids = ["doc1", "doc2", "doc3"]
    test_dense = np.random.rand(3, 1024).astype(np.float32)
    test_sparse = [
        {100: 0.8, 200: 0.6, 300: 0.4},  # 토큰 ID: 가중치
        {150: 0.9, 250: 0.5},
        {100: 0.7, 350: 0.8, 400: 0.3}
    ]
    test_payloads = [
        {"text": "2024년 매출 보고서", "type": "pdf"},
        {"text": "마케팅 전략 계획", "type": "docx"},
        {"text": "AI 프로젝트 제안서", "type": "pptx"}
    ]

    # 문서 추가
    logger.info("\n1. Adding documents with Multi-Vector")
    success = store.add_documents(test_ids, test_dense, test_payloads, test_sparse)
    logger.info(f"Add success: {success}")
    logger.info(f"Document count: {store.count_documents()}")
    logger.info(f"Collection info: {store.get_collection_info()}")

    # Dense 검색
    logger.info("\n2. Dense-only search")
    query_dense = np.random.rand(1024).astype(np.float32)
    results = store.search(query_dense, top_k=2, use_hybrid=False)
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result['payload']['text']} (score: {result['score']:.4f})")

    # 하이브리드 검색
    logger.info("\n3. Hybrid search (Dense + Sparse)")
    query_sparse = {100: 0.9, 200: 0.7}
    results = store.search(query_dense, top_k=2, sparse_vector=query_sparse, use_hybrid=True)
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result['payload']['text']} (score: {result['score']:.4f})")

    # Sparse 검색
    logger.info("\n4. Sparse-only search")
    results = store.search_sparse_only(query_sparse, top_k=2)
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result['payload']['text']} (score: {result['score']:.4f})")

    logger.info("\nMulti-Vector test completed!")


if __name__ == "__main__":
    test_multi_vector_store()
