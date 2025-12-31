"""
Qdrant 벡터 데이터베이스 관리
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant 벡터 데이터베이스 관리 클래스"""

    def __init__(
        self,
        storage_path: str,
        collection_name: str = "company_files",
        vector_size: int = 1024,
        distance: str = "Cosine",
        recreate_collection: bool = False
    ):
        """
        Args:
            storage_path: Qdrant 저장 경로
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원
            distance: 거리 메트릭 (Cosine, Euclidean, Dot)
            recreate_collection: 컬렉션 재생성 여부
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = self._parse_distance(distance)

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
        """컬렉션 설정"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if recreate and collection_exists:
            logger.info(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def add_documents(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict]
    ) -> bool:
        """
        문서를 벡터 DB에 추가

        Args:
            ids: 문서 ID 리스트
            vectors: 임베딩 벡터 (n, dim)
            payloads: 메타데이터 리스트

        Returns:
            성공 여부
        """
        try:
            points = []
            for id_, vector, payload in zip(ids, vectors, payloads):
                point = PointStruct(
                    id=id_,
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
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
            return False

    def add_document(
        self,
        doc_id: str,
        vector: Union[np.ndarray, List[float]],
        payload: Dict
    ) -> bool:
        """
        단일 문서 추가

        Args:
            doc_id: 문서 ID
            vector: 임베딩 벡터
            payload: 메타데이터

        Returns:
            성공 여부
        """
        return self.add_documents([doc_id], [vector], [payload])

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False
    ) -> List[Dict]:
        """
        벡터 유사도 검색

        Args:
            query_vector: 쿼리 벡터
            top_k: 반환할 결과 수
            filter_conditions: 필터 조건
            score_threshold: 점수 임계값
            with_vectors: 결과에 벡터 포함 여부

        Returns:
            검색 결과 리스트
            [
                {
                    'id': str,
                    'score': float,
                    'payload': dict,
                    'vector': list (optional)
                }
            ]
        """
        try:
            # 벡터를 리스트로 변환
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            # 필터 생성
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)

            # 검색 수행 (Qdrant client 1.x API)
            # 단일 벡터 컬렉션의 경우 using="" 파라미터 사용
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,  # 리스트로 직접 전달
                using="",  # 단일 unnamed 벡터 사용
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors
            ).points

            # 결과 포맷팅
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                }
                # 벡터 포함 옵션
                if with_vectors and hasattr(hit, 'vector') and hit.vector is not None:
                    result['vector'] = hit.vector
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def batch_search(
        self,
        query_vectors: List[Union[np.ndarray, List[float]]],
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        배치 검색

        Args:
            query_vectors: 쿼리 벡터 리스트
            top_k: 각 쿼리당 반환할 결과 수

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        all_results = []

        for query_vector in query_vectors:
            results = self.search(query_vector, top_k=top_k)
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
        """
        문서 삭제

        Args:
            ids: 삭제할 문서 ID 리스트

        Returns:
            성공 여부
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            logger.info(f"Deleted {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        문서 조회

        Args:
            doc_id: 문서 ID

        Returns:
            문서 정보 또는 None
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )

            if result:
                return {
                    'id': result[0].id,
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
        """컬렉션 내 모든 문서 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection(recreate=False)
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


def test_vector_store():
    """벡터 스토어 테스트"""
    logger.info("Testing QdrantVectorStore")

    # 테스트 스토어 생성
    store = QdrantVectorStore(
        storage_path="/tmp/test_qdrant",
        collection_name="test_collection",
        vector_size=1024,
        recreate_collection=True
    )

    # 테스트 데이터
    test_ids = ["doc1", "doc2", "doc3"]
    test_vectors = np.random.rand(3, 1024).astype(np.float32)
    test_payloads = [
        {"text": "문서 1", "type": "pdf"},
        {"text": "문서 2", "type": "docx"},
        {"text": "문서 3", "type": "pptx"}
    ]

    # 문서 추가
    logger.info("\n1. Adding documents")
    success = store.add_documents(test_ids, test_vectors, test_payloads)
    logger.info(f"Add success: {success}")
    logger.info(f"Document count: {store.count_documents()}")

    # 검색
    logger.info("\n2. Searching")
    query_vector = np.random.rand(1024).astype(np.float32)
    results = store.search(query_vector, top_k=2)

    for i, result in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Payload: {result['payload']}")

    # 문서 조회
    logger.info("\n3. Retrieving document")
    doc = store.get_document("doc1")
    if doc:
        logger.info(f"Retrieved doc: {doc['payload']}")

    # 삭제
    logger.info("\n4. Deleting documents")
    store.delete_documents(["doc1"])
    logger.info(f"Document count after delete: {store.count_documents()}")


if __name__ == "__main__":
    test_vector_store()
