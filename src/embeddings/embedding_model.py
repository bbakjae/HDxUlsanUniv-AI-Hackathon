"""
임베딩 모델 - BGE-M3 기반 텍스트 임베딩 생성
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """임베딩 생성 베이스 클래스"""

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        raise NotImplementedError

    def encode_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """배치 단위로 임베딩 생성"""
        raise NotImplementedError


class BGEM3Embedder(EmbeddingModel):
    """
    BGE-M3 임베딩 모델
    - Dense 벡터 (의미적 유사도)
    - Sparse 벡터 (키워드 매칭)
    - Multi-vector (ColBERT 스타일)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        max_length: int = 8192,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: 모델 이름
            device: 디바이스 (cuda/cpu)
            use_fp16: FP16 사용 여부
            max_length: 최대 입력 길이
            cache_dir: 모델 캐시 디렉토리
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.max_length = max_length
        self.cache_dir = cache_dir

        # 모델 로딩
        logger.info(f"Loading BGE-M3 model: {model_name}")
        self.model = self._load_model()
        logger.info(f"Model loaded successfully on {self.device}")

    def _load_model(self) -> BGEM3FlagModel:
        """BGE-M3 모델 로딩"""
        try:
            # safetensors 사용 강제 (PyTorch 2.6 미만 호환성)
            import os
            os.environ['SAFETENSORS_FAST_GPU'] = '1'

            model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
                model_kwargs={'use_safetensors': True}
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        batch_size: int = 32,
        max_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            texts: 입력 텍스트 (단일 또는 리스트)
            return_dense: Dense 벡터 반환 여부
            return_sparse: Sparse 벡터 반환 여부
            return_colbert_vecs: ColBERT 벡터 반환 여부
            batch_size: 배치 크기
            max_length: 최대 길이 (None이면 기본값 사용)

        Returns:
            {
                'dense': np.ndarray,  # (batch_size, 1024)
                'sparse': List[Dict],  # 희소 벡터
                'colbert': np.ndarray  # (batch_size, seq_len, dim)
            }
        """
        # 단일 텍스트를 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]

        max_len = max_length or self.max_length

        try:
            # 임베딩 생성
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_len,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert_vecs
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        대용량 텍스트를 배치로 인코딩

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            **kwargs: encode 메서드에 전달할 추가 인자

        Returns:
            임베딩 딕셔너리
        """
        return self.encode(texts, batch_size=batch_size, **kwargs)

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        검색 쿼리 인코딩 (Dense 벡터만)

        Args:
            queries: 쿼리 텍스트
            batch_size: 배치 크기

        Returns:
            Dense 임베딩 벡터
        """
        result = self.encode(
            queries,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=batch_size
        )

        return result['dense_vecs']

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        include_sparse: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        문서 인코딩 (Dense + Sparse)

        Args:
            documents: 문서 리스트
            batch_size: 배치 크기
            include_sparse: Sparse 벡터 포함 여부

        Returns:
            {
                'dense': np.ndarray,
                'sparse': List[Dict] (optional)
            }
        """
        return self.encode(
            documents,
            return_dense=True,
            return_sparse=include_sparse,
            return_colbert_vecs=False,
            batch_size=batch_size
        )

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        method: str = "cosine"
    ) -> np.ndarray:
        """
        쿼리와 문서 간 유사도 계산

        Args:
            query_embedding: 쿼리 임베딩 (1, dim) 또는 (dim,)
            doc_embeddings: 문서 임베딩 (n, dim)
            method: 유사도 계산 방법 (cosine)

        Returns:
            유사도 점수 배열 (n,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if method == "cosine":
            # 코사인 유사도
            query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            similarity = np.dot(doc_norm, query_norm.T).squeeze()
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        return similarity

    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return 1024  # BGE-M3의 dense 벡터 차원


class CachedEmbedder:
    """
    캐싱 기능이 있는 임베딩 생성기
    동일한 텍스트에 대해 중복 계산 방지
    """

    def __init__(self, embedder: EmbeddingModel, cache_size: int = 10000):
        """
        Args:
            embedder: 베이스 임베딩 모델
            cache_size: 캐시 크기
        """
        self.embedder = embedder
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """캐시를 사용한 인코딩"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        texts_to_encode = []
        indices_to_encode = []

        # 캐시 확인
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                embeddings.append(None)  # 플레이스홀더

        # 캐시되지 않은 텍스트 인코딩
        if texts_to_encode:
            new_embeddings = self.embedder.encode(texts_to_encode, **kwargs)

            if isinstance(new_embeddings, dict) and 'dense_vecs' in new_embeddings:
                new_embeddings = new_embeddings['dense_vecs']

            # 캐시 업데이트
            for text, emb in zip(texts_to_encode, new_embeddings):
                self.cache[text] = emb

                # 캐시 크기 제한
                if len(self.cache) > self.cache_size:
                    # 가장 오래된 항목 제거 (간단 구현)
                    first_key = next(iter(self.cache))
                    del self.cache[first_key]

            # 결과 배열 채우기
            for idx, emb in zip(indices_to_encode, new_embeddings):
                embeddings[idx] = emb

        return np.array(embeddings)


def test_embedder():
    """임베딩 모델 테스트"""
    logger.info("Testing BGE-M3 Embedder")

    embedder = BGEM3Embedder(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True
    )

    # 테스트 텍스트
    test_texts = [
        "2024년 상반기 매출 실적 보고서입니다.",
        "고객 만족도 향상을 위한 전략 계획",
        "AI 기반 자동화 시스템 개발 프로젝트"
    ]

    # Dense 임베딩 생성
    logger.info("\n1. Dense embedding test")
    dense_result = embedder.encode(
        test_texts,
        return_dense=True,
        return_sparse=False
    )
    logger.info(f"Dense shape: {dense_result['dense_vecs'].shape}")
    logger.info(f"Embedding dim: {embedder.get_embedding_dim()}")

    # Sparse 임베딩 생성
    logger.info("\n2. Sparse embedding test")
    sparse_result = embedder.encode(
        test_texts,
        return_dense=False,
        return_sparse=True
    )
    logger.info(f"Sparse result keys: {sparse_result.keys()}")

    # 유사도 계산
    logger.info("\n3. Similarity test")
    query = "매출 보고서"
    query_emb = embedder.encode_queries(query)
    doc_embs = dense_result['dense_vecs']

    similarities = embedder.compute_similarity(query_emb, doc_embs)
    logger.info(f"Similarities: {similarities}")

    # 가장 유사한 문서
    most_similar_idx = np.argmax(similarities)
    logger.info(f"\nMost similar document: {test_texts[most_similar_idx]}")
    logger.info(f"Similarity score: {similarities[most_similar_idx]:.4f}")


if __name__ == "__main__":
    test_embedder()
