"""
BM25 키워드 기반 검색 엔진
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KoreanTokenizer:
    """한국어 형태소 분석기"""

    def __init__(self):
        """Kiwi 형태소 분석기 초기화"""
        logger.info("Initializing Korean tokenizer (Kiwi)")
        self.kiwi = Kiwi()

    def tokenize(self, text: str) -> List[str]:
        """
        텍스트를 토큰으로 분할

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        if not text:
            return []

        # 형태소 분석
        tokens = self.kiwi.tokenize(text)

        # 명사, 동사, 형용사만 추출
        keywords = []
        for token in tokens:
            if token.tag in ['NNG', 'NNP', 'VV', 'VA', 'SL']:  # 명사, 동사, 형용사, 외래어
                keywords.append(token.form)

        return keywords


class BM25SearchEngine:
    """BM25 기반 키워드 검색 엔진"""

    def __init__(self, use_korean_tokenizer: bool = True):
        """
        Args:
            use_korean_tokenizer: 한국어 토크나이저 사용 여부
        """
        self.use_korean_tokenizer = use_korean_tokenizer
        self.tokenizer = KoreanTokenizer() if use_korean_tokenizer else None

        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.document_metadata = []

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # 기본 공백 분리
            return text.lower().split()

    def index_documents(
        self,
        documents: List[str],
        document_ids: List[str],
        metadata: List[Dict] = None
    ):
        """
        문서 인덱싱

        Args:
            documents: 문서 텍스트 리스트
            document_ids: 문서 ID 리스트
            metadata: 문서 메타데이터 리스트
        """
        logger.info(f"Indexing {len(documents)} documents for BM25")

        self.documents = documents
        self.document_ids = document_ids
        self.document_metadata = metadata or [{} for _ in documents]

        # 문서 토큰화
        tokenized_corpus = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_corpus.append(tokens)

        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 indexing complete")

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            return_scores: 점수 포함 여부

        Returns:
            검색 결과 리스트
            [
                {
                    'id': str,
                    'score': float,
                    'text': str,
                    'metadata': dict
                }
            ]
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return []

        # 쿼리 토큰화
        query_tokens = self._tokenize(query)

        # BM25 점수 계산
        scores = self.bm25.get_scores(query_tokens)

        # 상위 k개 결과 추출
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = {
                'id': self.document_ids[idx],
                'score': float(scores[idx]),
                'text': self.documents[idx],
                'metadata': self.document_metadata[idx]
            }
            results.append(result)

        return results

    def search_filenames(
        self,
        query: str,
        filenames: List[str],
        file_ids: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        파일명 기반 검색

        Args:
            query: 검색 쿼리
            filenames: 파일명 리스트
            file_ids: 파일 ID 리스트
            top_k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        # 파일명으로 임시 인덱스 생성
        temp_bm25 = self.bm25
        temp_docs = self.documents
        temp_ids = self.document_ids
        temp_meta = self.document_metadata

        # 파일명 인덱싱
        self.index_documents(
            documents=filenames,
            document_ids=file_ids,
            metadata=[{"filename": fn} for fn in filenames]
        )

        # 검색
        results = self.search(query, top_k=top_k)

        # 원래 인덱스 복원
        self.bm25 = temp_bm25
        self.documents = temp_docs
        self.document_ids = temp_ids
        self.document_metadata = temp_meta

        return results

    def save_index(self, filepath: str):
        """
        BM25 인덱스 저장

        Args:
            filepath: 저장 경로
        """
        data = {
            'bm25': self.bm25,
            'documents': self.documents,
            'document_ids': self.document_ids,
            'document_metadata': self.document_metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {filepath}")

    def load_index(self, filepath: str):
        """
        BM25 인덱스 로드

        Args:
            filepath: 로드 경로
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.documents = data['documents']
        self.document_ids = data['document_ids']
        self.document_metadata = data['document_metadata']

        logger.info(f"BM25 index loaded from {filepath}")


def test_bm25_search():
    """BM25 검색 엔진 테스트"""
    logger.info("Testing BM25 Search Engine")

    # 검색 엔진 초기화
    search_engine = BM25SearchEngine(use_korean_tokenizer=True)

    # 테스트 문서
    documents = [
        "2024년 상반기 매출 실적 보고서입니다. 목표 대비 120% 달성하였습니다.",
        "고객 만족도 향상을 위한 전략 계획서입니다. 서비스 품질을 개선합니다.",
        "AI 기반 자동화 시스템 개발 프로젝트 제안서입니다.",
        "마케팅 캠페인 결과 분석 보고서입니다. ROI는 150%입니다.",
        "신입 사원 채용 계획 및 인재 육성 전략입니다."
    ]

    document_ids = [f"doc_{i}" for i in range(len(documents))]

    metadata = [
        {"department": "재무팀", "type": "보고서"},
        {"department": "고객서비스팀", "type": "계획서"},
        {"department": "개발팀", "type": "제안서"},
        {"department": "마케팅팀", "type": "보고서"},
        {"department": "인사팀", "type": "계획서"}
    ]

    # 인덱싱
    logger.info("\n1. Indexing documents")
    search_engine.index_documents(documents, document_ids, metadata)

    # 검색 테스트
    test_queries = [
        "매출 보고서",
        "고객 만족도",
        "AI 개발",
        "마케팅 분석"
    ]

    for query in test_queries:
        logger.info(f"\n검색 쿼리: '{query}'")
        results = search_engine.search(query, top_k=3)

        for i, result in enumerate(results):
            logger.info(f"  {i+1}. [Score: {result['score']:.2f}] {result['text'][:50]}...")
            logger.info(f"     메타데이터: {result['metadata']}")


if __name__ == "__main__":
    test_bm25_search()
