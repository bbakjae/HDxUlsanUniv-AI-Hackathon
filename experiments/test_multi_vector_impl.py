"""
Multi-Vector 구현 테스트 스크립트
기존 데이터를 사용하여 Multi-Vector 검색 기능 검증
"""

import sys
from pathlib import Path
import logging
import time
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_embedding_model():
    """임베딩 모델의 Sparse 벡터 생성 기능 테스트"""
    logger.info("=" * 60)
    logger.info("1. 임베딩 모델 Sparse 벡터 생성 테스트")
    logger.info("=" * 60)

    from src.embeddings.embedding_model import BGEM3Embedder

    # 임베딩 모델 초기화
    embedder = BGEM3Embedder(
        model_name="dragonkue/BGE-m3-ko",
        device="cuda",
        use_fp16=True
    )

    # 테스트 텍스트
    test_texts = [
        "2024년 상반기 매출 실적 보고서입니다.",
        "고객 만족도 향상을 위한 전략 계획",
        "AI 기반 자동화 시스템 개발 프로젝트"
    ]

    # encode_for_indexing 테스트
    logger.info("\n[encode_for_indexing 테스트]")
    result = embedder.encode_for_indexing(test_texts)

    logger.info(f"Dense vectors shape: {result['dense_vecs'].shape}")
    logger.info(f"Sparse vectors count: {len(result['sparse_vecs'])}")

    for i, sparse in enumerate(result['sparse_vecs']):
        logger.info(f"  Text {i}: sparse vector with {len(sparse)} tokens")

    # encode_query_for_search 테스트
    logger.info("\n[encode_query_for_search 테스트]")
    query = "매출 보고서"
    query_result = embedder.encode_query_for_search(query)

    logger.info(f"Query dense vector shape: {query_result['dense_vec'].shape}")
    logger.info(f"Query sparse vector tokens: {len(query_result['sparse_vec'])}")

    return True


def test_vector_store_multi_vector():
    """Vector Store의 Multi-Vector 기능 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("2. Vector Store Multi-Vector 기능 테스트")
    logger.info("=" * 60)

    from src.search.vector_store import QdrantVectorStore
    from src.embeddings.embedding_model import BGEM3Embedder

    # 테스트용 저장 경로
    test_storage = str(project_root / "experiments" / "test_multi_vector_storage")

    # Multi-Vector 모드로 Vector Store 초기화
    logger.info("\n[Multi-Vector 모드 Vector Store 초기화]")
    vector_store = QdrantVectorStore(
        storage_path=test_storage,
        collection_name="test_multi_vector",
        vector_size=1024,
        distance="Cosine",
        use_sparse=True  # Multi-Vector 활성화
    )

    # 컬렉션 재생성
    vector_store.clear_collection()

    # 임베딩 모델 초기화
    embedder = BGEM3Embedder(
        model_name="dragonkue/BGE-m3-ko",
        device="cuda",
        use_fp16=True
    )

    # 테스트 문서
    test_docs = [
        {"id": "doc1", "text": "2024년 상반기 매출 실적 보고서입니다. 전년 대비 15% 성장을 기록했습니다."},
        {"id": "doc2", "text": "고객 만족도 향상을 위한 전략 계획입니다. CS 팀의 응대 품질 개선이 필요합니다."},
        {"id": "doc3", "text": "AI 기반 자동화 시스템 개발 프로젝트 제안서입니다. 딥러닝 모델을 활용합니다."},
        {"id": "doc4", "text": "2024년 하반기 마케팅 예산 계획서입니다. 디지털 마케팅 비중을 확대합니다."},
        {"id": "doc5", "text": "신규 제품 출시 일정표입니다. Q3에 3개 제품을 출시 예정입니다."}
    ]

    # 문서 인덱싱
    logger.info("\n[Multi-Vector 문서 인덱싱]")
    texts = [doc["text"] for doc in test_docs]
    ids = [doc["id"] for doc in test_docs]

    # Dense + Sparse 벡터 생성
    embeddings_result = embedder.encode_for_indexing(texts)

    # 페이로드 생성
    payloads = [
        {"text": doc["text"], "file_name": f"test_{doc['id']}.txt"}
        for doc in test_docs
    ]

    # 문서 추가 (Multi-Vector)
    vector_store.add_documents(
        ids=ids,
        vectors=embeddings_result['dense_vecs'],
        payloads=payloads,
        sparse_vectors=embeddings_result['sparse_vecs']
    )

    logger.info(f"인덱싱 완료: {vector_store.count_documents()} 문서")

    # 검색 테스트
    logger.info("\n[Multi-Vector 하이브리드 검색 테스트]")
    test_queries = [
        "매출 실적 보고서",
        "AI 자동화 시스템",
        "마케팅 예산"
    ]

    for query in test_queries:
        logger.info(f"\n쿼리: '{query}'")

        # 쿼리 벡터 생성
        query_result = embedder.encode_query_for_search(query)

        # 하이브리드 검색 (Dense + Sparse RRF)
        start_time = time.time()
        results = vector_store.search(
            query_vector=query_result['dense_vec'],
            sparse_vector=query_result['sparse_vec'],
            top_k=3,
            use_hybrid=True
        )
        search_time = (time.time() - start_time) * 1000

        logger.info(f"검색 시간: {search_time:.2f}ms")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Score: {result['score']:.4f} - {result['payload']['text'][:50]}...")

    # 정리
    import shutil
    if Path(test_storage).exists():
        shutil.rmtree(test_storage)

    return True


def test_hybrid_search_engine():
    """HybridSearchEngine의 Multi-Vector 통합 테스트"""
    logger.info("\n" + "=" * 60)
    logger.info("3. HybridSearchEngine Multi-Vector 통합 테스트")
    logger.info("=" * 60)

    from src.search.hybrid_search import HybridSearchEngine
    from src.search.vector_store import QdrantVectorStore
    from src.embeddings.embedding_model import BGEM3Embedder

    # 테스트용 저장 경로
    test_storage = str(project_root / "experiments" / "test_hybrid_storage")

    # 컴포넌트 초기화
    vector_store = QdrantVectorStore(
        storage_path=test_storage,
        collection_name="test_hybrid",
        vector_size=1024,
        distance="Cosine",
        use_sparse=True
    )
    vector_store.clear_collection()

    embedder = BGEM3Embedder(
        model_name="dragonkue/BGE-m3-ko",
        device="cuda",
        use_fp16=True
    )

    # 하이브리드 검색 엔진 초기화 (Multi-Vector 모드)
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedder=embedder,
        bm25_engine=None,  # Multi-Vector 모드에서는 불필요
        vector_weight=0.7,
        bm25_weight=0.3,
        use_multi_vector=True  # Multi-Vector 활성화
    )

    # 테스트 문서 인덱싱
    test_docs = [
        {"id": "doc1", "text": "현대자동차 2024년 글로벌 판매 실적 보고서입니다."},
        {"id": "doc2", "text": "전기차 배터리 기술 개발 현황 보고서입니다."},
        {"id": "doc3", "text": "2024년 상반기 재무제표 및 손익계산서입니다."},
        {"id": "doc4", "text": "신규 SUV 모델 디자인 검토 회의록입니다."},
        {"id": "doc5", "text": "글로벌 공급망 관리 전략 문서입니다."}
    ]

    texts = [doc["text"] for doc in test_docs]
    ids = [doc["id"] for doc in test_docs]

    embeddings_result = embedder.encode_for_indexing(texts)
    payloads = [{"text": doc["text"], "file_name": f"{doc['id']}.txt"} for doc in test_docs]

    vector_store.add_documents(
        ids=ids,
        vectors=embeddings_result['dense_vecs'],
        payloads=payloads,
        sparse_vectors=embeddings_result['sparse_vecs']
    )

    logger.info(f"인덱싱 완료: {vector_store.count_documents()} 문서")

    # 검색 테스트
    logger.info("\n[HybridSearchEngine.search_with_sparse 테스트]")
    test_queries = [
        "현대자동차 판매 실적",
        "전기차 배터리",
        "재무제표"
    ]

    for query in test_queries:
        logger.info(f"\n쿼리: '{query}'")

        start_time = time.time()
        results = hybrid_engine.search_with_sparse(query, top_k=3)
        search_time = (time.time() - start_time) * 1000

        logger.info(f"검색 시간: {search_time:.2f}ms")
        for i, result in enumerate(results):
            score = result.get('score', result.get('final_score', 0))
            text = result.get('text', result.get('payload', {}).get('text', ''))[:50]
            logger.info(f"  {i+1}. Score: {score:.4f} - {text}...")

    # 정리
    import shutil
    if Path(test_storage).exists():
        shutil.rmtree(test_storage)

    return True


def test_with_existing_data():
    """기존 Qdrant 저장소 데이터로 검색 성능 비교"""
    logger.info("\n" + "=" * 60)
    logger.info("4. 기존 데이터 로드 및 Multi-Vector 재인덱싱 테스트")
    logger.info("=" * 60)

    from qdrant_client import QdrantClient
    from src.search.vector_store import QdrantVectorStore
    from src.embeddings.embedding_model import BGEM3Embedder
    from src.search.hybrid_search import HybridSearchEngine

    # 기존 데이터 로드
    existing_storage = str(project_root / "qdrant_storage_gdrive")

    if not Path(existing_storage).exists():
        logger.warning(f"기존 저장소 없음: {existing_storage}")
        return False

    logger.info(f"기존 저장소 로드: {existing_storage}")
    existing_client = QdrantClient(path=existing_storage)

    # 기존 컬렉션에서 문서 추출
    collections = existing_client.get_collections().collections
    if not collections:
        logger.warning("컬렉션 없음")
        return False

    collection_name = collections[0].name
    logger.info(f"컬렉션: {collection_name}")

    # 기존 문서 가져오기 (최대 100개)
    existing_docs = existing_client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )[0]

    logger.info(f"기존 문서 수: {len(existing_docs)}")

    if len(existing_docs) == 0:
        logger.warning("문서 없음")
        return False

    # 테스트용 Multi-Vector 저장소 생성
    test_storage = str(project_root / "experiments" / "test_existing_mv_storage")

    embedder = BGEM3Embedder(
        model_name="dragonkue/BGE-m3-ko",
        device="cuda",
        use_fp16=True
    )

    vector_store = QdrantVectorStore(
        storage_path=test_storage,
        collection_name="test_existing_mv",
        vector_size=1024,
        distance="Cosine",
        use_sparse=True
    )
    vector_store.clear_collection()

    # 기존 문서 재인덱싱 (Multi-Vector)
    logger.info("\n[기존 문서 Multi-Vector 재인덱싱]")
    texts = []
    ids = []
    payloads = []

    for doc in existing_docs[:50]:  # 50개만 테스트
        text = doc.payload.get('text', '')
        if text:
            texts.append(text)
            ids.append(str(doc.id))
            payloads.append(doc.payload)

    if texts:
        embeddings_result = embedder.encode_for_indexing(texts)
        vector_store.add_documents(
            ids=ids,
            vectors=embeddings_result['dense_vecs'],
            payloads=payloads,
            sparse_vectors=embeddings_result['sparse_vecs']
        )
        logger.info(f"Multi-Vector 재인덱싱 완료: {vector_store.count_documents()} 문서")

    # 검색 성능 테스트
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedder=embedder,
        bm25_engine=None,
        vector_weight=0.7,
        bm25_weight=0.3,
        use_multi_vector=True
    )

    test_queries = [
        "매출 실적",
        "프로젝트 계획",
        "회의록"
    ]

    logger.info("\n[Multi-Vector 검색 성능 테스트]")
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        results = hybrid_engine.search_with_sparse(query, top_k=5)
        search_time = (time.time() - start_time) * 1000
        total_time += search_time

        logger.info(f"\n쿼리: '{query}' - {search_time:.2f}ms")
        for i, result in enumerate(results[:3]):
            score = result.get('score', result.get('final_score', 0))
            text = result.get('text', result.get('payload', {}).get('text', ''))[:40]
            logger.info(f"  {i+1}. Score: {score:.4f} - {text}...")

    avg_time = total_time / len(test_queries)
    logger.info(f"\n평균 검색 시간: {avg_time:.2f}ms")

    # 정리
    import shutil
    if Path(test_storage).exists():
        shutil.rmtree(test_storage)

    return True


def main():
    """메인 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Multi-Vector 구현 통합 테스트")
    logger.info("=" * 60)

    results = {}

    # 1. 임베딩 모델 테스트
    try:
        results['embedding_model'] = test_embedding_model()
    except Exception as e:
        logger.error(f"임베딩 모델 테스트 실패: {e}")
        results['embedding_model'] = False

    # 2. Vector Store 테스트
    try:
        results['vector_store'] = test_vector_store_multi_vector()
    except Exception as e:
        logger.error(f"Vector Store 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results['vector_store'] = False

    # 3. Hybrid Search Engine 테스트
    try:
        results['hybrid_search'] = test_hybrid_search_engine()
    except Exception as e:
        logger.error(f"Hybrid Search 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results['hybrid_search'] = False

    # 4. 기존 데이터 테스트
    try:
        results['existing_data'] = test_with_existing_data()
    except Exception as e:
        logger.error(f"기존 데이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results['existing_data'] = False

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("테스트 결과 요약")
    logger.info("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\n전체 결과: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


if __name__ == "__main__":
    main()
