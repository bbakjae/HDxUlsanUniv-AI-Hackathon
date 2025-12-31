"""
빠른 테스트 스크립트
각 컴포넌트가 정상 작동하는지 확인
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """모듈 import 테스트"""
    print("\n=== Testing Imports ===")
    try:
        from src.parsers.document_parser import DocumentParser
        print("✅ DocumentParser imported")

        from src.embeddings.embedding_model import BGEM3Embedder
        print("✅ BGEM3Embedder imported")

        from src.search.vector_store import QdrantVectorStore
        print("✅ QdrantVectorStore imported")

        from src.search.bm25_search import BM25SearchEngine
        print("✅ BM25SearchEngine imported")

        from src.search.hybrid_search import HybridSearchEngine
        print("✅ HybridSearchEngine imported")

        from src.llm.qwen_model import QwenSummarizer
        print("✅ QwenSummarizer imported")

        from src.recommend.recommender import FileRecommender
        print("✅ FileRecommender imported")

        print("\n✅ All imports successful!")
        return True

    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_data_generation():
    """데이터 생성 테스트"""
    print("\n=== Testing Data Generation ===")
    try:
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        import tempfile
        import shutil

        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")

        generator = SyntheticDataGenerator(temp_dir)

        # 소량 생성 (각 타입당 2개)
        files = generator.generate_dataset(num_files_per_type=2)

        print(f"✅ Generated {len(files)} test files")

        # 정리
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")

        return True

    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False


def test_parser():
    """파서 테스트"""
    print("\n=== Testing Parser ===")
    try:
        from src.parsers.document_parser import DocumentParser
        from scripts.generate_synthetic_data import SyntheticDataGenerator
        import tempfile
        import shutil

        # 테스트 파일 생성
        temp_dir = tempfile.mkdtemp()
        generator = SyntheticDataGenerator(temp_dir)

        # PDF 하나만 생성
        test_file = generator.generate_pdf("test_document.pdf")
        print(f"Created test file: {test_file}")

        # 파싱
        parser = DocumentParser()
        result = parser.parse_file(str(test_file))

        if result['success']:
            print(f"✅ Parsed successfully")
            print(f"   Text length: {len(result['text'])} characters")
            print(f"   Metadata: {result['metadata']}")
        else:
            print(f"❌ Parsing failed: {result.get('error')}")

        # 정리
        shutil.rmtree(temp_dir)

        return result['success']

    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False


def test_embedder():
    """임베딩 모델 테스트"""
    print("\n=== Testing Embedder ===")
    try:
        import torch
        from src.embeddings.embedding_model import BGEM3Embedder

        print("Loading BGE-M3 model (this may take a while)...")

        embedder = BGEM3Embedder(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_fp16=torch.cuda.is_available()
        )

        # 테스트 텍스트
        test_texts = [
            "2024년 매출 보고서입니다.",
            "고객 만족도 조사 결과입니다."
        ]

        print("Generating embeddings...")
        embeddings = embedder.encode_queries(test_texts)

        print(f"✅ Embeddings generated")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embedder.get_embedding_dim()}")

        return True

    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        return False


def test_vector_store():
    """벡터 스토어 테스트"""
    print("\n=== Testing Vector Store ===")
    try:
        import numpy as np
        from src.search.vector_store import QdrantVectorStore
        import tempfile
        import shutil

        # 임시 디렉토리
        temp_dir = tempfile.mkdtemp()

        store = QdrantVectorStore(
            storage_path=temp_dir,
            collection_name="test_collection",
            recreate_collection=True
        )

        # 테스트 데이터
        test_vectors = np.random.rand(3, 1024).astype(np.float32)
        test_ids = ["doc1", "doc2", "doc3"]
        test_payloads = [
            {"text": "문서 1", "type": "pdf"},
            {"text": "문서 2", "type": "docx"},
            {"text": "문서 3", "type": "pptx"}
        ]

        # 추가
        success = store.add_documents(test_ids, test_vectors, test_payloads)

        if success:
            print(f"✅ Documents added")
            print(f"   Total documents: {store.count_documents()}")

            # 검색 테스트
            query_vector = np.random.rand(1024).astype(np.float32)
            results = store.search(query_vector, top_k=2)

            print(f"✅ Search successful")
            print(f"   Found {len(results)} results")

        # 정리
        shutil.rmtree(temp_dir)

        return success

    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_bm25():
    """BM25 검색 테스트"""
    print("\n=== Testing BM25 Search ===")
    try:
        from src.search.bm25_search import BM25SearchEngine

        engine = BM25SearchEngine(use_korean_tokenizer=True)

        # 테스트 문서
        documents = [
            "2024년 매출 보고서입니다.",
            "고객 만족도 향상 계획서입니다.",
            "AI 개발 프로젝트 제안서입니다."
        ]
        doc_ids = ["doc1", "doc2", "doc3"]

        # 인덱싱
        engine.index_documents(documents, doc_ids)

        # 검색
        results = engine.search("매출 보고서", top_k=2)

        print(f"✅ BM25 search successful")
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"   {i+1}. Score: {result['score']:.2f} - {result['text'][:30]}...")

        return True

    except Exception as e:
        print(f"❌ BM25 test failed: {e}")
        return False


def main():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("🧪 Quick Component Test")
    print("="*60)

    results = {
        "Imports": test_imports(),
        "Data Generation": test_data_generation(),
        "Parser": test_parser(),
        "Embedder": test_embedder(),
        "Vector Store": test_vector_store(),
        "BM25 Search": test_bm25()
    }

    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    passed = sum(results.values())
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
