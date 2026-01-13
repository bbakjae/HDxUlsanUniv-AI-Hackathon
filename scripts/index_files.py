"""
파일 인덱싱 스크립트
네트워크 드라이브의 모든 파일을 파싱하고 벡터 DB에 인덱싱
"""

import os
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
import argparse
from datetime import datetime

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.multimodal_parser import MultimodalParser
from src.embeddings.embedding_model import BGEM3Embedder
from src.search.vector_store import QdrantVectorStore
from src.search.bm25_search import BM25SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileIndexer:
    """파일 인덱서 (Multi-Vector 지원)"""

    def __init__(self, config_path: str, use_sparse: bool = True):
        """
        Args:
            config_path: 설정 파일 경로
            use_sparse: Sparse 벡터 사용 여부 (Multi-Vector)
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.use_sparse = use_sparse

        # 컴포넌트 초기화
        logger.info("Initializing components...")
        logger.info(f"Multi-Vector mode: {use_sparse}")

        self.parser = MultimodalParser(
            chunk_size=self.config['parsing']['chunk_size'],
            chunk_overlap=self.config['parsing']['chunk_overlap'],
            use_semantic_chunking=True
        )

        self.embedder = BGEM3Embedder(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device'],
            use_fp16=self.config['embedding']['use_fp16'],
            max_length=self.config['embedding']['max_length']
        )

        self.vector_store = QdrantVectorStore(
            storage_path=self.config['data']['qdrant_storage'],
            collection_name=self.config['qdrant']['collection_name'],
            vector_size=self.config['qdrant']['vector_size'],
            distance=self.config['qdrant']['distance'],
            use_sparse=use_sparse  # Multi-Vector 설정
        )

        self.bm25_engine = BM25SearchEngine(use_korean_tokenizer=True)

        logger.info("Components initialized successfully")

    def scan_directory(self, directory: str) -> list:
        """
        디렉토리에서 지원하는 파일 스캔

        Args:
            directory: 스캔할 디렉토리

        Returns:
            파일 경로 리스트
        """
        directory = Path(directory)
        supported_formats = self.config['parsing']['supported_formats']

        files = []
        for ext in supported_formats:
            pattern = f"**/*.{ext}"
            found_files = list(directory.glob(pattern))
            files.extend(found_files)

        logger.info(f"Found {len(files)} files in {directory}")
        return [str(f) for f in files]

    def index_files(
        self,
        file_paths: list,
        batch_size: int = 10
    ) -> dict:
        """
        파일들을 인덱싱

        Args:
            file_paths: 파일 경로 리스트
            batch_size: 배치 크기

        Returns:
            인덱싱 통계
        """
        stats = {
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0
        }

        all_documents = []
        all_doc_ids = []
        all_metadata = []
        all_embeddings = []

        logger.info(f"Starting indexing of {len(file_paths)} files...")

        for i in tqdm(range(0, len(file_paths), batch_size), desc="Indexing files"):
            batch_paths = file_paths[i:i+batch_size]

            # 1. 파일 파싱
            parsed_results = self.parser.batch_parse(batch_paths)

            batch_texts = []
            batch_ids = []
            batch_metadata = []

            for result in parsed_results:
                if not result['success']:
                    stats['failed'] += 1
                    continue

                full_text = result['full_text']
                if not full_text.strip():
                    stats['failed'] += 1
                    continue

                # 청킹 사용 여부 결정 (긴 문서는 청킹)
                max_chunk_length = self.config['embedding'].get('max_length', 8192)
                chunks = result.get('chunks', [])

                # 청킹이 없거나 짧은 문서는 전체 텍스트 사용
                if not chunks or len(full_text) <= max_chunk_length:
                    chunks = [{'text': full_text, 'chunk_id': 0}]

                # 각 청크를 개별적으로 인덱싱 (긴 문서 검색 품질 향상)
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else chunk
                    if not chunk_text.strip():
                        continue

                    # 청크 ID 생성 (파일 ID + 청크 인덱스)
                    chunk_id = f"{result['file_id']}_{chunk_idx}"

                    file_info = {
                        'file_id': result['file_id'],
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'file_name': result['file_name'],
                        'file_path': result['file_path'],
                        'file_type': result['file_type'],
                        'modified_time': datetime.fromtimestamp(
                            Path(result['file_path']).stat().st_mtime
                        ).isoformat(),
                        'text': chunk_text,
                        **result['metadata']
                    }

                    batch_texts.append(chunk_text)
                    batch_ids.append(chunk_id)
                    batch_metadata.append(file_info)

                    stats['total_chunks'] += 1

                stats['successful'] += 1

            if not batch_texts:
                continue

            # 2. 임베딩 생성 (Dense + Sparse for Multi-Vector)
            try:
                if self.use_sparse:
                    # Multi-Vector: Dense + Sparse 동시 생성
                    embeddings_result = self.embedder.encode_for_indexing(
                        batch_texts,
                        batch_size=self.config['embedding']['batch_size']
                    )
                    batch_embeddings = embeddings_result['dense_vecs']
                    batch_sparse = embeddings_result['sparse_vecs']
                else:
                    # 기존 방식: Dense만
                    embeddings_result = self.embedder.encode_documents(
                        batch_texts,
                        batch_size=self.config['embedding']['batch_size'],
                        include_sparse=False
                    )
                    batch_embeddings = embeddings_result['dense_vecs']
                    batch_sparse = None

                # 3. 벡터 DB에 저장 (청크 정보 포함)
                payloads = [
                    {
                        'text': meta['text'],
                        'file_id': meta.get('file_id', ''),
                        'file_name': meta['file_name'],
                        'file_path': meta['file_path'],
                        'file_type': meta['file_type'],
                        'modified_time': meta['modified_time'],
                        'chunk_index': meta.get('chunk_index', 0),
                        'total_chunks': meta.get('total_chunks', 1)
                    }
                    for meta in batch_metadata
                ]

                self.vector_store.add_documents(
                    ids=batch_ids,
                    vectors=batch_embeddings,
                    payloads=payloads,
                    sparse_vectors=batch_sparse  # Multi-Vector: Sparse 벡터 추가
                )

                # BM25용 데이터 수집
                all_documents.extend(batch_texts)
                all_doc_ids.extend(batch_ids)
                all_metadata.extend(batch_metadata)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                stats['failed'] += len(batch_texts)

        # 4. BM25 인덱싱
        if all_documents:
            logger.info("Creating BM25 index...")
            self.bm25_engine.index_documents(
                documents=all_documents,
                document_ids=all_doc_ids,
                metadata=all_metadata
            )

            # BM25 인덱스 저장
            bm25_index_path = Path(self.config['data']['cache_dir']) / 'bm25_index.pkl'
            self.bm25_engine.save_index(str(bm25_index_path))

        logger.info(f"\nIndexing complete!")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"  Vector DB count: {self.vector_store.count_documents()}")

        return stats


def main():
    parser = argparse.ArgumentParser(description='Index files for AI Agent')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory to index (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate vector database'
    )
    parser.add_argument(
        '--no-sparse',
        action='store_true',
        help='Disable Sparse vectors (use Dense only)'
    )

    args = parser.parse_args()

    # 설정 파일 경로
    config_path = Path(project_root) / args.config

    # 인덱서 초기화 (Multi-Vector 기본 활성화)
    use_sparse = not args.no_sparse if hasattr(args, 'no_sparse') else True
    indexer = FileIndexer(str(config_path), use_sparse=use_sparse)

    # 컬렉션 재생성
    if args.recreate:
        logger.info("Recreating vector database...")
        indexer.vector_store.clear_collection()

    # 디렉토리 결정
    if args.directory:
        directory = args.directory
    else:
        directory = indexer.config['data']['network_drive']

    # 파일 스캔
    file_paths = indexer.scan_directory(directory)

    if not file_paths:
        logger.warning("No files found to index!")
        return

    # 인덱싱 실행
    stats = indexer.index_files(file_paths, batch_size=args.batch_size)

    logger.info("\n" + "="*50)
    logger.info("Indexing Summary")
    logger.info("="*50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
