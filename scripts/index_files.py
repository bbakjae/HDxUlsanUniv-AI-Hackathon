"""
파일 인덱싱 스크립트
네트워크 드라이브의 모든 파일을 파싱하고 벡터 DB에 인덱싱
"""
import hashlib
import os
import sys
import uuid
from pathlib import Path
import yaml
from pymongo import MongoClient, UpdateOne
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
        # 1. MongoDB 초기화
        self.mongo_client = MongoClient(self.config['mongodb']['uri'])
        self.db = self.mongo_client[self.config['mongodb']['db_name']]
        self.files_col = self.db['files']
        self.chunks_col = self.db['chunks']

        # 인덱스 생성 (조회 속도 최적화)
        self.files_col.create_index("path", unique=True)
        self.chunks_col.create_index("file_id")

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
            # word파일의 ~$가 포함된 임시 파일 필터링
            filtered_files = [
                f for f in found_files
                if not f.name.startswith("~$")
            ]
            files.extend(filtered_files)

        logger.info(f"Found {len(files)} files in {directory}")
        return [str(f) for f in files]


    def delete_file_index(self, file_id: str):
        """기존 파일의 모든 인덱스 데이터 삭제"""
        if not file_id:
            return

        try:
            # 1. Vector DB에서 삭제
            from qdrant_client.http import models as rest

            self.vector_store.client.delete(
                collection_name=self.vector_store.collection_name,
                points_selector=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_id",
                            match=rest.MatchValue(value=file_id),
                        ),
                    ]
                ),
            )
            logger.info(f"Deleted old vectors for file_id: {file_id}")
        except Exception as e:
            logger.warning(f"Could not delete vectors for {file_id}: {e}")

        # 2. MongoDB chunks 삭제
        self.chunks_col.delete_many({"file_id": file_id})

    def index_files(
            self,
            file_paths: list,
            batch_size: int = 10
    ) -> dict:
        """
        파일들을 인덱싱 (1차 메타데이터 -> 2차 텍스트 내용 해시 검증 적용)
        """
        stats = {
            'total_files': len(file_paths),
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'total_chunks': 0
        }

        all_documents = []
        all_doc_ids = []
        all_metadata = []

        logger.info(f"Starting incremental indexing of {len(file_paths)} files...")

        for i in tqdm(range(0, len(file_paths), batch_size), desc="Indexing files"):
            full_batch_paths = file_paths[i:i + batch_size]

            valid_batch_paths = []
            batch_new_hashes = {}
            # 해시 체크를 위해 미리 파싱된 결과를 보관 (중복 파싱 방지)
            pre_parsed_results = {}

            for path in full_batch_paths:
                path_obj = Path(path)
                stat = path_obj.stat()
                current_mtime = datetime.fromtimestamp(stat.st_mtime)
                current_size = stat.st_size

                existing_doc = self.files_col.find_one({"path": path})

                needs_idx = True
                new_hash = None

                # --- [1차 검증: 메타데이터] ---
                if existing_doc:
                    # DB에서 가져온 시간과 현재 파일 시간을 모두 초 단위 정수로 변환. 파일 시간과 mongoDB 파일 시간 단위가 다름
                    db_mtime_int = int(existing_doc.get('mtime').timestamp())
                    current_mtime_int = int(current_mtime.timestamp())
                    if (existing_doc.get('size') == current_size and
                            db_mtime_int == current_mtime_int):
                        # 메타데이터가 완벽히 일치하면 패스
                        needs_idx = False
                    else:
                        # --- [2차 검증: 텍스트 내용 해시] ---
                        parse_results = self.parser.batch_parse([path])
                        if parse_results and parse_results[0].get('success'):
                            result = parse_results[0]  # 리스트의 첫 번째 결과 추출
                            full_text = result.get('full_text', '').strip()
                            new_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()

                            if existing_doc.get('content_hash') == new_hash:
                                logger.info("해당 파일 내용 안 바뀜")
                                self.files_col.update_one(
                                    {"path": path},
                                    {"$set": {"mtime": current_mtime, "size": current_size}}
                                )
                                needs_idx = False
                            else:
                                logger.info("해당 파일 내용 바뀜")
                                # 진짜 내용이 바뀐 경우: 파싱 결과를 보관하여 아래에서 재사용
                                pre_parsed_results[path] = result

                if needs_idx:
                    if existing_doc:
                        self.delete_file_index(existing_doc.get('file_id'))

                    valid_batch_paths.append(path)
                    # 해시가 아직 생성되지 않았다면 파싱 후 생성
                    if new_hash is None:
                        if path not in pre_parsed_results:

                            res = self.parser.batch_parse([path])
                            if res and res[0].get('success'):
                                pre_parsed_results[path] = res[0]

                        res = pre_parsed_results.get(path)
                        txt = res.get('full_text', '').strip() if res else ""
                        new_hash = hashlib.sha256(txt.encode('utf-8')).hexdigest()

                    batch_new_hashes[path] = new_hash
                else:
                    stats['skipped'] += 1

            if not valid_batch_paths:
                continue

            # 1. 파일 파싱 결과 정리 (이미 위에서 파싱했다면 보관된 것을 사용)
            parsed_results = []
            remaining_paths = [p for p in valid_batch_paths if p not in pre_parsed_results]

            # 아직 파싱되지 않은 나머지가 있다면 배치 파싱 수행
            if remaining_paths:
                parsed_results.extend(self.parser.batch_parse(remaining_paths))

            # 미리 파싱해둔 결과 추가
            for p in valid_batch_paths:
                if p in pre_parsed_results:
                    parsed_results.append(pre_parsed_results[p])

            batch_texts = []
            batch_ids = []
            batch_metadata = []
            batch_db_updates = []
            batch_chunk_docs = []

            for result in parsed_results:
                if not result or not result['success']:
                    stats['failed'] += 1
                    continue

                full_text = result['full_text']
                file_path = result['file_path']
                if not full_text.strip():
                    stats['failed'] += 1
                    continue

                current_file_id = str(uuid.uuid4())
                now = datetime.now()

                # created_at 유지를 위해 다시 확인
                existing_doc = self.files_col.find_one({"path": file_path})
                created_at = existing_doc.get('created_at', now) if existing_doc else now

                max_chunk_length = self.config['embedding'].get('max_length', 8192)
                chunks = result.get('chunks', [])

                if not chunks or len(full_text) <= max_chunk_length:
                    chunks = [{'text': full_text, 'chunk_id': 0}]

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else chunk
                    if not chunk_text.strip():
                        continue

                    v_id_str = f"{current_file_id}_{chunk_idx}"
                    chunk_uuid = str(uuid.uuid4())

                    file_info = {
                        'file_id': current_file_id,
                        'chunk_id': chunk_uuid,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'file_name': result['file_name'],
                        'file_path': file_path,
                        'file_type': Path(file_path).suffix.replace('.', ''),
                        'modified_time': datetime.fromtimestamp(Path(file_path).stat().st_mtime).isoformat(),
                        'text': chunk_text,
                        'vector_id': v_id_str,
                        **result.get('metadata', {})
                    }

                    batch_texts.append(chunk_text)
                    batch_ids.append(v_id_str)
                    batch_metadata.append(file_info)

                    batch_chunk_docs.append({
                        "chunk_id": chunk_uuid,
                        "file_id": current_file_id,
                        "chunk_index": chunk_idx,
                        "text": chunk_text,
                        "vector_id": v_id_str
                    })
                    stats['total_chunks'] += 1

                stat = Path(file_path).stat()
                batch_db_updates.append(UpdateOne(
                    {"path": file_path},
                    {"$set": {
                        "file_id": current_file_id,
                        "path": file_path,
                        "size": stat.st_size,
                        "mtime": datetime.fromtimestamp(stat.st_mtime),
                        "file_type": Path(file_path).suffix.replace('.', ''),
                        "content_hash": batch_new_hashes.get(file_path),
                        "created_at": created_at,
                        "last_indexed_at": now
                    }},
                    upsert=True
                ))
                stats['successful'] += 1

            if not batch_texts:
                continue

            # 2. 임베딩 및 Vector DB 저장
            try:
                if self.use_sparse:
                    embeddings_result = self.embedder.encode_for_indexing(batch_texts)
                    batch_embeddings = embeddings_result['dense_vecs']
                    batch_sparse = embeddings_result['sparse_vecs']
                else:
                    embeddings_result = self.embedder.encode_documents(batch_texts, include_sparse=False)
                    batch_embeddings = embeddings_result['dense_vecs']
                    batch_sparse = None

                payloads = [
                    {
                        'text': meta['text'],
                        'file_id': meta['file_id'],
                        'file_name': meta['file_name'],
                        'file_path': meta['file_path'],
                        'file_type': meta['file_type'],
                        'modified_time': meta['modified_time'],
                        'chunk_index': meta['chunk_index'],
                        'total_chunks': meta['total_chunks']
                    }
                    for meta in batch_metadata
                ]

                self.vector_store.add_documents(
                    ids=batch_ids,
                    vectors=batch_embeddings,
                    payloads=payloads,
                    sparse_vectors=batch_sparse
                )

                if batch_db_updates:
                    self.files_col.bulk_write(batch_db_updates)
                if batch_chunk_docs:
                    self.chunks_col.insert_many(batch_chunk_docs)

                all_documents.extend(batch_texts)
                all_doc_ids.extend(batch_ids)
                all_metadata.extend(batch_metadata)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                stats['failed'] += len(valid_batch_paths)

        # 5. BM25 인덱싱
        if all_documents:
            logger.info("Updating BM25 index...")
            self.bm25_engine.index_documents(
                documents=all_documents,
                document_ids=all_doc_ids,
                metadata=all_metadata
            )
            bm25_index_path = Path(self.config['data']['cache_dir']) / 'bm25_index.pkl'
            self.bm25_engine.save_index(str(bm25_index_path))

        logger.info(f"\nIndexing complete!")
        logger.info(
            f"  Total: {stats['total_files']}, Indexed: {stats['successful']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")



        return stats

    def cleanup_deleted_files(self, scanned_files: list):
        """
        DB에는 있지만 실제 디렉토리에는 없는 파일을 찾아 인덱스에서 삭제

        Args:
            scanned_files: 이번 스캔에서 발견된 전체 파일 경로 리스트
        """
        logger.info("Checking for deleted files in the index...")

        # DB에 저장된 모든 파일 정보 가져오기
        db_files = list(self.files_col.find({}, {"file_id": 1, "path": 1}))

        scanned_files_set = set(scanned_files)
        deleted_count = 0

        for doc in db_files:
            db_path = doc.get('path')
            file_id = doc.get('file_id')

            # 이번 스캔 목록에 없고, 실제 경로에도 파일이 없다면 삭제 대상으로 판단
            if db_path not in scanned_files_set and not Path(db_path).exists():
                logger.info(f"Detected deleted file: {db_path}. Removing from index...")

                # 벡터 DB 및 MongoDB chunks 삭제
                self.delete_file_index(file_id)

                # MongoDB files 정보 삭제
                self.files_col.delete_one({"file_id": file_id})
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleanup complete. Removed {deleted_count} deleted files from index.")
        else:
            logger.info("No deleted files found. Index is up to date.")

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

    # 인덱싱 후 삭제된 파일 정리
    indexer.cleanup_deleted_files(file_paths)

    logger.info("\n" + "="*50)
    logger.info("Indexing Summary")
    logger.info("="*50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
