from indexing.scanner import scan_files
from indexing.metadata import save_file_metadata
from indexing.extractor import extract_text
from indexing.chunker import chunk_text
from indexing.indexer import index_chunks
from db.mongo import files_col, chunks_col
from index.vector.faiss_index import (
    reset_index,
    delete_index_file,
    save_index,
)
from vector.faiss_index import reset_index

def full_indexing(root_path: str):
    # 풀 인덱싱 다시 할 때 DB 전체 삭제
    files_col.delete_many({})
    chunks_col.delete_many({})
    delete_index_file()
    reset_index()

    for path in scan_files(root_path):
        print(f"Indexing: {path}")
        #file DB에 저장
        file_doc = save_file_metadata(path)

        #file_type: 파일확장자
        text = extract_text(path, file_doc["file_type"])
        if not text.strip():
            continue

        #파일 청크단위로 분리 (리스트)
        chunks = chunk_text(text)
        index_chunks(file_doc, chunks)
    save_index()

if __name__ == "__main__":
    #파일 스캔 후 임베딩,벡터DB 저장 몽고DB 인덱싱 저장
    full_indexing("C:/aaa")
