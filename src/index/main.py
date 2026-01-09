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
    files_col.delete_many({})
    chunks_col.delete_many({})
    delete_index_file()
    reset_index()

    for path in scan_files(root_path):
        print(f"Indexing: {path}")
        file_doc = save_file_metadata(path)

        text = extract_text(path, file_doc["file_type"])
        if not text.strip():
            continue

        chunks = chunk_text(text)
        index_chunks(file_doc, chunks)
    save_index()

if __name__ == "__main__":
    full_indexing("C:/aaa")
