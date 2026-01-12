import faiss
import numpy as np
from index.config import EMBED_DIM
from pathlib import Path

#faiss 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss"
FAISS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = FAISS_DIR / "faiss.index"

index = faiss.IndexFlatIP(EMBED_DIM)

def reset_index():
    index.reset()

def add_vectors(vectors: np.ndarray):
    index.add(vectors)

def search_vectors(query_vec: np.ndarray, top_k: int):
    return index.search(query_vec, top_k)

def save_index():
    faiss.write_index(index, str(FAISS_INDEX_PATH))

def load_index():
    global index

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index

def delete_index_file():
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()  # 파일 삭제