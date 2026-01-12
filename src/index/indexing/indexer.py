import uuid
import numpy as np
from index.db.mongo import chunks_col
from index.embedding.embedder import model
from index.vector.faiss_index import add_vectors, index

#벡터 DB에 저장 및 mongoDB에 chunk 컬렉션(테이블) 저장
def index_chunks(file_doc, chunks):
    # 비어있는 청크 제외
    texts = [c for c in chunks if c.strip()]
    if not texts:
        return

    #텍스트를 벡터 리스트로 변경
    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    #검색 엔진이 찾았을 때 DB내 위치를 알기 위해 인덱싱
    start_id = index.ntotal
    # faiss에 백터 저장
    add_vectors(embeddings)

    #DB에  chunk테이블 내용 저장
    for i, text in enumerate(texts):
        chunk_doc = {
            "chunk_id": str(uuid.uuid4()),
            "file_id": file_doc["file_id"],
            "chunk_index": i,
            "text": text,
            "vector_id": start_id + i
        }
        chunks_col.insert_one(chunk_doc)