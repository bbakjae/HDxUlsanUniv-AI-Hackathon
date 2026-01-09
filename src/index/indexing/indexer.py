import uuid
import numpy as np
from index.db.mongo import chunks_col
from index.embedding.embedder import model
from index.vector.faiss_index import add_vectors, index

def index_chunks(file_doc, chunks):
    texts = [c for c in chunks if c.strip()]
    if not texts:
        return

    embeddings = model.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    start_id = index.ntotal
    add_vectors(embeddings)

    for i, text in enumerate(texts):
        chunk_doc = {
            "chunk_id": str(uuid.uuid4()),
            "file_id": file_doc["file_id"],
            "chunk_index": i,
            "text": text,
            "vector_id": start_id + i
        }
        chunks_col.insert_one(chunk_doc)