import numpy as np
from index.embedding.embedder import model
from index.vector.faiss_index import search_vectors
from index.db.mongo import chunks_col, files_col

def search(query: str, top_k=5):
    q_emb = model.encode([query], prompt_name="query")
    q_emb = q_emb / np.linalg.norm(q_emb)

    scores, ids = search_vectors(q_emb, top_k)

    results = []
    for vid, score in zip(ids[0], scores[0]):
        chunk = chunks_col.find_one({"vector_id": int(vid)})
        file = files_col.find_one({"file_id": chunk["file_id"]})

        results.append({
            "score": float(score),
            "file_path": file["path"],
            "text": chunk["text"]
        })

    return results