import numpy as np
from index.embedding.embedder import model
from index.vector.faiss_index import load_index
from index.db.mongo import chunks_col, files_col
from index.vector.faiss_index import search_vectors
# FAISS index 로드 (1회)
index = load_index()

# 임베딩 모델 로드 (인덱싱과 동일)
model = model

def search(query: str, top_k: int = 5, oversample: int = 3):

    # 질의 임베딩
    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    # FAISS에서 넉넉히 검색 (한 파일에 여러 청크가 있기 때문)
    scores, ids = search_vectors(q_emb, top_k * oversample)

    best_per_file = {}

    # 각 벡터 청크에서 file_id 기준으로 최고 점수 chunk만 유지
    for vid, score in zip(ids[0], scores[0]):
        if vid == -1:
            continue

        chunk = chunks_col.find_one({"vector_id": int(vid)})
        if chunk is None:
            continue

        file_id = chunk["file_id"]

        #같은 파일인데 청크 중 가장 높은 유사도를 하나로 지정.
        prev = best_per_file.get(file_id)
        if prev is None or score > prev["score"]:
            best_per_file[file_id] = {
                "score": float(score),
                "vector_id": int(vid),
                "chunk": chunk,
            }

    # score 기준 정렬 후 top_k
    ranked = sorted(
        best_per_file.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    # 최종 결과 구성
    results = []
    for item in ranked:
        chunk = item["chunk"]
        file = files_col.find_one({"file_id": chunk["file_id"]})

        results.append({
            "score": item["score"],
            "file_path": file["path"] if file else None,
            "text": chunk["text"],
            "vector_id": item["vector_id"],
        })

    return results