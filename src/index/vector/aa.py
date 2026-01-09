import numpy as np
from faiss_index import load_index
from sentence_transformers import SentenceTransformer
from index.db.mongo import chunks_col, files_col
index = load_index()

print("index type:", type(index))
print("dimension:", index.d)
print("total vectors:", index.ntotal)

# FAISS 로드
index = load_index()

# 임베딩 모델 로드 (인덱싱 때 쓴 것과 동일해야 함)
model = SentenceTransformer("nlpai-lab/KURE-v1")

# 예시 질의
query = "선박의 연료 소비 최소화 관련 내용"

# 질의 임베딩
q_emb = model.encode([query])
q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

# 검색
scores, ids = index.search(q_emb, 5)

print("IDS:", ids)
print("SCORES:", scores)


for vid, score in zip(ids[0], scores[0]):
    chunk = chunks_col.find_one({"vector_id": int(vid)})
    file = files_col.find_one({"file_id": chunk["file_id"]})

    print("=" * 80)
    print(f"Score: {score:.4f}")
    print(f"File: {file['path']}")
    print("Text snippet:")
    print(chunk["text"][:300])