from sentence_transformers import SentenceTransformer
from index.config import EMBEDDING_MODEL

#임베딩 모델 불러오기
model = SentenceTransformer(EMBEDDING_MODEL)