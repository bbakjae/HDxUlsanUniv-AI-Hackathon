from sentence_transformers import SentenceTransformer
from index.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)