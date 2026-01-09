MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_db"

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
EMBED_DIM = 1024

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
FAISS_INDEX_PATH = "faiss/faiss.index"