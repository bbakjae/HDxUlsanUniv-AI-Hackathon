from pymongo import MongoClient
from src.index.config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

#파일 테이블
files_col = db["files"]

#파일 내 청크 테이블
chunks_col = db["chunks"]
