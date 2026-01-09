from pymongo import MongoClient
from index.config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

files_col = db["files"]
chunks_col = db["chunks"]
