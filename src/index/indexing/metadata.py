import uuid
from datetime import datetime
from pathlib import Path
from index.db.mongo import files_col

def save_file_metadata(path: Path):
    doc = {
        "file_id": str(uuid.uuid4()),
        "path": str(path),
        "size": path.stat().st_size,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime),
        "file_type": path.suffix.replace(".", ""),
        "content_hash": None,
        "created_at": datetime.utcnow(),
        "last_indexed_at": datetime.utcnow()
    }
    files_col.insert_one(doc)
    return doc