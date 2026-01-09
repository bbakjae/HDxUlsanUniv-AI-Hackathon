from pathlib import Path
from index.config import SUPPORTED_EXTENSIONS

def scan_files(root_path: str):
    for path in Path(root_path).rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path