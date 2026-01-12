from pathlib import Path
from index.config import SUPPORTED_EXTENSIONS

#해당 폴더내 모든 하위폴더, 파일 찾는 기능
def scan_files(root_path: str):
    for path in Path(root_path).rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path