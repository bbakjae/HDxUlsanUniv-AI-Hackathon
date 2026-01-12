import pdfplumber
from docx import Document
from pathlib import Path

#해당 경로의 파일 타입별 텍스트 추출
def extract_text(path: Path, file_type: str) -> str:
    if file_type == "pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    if file_type == "docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    if file_type == "txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    #해당 확장자 없을시 빈값
    return ""
