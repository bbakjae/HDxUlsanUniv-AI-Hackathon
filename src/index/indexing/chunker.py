from index.config import CHUNK_SIZE, CHUNK_OVERLAP

# 텍스트를 청크사이즈 만큼 짤라서 리스트로 저장
def chunk_text(text: str):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
