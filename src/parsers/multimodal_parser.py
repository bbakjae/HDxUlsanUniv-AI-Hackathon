"""
멀티모달 파서 - 문서 청킹 및 전처리 포함
"""

from typing import List, Dict, Optional
from pathlib import Path
import hashlib
import logging

from .document_parser import DocumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """텍스트 청킹 유틸리티"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 오버랩 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 입력 텍스트

        Returns:
            청크 리스트
        """
        if not text or len(text) < self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 청크 경계에서 단어 분리 방지
            if end < len(text):
                # 다음 공백이나 줄바꿈을 찾음
                while end < len(text) and text[end] not in [' ', '\n', '\t']:
                    end += 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 다음 시작 위치 (오버랩 고려)
            start = end - self.chunk_overlap

        return chunks

    def semantic_chunk(self, text: str) -> List[str]:
        """
        의미 단위로 텍스트 분할 (단락 기준)

        Args:
            text: 입력 텍스트

        Returns:
            청크 리스트
        """
        # 단락 분리
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 현재 청크 + 새 단락이 크기를 초과하면 청크 저장
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class MultimodalParser:
    """
    멀티모달 파일 파서
    파일 파싱 + 청킹 + 메타데이터 생성 통합
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_semantic_chunking: bool = True,
        ocr_config: Optional[Dict] = None
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            use_semantic_chunking: 의미 기반 청킹 사용 여부
            ocr_config: OCR 설정
        """
        self.document_parser = DocumentParser(ocr_config)
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
        self.use_semantic_chunking = use_semantic_chunking

    def parse_and_chunk(self, file_path: str) -> Dict:
        """
        파일을 파싱하고 청크로 분할

        Args:
            file_path: 파일 경로

        Returns:
            {
                'file_id': str,
                'file_path': str,
                'file_name': str,
                'file_type': str,
                'full_text': str,
                'chunks': List[Dict],
                'metadata': dict,
                'success': bool
            }
        """
        # 1. 파일 파싱
        parse_result = self.document_parser.parse_file(file_path)

        if not parse_result['success']:
            return {
                'file_id': None,
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_type': Path(file_path).suffix[1:],
                'full_text': '',
                'chunks': [],
                'metadata': {},
                'success': False,
                'error': parse_result.get('error', 'Unknown error')
            }

        # 2. 파일 ID 생성 (경로 해시)
        file_id = self._generate_file_id(file_path)

        # 3. 텍스트 청킹
        full_text = parse_result['text']

        if self.use_semantic_chunking:
            chunk_texts = self.text_chunker.semantic_chunk(full_text)
        else:
            chunk_texts = self.text_chunker.chunk_text(full_text)

        # 4. 청크 메타데이터 생성
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk = {
                'chunk_id': f"{file_id}_chunk_{idx}",
                'chunk_index': idx,
                'text': chunk_text,
                'char_count': len(chunk_text),
                'file_id': file_id
            }
            chunks.append(chunk)

        # 5. 전체 메타데이터
        metadata = parse_result['metadata']
        metadata['chunk_count'] = len(chunks)
        metadata['total_chars'] = len(full_text)

        return {
            'file_id': file_id,
            'file_path': str(Path(file_path).absolute()),
            'file_name': Path(file_path).name,
            'file_type': Path(file_path).suffix[1:],
            'full_text': full_text,
            'chunks': chunks,
            'metadata': metadata,
            'success': True
        }

    def batch_parse(self, file_paths: List[str]) -> List[Dict]:
        """
        여러 파일을 배치로 파싱

        Args:
            file_paths: 파일 경로 리스트

        Returns:
            파싱 결과 리스트
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.parse_and_chunk(file_path)
                results.append(result)

                if result['success']:
                    logger.info(f"✅ Parsed: {result['file_name']} ({len(result['chunks'])} chunks)")
                else:
                    logger.warning(f"❌ Failed: {result['file_name']} - {result.get('error', 'Unknown')}")

            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                results.append({
                    'file_id': None,
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'success': False,
                    'error': str(e)
                })

        return results

    def _generate_file_id(self, file_path: str) -> str:
        """파일 경로 기반 고유 ID 생성"""
        path_str = str(Path(file_path).absolute())
        return hashlib.md5(path_str.encode()).hexdigest()


def test_multimodal_parser():
    """멀티모달 파서 테스트"""
    parser = MultimodalParser(
        chunk_size=512,
        chunk_overlap=50,
        use_semantic_chunking=True
    )

    test_file = "/path/to/test_document.pdf"

    if Path(test_file).exists():
        result = parser.parse_and_chunk(test_file)

        print(f"\n{'='*60}")
        print(f"File: {result['file_name']}")
        print('='*60)
        print(f"Success: {result['success']}")
        print(f"File ID: {result['file_id']}")
        print(f"Text length: {len(result['full_text'])}")
        print(f"Chunk count: {len(result['chunks'])}")
        print(f"\nMetadata: {result['metadata']}")

        print(f"\n첫 번째 청크:")
        if result['chunks']:
            print(result['chunks'][0]['text'][:300])


if __name__ == "__main__":
    test_multimodal_parser()
