"""
VectorDB 스키마 비교 실험
세 가지 시나리오 비교:
1. Baseline: 현재 스키마 (Dense 벡터 + 기본 메타데이터)
2. Enhanced Metadata: 확장 메타데이터 + Payload Index
3. Multi-Vector: Dense + Sparse Named Vectors (BGE-M3 활용)
4. Combined: Enhanced Metadata + Multi-Vector
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import re

import numpy as np
import torch
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny,
    SparseVectorParams, SparseVector, NamedVector,
    PayloadSchemaType, TextIndexParams, TokenizerType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SchemaEvalResult:
    """스키마 평가 결과"""
    schema_name: str
    description: str

    # 인덱싱 성능
    indexing_time_sec: float
    index_size_mb: float

    # 검색 성능
    search_time_avg_ms: float
    search_time_p95_ms: float

    # 검색 품질
    mrr: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float

    # 필터링 성능
    filter_search_time_avg_ms: float
    filter_improvement_pct: float

    # 종합 점수
    weighted_score: float

    total_queries: int
    total_documents: int
    eval_timestamp: str

    # 추가 메트릭
    details: Dict = field(default_factory=dict)


class KeywordExtractor:
    """LLM 기반 키워드/메타데이터 추출기"""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None

        if use_llm:
            self._load_llm()

    def _load_llm(self):
        """LLM 모델 로드"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-7B-Instruct"
        logger.info(f"Loading LLM for keyword extraction: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def extract_keywords_rule_based(self, text: str, file_path: str) -> Dict:
        """규칙 기반 키워드/메타데이터 추출"""
        metadata = {
            'keywords': [],
            'document_type': 'unknown',
            'department': 'unknown',
            'year': None,
            'quarter': None,
            'entities': []
        }

        # 1. 부서 추출 (경로에서)
        path_parts = file_path.lower()
        departments = ['재무', '인사', '마케팅', '개발', '기획', '영업', '총무', '경영']
        for dept in departments:
            if dept in path_parts:
                metadata['department'] = dept
                break

        # 2. 문서 유형 추출
        doc_types = {
            '보고서': ['보고서', 'report', '리포트'],
            '제안서': ['제안서', 'proposal', '제안'],
            '계획서': ['계획서', '계획', 'plan'],
            '회의록': ['회의록', 'meeting', '회의'],
            '분석': ['분석', 'analysis', '분석서'],
            '매뉴얼': ['매뉴얼', 'manual', '가이드'],
        }
        text_lower = text.lower()
        for doc_type, patterns in doc_types.items():
            for pattern in patterns:
                if pattern in text_lower:
                    metadata['document_type'] = doc_type
                    break

        # 3. 연도/분기 추출
        year_match = re.search(r'(20\d{2})년', text)
        if year_match:
            metadata['year'] = int(year_match.group(1))

        quarter_match = re.search(r'([1-4])분기|Q([1-4])|([상하])반기', text)
        if quarter_match:
            if quarter_match.group(1):
                metadata['quarter'] = f"Q{quarter_match.group(1)}"
            elif quarter_match.group(2):
                metadata['quarter'] = f"Q{quarter_match.group(2)}"
            elif quarter_match.group(3):
                metadata['quarter'] = f"{quarter_match.group(3)}반기"

        # 4. 키워드 추출 (간단한 규칙)
        # 중요 명사/키워드 패턴
        keyword_patterns = [
            r'매출|실적|성과|목표|계획|전략|분석|개발|프로젝트',
            r'고객|사용자|시스템|서비스|제품',
            r'AI|ML|데이터|클라우드|보안',
        ]

        keywords = set()
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)

        metadata['keywords'] = list(keywords)[:10]  # 최대 10개

        # 5. 고유명사 추출 (대문자로 시작하는 단어들)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, text)
        metadata['entities'] = list(set(entities))[:5]

        return metadata

    def extract_keywords_llm(self, text: str) -> Dict:
        """LLM 기반 키워드 추출"""
        if not self.model:
            return {}

        prompt = f"""다음 문서에서 핵심 정보를 추출해주세요.

문서:
{text[:1500]}

다음 JSON 형식으로 응답해주세요:
{{
    "keywords": ["키워드1", "키워드2", ...],  // 3-5개의 핵심 키워드
    "document_type": "보고서/제안서/계획서/회의록/분석/기타",
    "summary": "1문장 요약",
    "entities": ["고유명사1", "고유명사2", ...]  // 인물, 회사, 제품명 등
}}

JSON:"""

        messages = [
            {"role": "system", "content": "당신은 문서 분석 전문가입니다. JSON 형식으로만 응답합니다."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # JSON 파싱 시도
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {}


class SchemaExperiment:
    """스키마 비교 실험"""

    def __init__(
        self,
        base_storage_path: str,
        embedding_model_name: str = "dragonkue/BGE-m3-ko",
        test_queries_path: str = None
    ):
        self.base_storage_path = Path(base_storage_path)
        self.embedding_model_name = embedding_model_name
        self.test_queries_path = test_queries_path

        self.embedder = None
        self.keyword_extractor = KeywordExtractor(use_llm=False)  # 규칙 기반 사용

    def _load_embedder(self):
        """임베딩 모델 로드"""
        if self.embedder is None:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedder = BGEM3FlagModel(
                self.embedding_model_name,
                use_fp16=True,
                device="cuda"
            )
        return self.embedder

    def _string_to_int_id(self, string_id: str) -> int:
        """문자열 ID를 정수 ID로 변환"""
        hash_hex = hashlib.md5(string_id.encode()).hexdigest()[:16]
        return int(hash_hex, 16)

    def _load_test_queries(self) -> List[Dict]:
        """테스트 쿼리 로드"""
        if self.test_queries_path and Path(self.test_queries_path).exists():
            with open(self.test_queries_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('queries', [])

        # 기본 테스트 쿼리
        return [
            {"query": "2024년 매출 실적 보고서", "filter": {"year": 2024}},
            {"query": "마케팅 전략 계획", "filter": {"department": "마케팅"}},
            {"query": "AI 개발 프로젝트 제안서", "filter": {"document_type": "제안서"}},
            {"query": "상반기 실적 분석", "filter": {"quarter": "상반기"}},
            {"query": "고객 만족도 조사 결과", "filter": {}},
            {"query": "신규 시스템 도입 계획", "filter": {"document_type": "계획서"}},
            {"query": "분기별 매출 현황", "filter": {}},
            {"query": "인사 평가 기준", "filter": {"department": "인사"}},
            {"query": "보안 정책 가이드", "filter": {}},
            {"query": "프로젝트 진행 현황 보고", "filter": {"document_type": "보고서"}},
        ]

    def _load_sample_documents(self, max_docs: int = 500) -> List[Dict]:
        """기존 인덱싱된 문서에서 샘플 로드"""
        # 기존 Qdrant 저장소에서 문서 로드
        original_storage = Path(self.base_storage_path) / "qdrant_storage_gdrive"

        if not original_storage.exists():
            logger.warning(f"Original storage not found: {original_storage}")
            return self._generate_sample_documents()

        try:
            client = QdrantClient(path=str(original_storage))

            # 컬렉션 확인
            collections = client.get_collections().collections
            if not any(c.name == "company_files" for c in collections):
                logger.warning("company_files collection not found")
                return self._generate_sample_documents()

            # 문서 로드 (최대 max_docs개)
            result = client.scroll(
                collection_name="company_files",
                limit=max_docs,
                with_payload=True,
                with_vectors=False
            )

            documents = []
            for point in result[0]:
                doc = {
                    'id': point.payload.get('original_id', str(point.id)),
                    'text': point.payload.get('text', ''),
                    'file_path': point.payload.get('file_path', ''),
                    'file_name': point.payload.get('file_name', ''),
                    'file_type': point.payload.get('file_type', ''),
                    'modified_time': point.payload.get('modified_time', ''),
                }
                if doc['text']:
                    documents.append(doc)

            logger.info(f"Loaded {len(documents)} documents from existing storage")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return self._generate_sample_documents()

    def _generate_sample_documents(self) -> List[Dict]:
        """샘플 문서 생성 (실제 문서가 없을 경우)"""
        logger.info("Generating sample documents for testing")

        documents = [
            {"id": "doc_1", "text": "2024년 상반기 매출 실적 보고서입니다. 목표 대비 120% 달성하였습니다.",
             "file_path": "/data/재무팀/2024_상반기_매출보고서.pdf", "file_name": "2024_상반기_매출보고서.pdf", "file_type": "pdf"},
            {"id": "doc_2", "text": "마케팅팀 2024년 전략 계획서입니다. 신규 고객 확보를 위한 디지털 마케팅 강화.",
             "file_path": "/data/마케팅팀/2024_마케팅전략.docx", "file_name": "2024_마케팅전략.docx", "file_type": "docx"},
            {"id": "doc_3", "text": "AI 기반 자동화 시스템 개발 프로젝트 제안서입니다. 업무 효율성 30% 향상 예상.",
             "file_path": "/data/개발팀/AI_자동화_제안서.pptx", "file_name": "AI_자동화_제안서.pptx", "file_type": "pptx"},
            {"id": "doc_4", "text": "2024년 1분기 고객 만족도 조사 결과 보고서. 전체 만족도 4.2점 달성.",
             "file_path": "/data/고객서비스팀/Q1_고객만족도.pdf", "file_name": "Q1_고객만족도.pdf", "file_type": "pdf"},
            {"id": "doc_5", "text": "신입 사원 채용 계획 및 인재 육성 전략입니다. 2024년 50명 채용 목표.",
             "file_path": "/data/인사팀/2024_채용계획.docx", "file_name": "2024_채용계획.docx", "file_type": "docx"},
            {"id": "doc_6", "text": "클라우드 보안 정책 가이드라인입니다. AWS, GCP 환경 보안 설정 포함.",
             "file_path": "/data/IT팀/클라우드_보안_가이드.pdf", "file_name": "클라우드_보안_가이드.pdf", "file_type": "pdf"},
            {"id": "doc_7", "text": "2024년 2분기 프로젝트 진행 현황 보고서입니다. 5개 프로젝트 중 3개 완료.",
             "file_path": "/data/기획팀/Q2_프로젝트현황.pptx", "file_name": "Q2_프로젝트현황.pptx", "file_type": "pptx"},
            {"id": "doc_8", "text": "데이터 분석 플랫폼 도입 계획서입니다. Tableau와 Power BI 비교 분석.",
             "file_path": "/data/개발팀/데이터분석_플랫폼_계획.docx", "file_name": "데이터분석_플랫폼_계획.docx", "file_type": "docx"},
            {"id": "doc_9", "text": "2024년 하반기 영업 목표 및 전략입니다. 신규 시장 진출 계획 포함.",
             "file_path": "/data/영업팀/2024_하반기_영업전략.pdf", "file_name": "2024_하반기_영업전략.pdf", "file_type": "pdf"},
            {"id": "doc_10", "text": "사내 교육 프로그램 안내입니다. AI/ML 기초 과정 신설.",
             "file_path": "/data/인사팀/2024_교육프로그램.pdf", "file_name": "2024_교육프로그램.pdf", "file_type": "pdf"},
        ]

        return documents

    def create_baseline_schema(self, storage_path: str, documents: List[Dict]) -> Tuple[float, float]:
        """
        Baseline 스키마: 현재 방식 (Dense 벡터 + 기본 메타데이터)
        """
        logger.info("Creating Baseline schema...")
        start_time = time.time()

        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(storage))

        # 컬렉션 생성
        if client.collection_exists("baseline"):
            client.delete_collection("baseline")

        client.create_collection(
            collection_name="baseline",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

        # 임베딩 생성 및 인덱싱
        embedder = self._load_embedder()

        texts = [doc['text'] for doc in documents]
        embeddings_result = embedder.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        dense_vecs = embeddings_result['dense_vecs']

        # 포인트 생성
        points = []
        for i, (doc, vec) in enumerate(zip(documents, dense_vecs)):
            point = PointStruct(
                id=self._string_to_int_id(doc['id']),
                vector=vec.tolist(),
                payload={
                    'original_id': doc['id'],
                    'text': doc['text'],
                    'file_name': doc.get('file_name', ''),
                    'file_path': doc.get('file_path', ''),
                    'file_type': doc.get('file_type', ''),
                    'modified_time': doc.get('modified_time', ''),
                }
            )
            points.append(point)

        client.upsert(collection_name="baseline", points=points)

        indexing_time = time.time() - start_time

        # 인덱스 크기 계산 (폴더 크기)
        index_size = sum(f.stat().st_size for f in storage.rglob('*') if f.is_file()) / (1024 * 1024)

        logger.info(f"Baseline indexing complete: {indexing_time:.2f}s, {index_size:.2f}MB")
        return indexing_time, index_size

    def create_enhanced_metadata_schema(self, storage_path: str, documents: List[Dict]) -> Tuple[float, float]:
        """
        Enhanced Metadata 스키마: 확장 메타데이터 + Payload Index
        """
        logger.info("Creating Enhanced Metadata schema...")
        start_time = time.time()

        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(storage))

        # 컬렉션 생성
        if client.collection_exists("enhanced_metadata"):
            client.delete_collection("enhanced_metadata")

        client.create_collection(
            collection_name="enhanced_metadata",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

        # Payload Index 생성
        client.create_payload_index(
            collection_name="enhanced_metadata",
            field_name="keywords",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="enhanced_metadata",
            field_name="department",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="enhanced_metadata",
            field_name="document_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="enhanced_metadata",
            field_name="year",
            field_schema=PayloadSchemaType.INTEGER
        )

        # 임베딩 생성
        embedder = self._load_embedder()
        texts = [doc['text'] for doc in documents]
        embeddings_result = embedder.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        dense_vecs = embeddings_result['dense_vecs']

        # 포인트 생성 (확장 메타데이터 포함)
        points = []
        for i, (doc, vec) in enumerate(zip(documents, dense_vecs)):
            # 키워드/메타데이터 추출
            extracted = self.keyword_extractor.extract_keywords_rule_based(
                doc['text'], doc.get('file_path', '')
            )

            payload = {
                'original_id': doc['id'],
                'text': doc['text'],
                'file_name': doc.get('file_name', ''),
                'file_path': doc.get('file_path', ''),
                'file_type': doc.get('file_type', ''),
                'modified_time': doc.get('modified_time', ''),
                # 확장 메타데이터
                'keywords': extracted.get('keywords', []),
                'department': extracted.get('department', 'unknown'),
                'document_type': extracted.get('document_type', 'unknown'),
                'year': extracted.get('year'),
                'quarter': extracted.get('quarter'),
                'entities': extracted.get('entities', []),
            }

            point = PointStruct(
                id=self._string_to_int_id(doc['id']),
                vector=vec.tolist(),
                payload=payload
            )
            points.append(point)

        client.upsert(collection_name="enhanced_metadata", points=points)

        indexing_time = time.time() - start_time
        index_size = sum(f.stat().st_size for f in storage.rglob('*') if f.is_file()) / (1024 * 1024)

        logger.info(f"Enhanced Metadata indexing complete: {indexing_time:.2f}s, {index_size:.2f}MB")
        return indexing_time, index_size

    def create_multi_vector_schema(self, storage_path: str, documents: List[Dict]) -> Tuple[float, float]:
        """
        Multi-Vector 스키마: Dense + Sparse Named Vectors (BGE-M3 활용)
        """
        logger.info("Creating Multi-Vector schema...")
        start_time = time.time()

        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(storage))

        # 컬렉션 생성 (Named Vectors)
        if client.collection_exists("multi_vector"):
            client.delete_collection("multi_vector")

        client.create_collection(
            collection_name="multi_vector",
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )

        # 임베딩 생성 (Dense + Sparse)
        embedder = self._load_embedder()
        texts = [doc['text'] for doc in documents]
        embeddings_result = embedder.encode(
            texts,
            return_dense=True,
            return_sparse=True,  # Sparse 벡터도 생성
            return_colbert_vecs=False
        )
        dense_vecs = embeddings_result['dense_vecs']
        sparse_vecs = embeddings_result.get('lexical_weights', [])

        # 포인트 생성
        points = []
        for i, doc in enumerate(documents):
            # Sparse 벡터 처리
            sparse_vec = None
            if sparse_vecs and i < len(sparse_vecs):
                sparse_dict = sparse_vecs[i]
                if sparse_dict:
                    indices = list(sparse_dict.keys())
                    values = list(sparse_dict.values())
                    sparse_vec = SparseVector(indices=indices, values=values)

            # Named vectors
            vectors = {
                "dense": dense_vecs[i].tolist()
            }

            point = PointStruct(
                id=self._string_to_int_id(doc['id']),
                vector=vectors,
                payload={
                    'original_id': doc['id'],
                    'text': doc['text'],
                    'file_name': doc.get('file_name', ''),
                    'file_path': doc.get('file_path', ''),
                    'file_type': doc.get('file_type', ''),
                    'modified_time': doc.get('modified_time', ''),
                }
            )

            # Sparse 벡터 추가
            if sparse_vec:
                point.vector["sparse"] = sparse_vec

            points.append(point)

        client.upsert(collection_name="multi_vector", points=points)

        indexing_time = time.time() - start_time
        index_size = sum(f.stat().st_size for f in storage.rglob('*') if f.is_file()) / (1024 * 1024)

        logger.info(f"Multi-Vector indexing complete: {indexing_time:.2f}s, {index_size:.2f}MB")
        return indexing_time, index_size

    def create_combined_schema(self, storage_path: str, documents: List[Dict]) -> Tuple[float, float]:
        """
        Combined 스키마: Enhanced Metadata + Multi-Vector
        """
        logger.info("Creating Combined schema...")
        start_time = time.time()

        storage = Path(storage_path)
        storage.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(storage))

        # 컬렉션 생성
        if client.collection_exists("combined"):
            client.delete_collection("combined")

        client.create_collection(
            collection_name="combined",
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )

        # Payload Index 생성
        client.create_payload_index(
            collection_name="combined",
            field_name="keywords",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="combined",
            field_name="department",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="combined",
            field_name="document_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="combined",
            field_name="year",
            field_schema=PayloadSchemaType.INTEGER
        )

        # 임베딩 생성
        embedder = self._load_embedder()
        texts = [doc['text'] for doc in documents]
        embeddings_result = embedder.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        dense_vecs = embeddings_result['dense_vecs']
        sparse_vecs = embeddings_result.get('lexical_weights', [])

        # 포인트 생성
        points = []
        for i, doc in enumerate(documents):
            # 키워드/메타데이터 추출
            extracted = self.keyword_extractor.extract_keywords_rule_based(
                doc['text'], doc.get('file_path', '')
            )

            # Sparse 벡터
            sparse_vec = None
            if sparse_vecs and i < len(sparse_vecs):
                sparse_dict = sparse_vecs[i]
                if sparse_dict:
                    indices = list(sparse_dict.keys())
                    values = list(sparse_dict.values())
                    sparse_vec = SparseVector(indices=indices, values=values)

            vectors = {"dense": dense_vecs[i].tolist()}
            if sparse_vec:
                vectors["sparse"] = sparse_vec

            payload = {
                'original_id': doc['id'],
                'text': doc['text'],
                'file_name': doc.get('file_name', ''),
                'file_path': doc.get('file_path', ''),
                'file_type': doc.get('file_type', ''),
                'modified_time': doc.get('modified_time', ''),
                'keywords': extracted.get('keywords', []),
                'department': extracted.get('department', 'unknown'),
                'document_type': extracted.get('document_type', 'unknown'),
                'year': extracted.get('year'),
                'quarter': extracted.get('quarter'),
                'entities': extracted.get('entities', []),
            }

            point = PointStruct(
                id=self._string_to_int_id(doc['id']),
                vector=vectors,
                payload=payload
            )
            points.append(point)

        client.upsert(collection_name="combined", points=points)

        indexing_time = time.time() - start_time
        index_size = sum(f.stat().st_size for f in storage.rglob('*') if f.is_file()) / (1024 * 1024)

        logger.info(f"Combined indexing complete: {indexing_time:.2f}s, {index_size:.2f}MB")
        return indexing_time, index_size

    def evaluate_search(
        self,
        storage_path: str,
        collection_name: str,
        queries: List[Dict],
        use_named_vectors: bool = False
    ) -> Dict:
        """검색 성능 평가"""
        client = QdrantClient(path=str(storage_path))
        embedder = self._load_embedder()

        search_times = []
        filter_search_times = []
        mrr_scores = []
        recall_5_scores = []
        recall_10_scores = []

        for query_data in tqdm(queries, desc=f"Evaluating {collection_name}"):
            query = query_data['query']
            query_filter = query_data.get('filter', {})

            # 쿼리 임베딩
            query_embedding = embedder.encode(
                [query],
                return_dense=True,
                return_sparse=use_named_vectors,
                return_colbert_vecs=False
            )
            query_vec = query_embedding['dense_vecs'][0]

            # 일반 검색
            start = time.time()
            if use_named_vectors:
                results = client.query_points(
                    collection_name=collection_name,
                    query=query_vec.tolist(),
                    using="dense",
                    limit=10,
                    with_payload=True
                )
            else:
                results = client.query_points(
                    collection_name=collection_name,
                    query=query_vec.tolist(),
                    limit=10,
                    with_payload=True
                )
            search_time = (time.time() - start) * 1000  # ms
            search_times.append(search_time)

            # 필터 검색 (필터가 있는 경우)
            if query_filter:
                filter_conditions = []
                for key, value in query_filter.items():
                    if value is not None:
                        filter_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

                if filter_conditions:
                    start = time.time()
                    if use_named_vectors:
                        filter_results = client.query_points(
                            collection_name=collection_name,
                            query=query_vec.tolist(),
                            using="dense",
                            query_filter=Filter(must=filter_conditions),
                            limit=10,
                            with_payload=True
                        )
                    else:
                        filter_results = client.query_points(
                            collection_name=collection_name,
                            query=query_vec.tolist(),
                            query_filter=Filter(must=filter_conditions),
                            limit=10,
                            with_payload=True
                        )
                    filter_time = (time.time() - start) * 1000
                    filter_search_times.append(filter_time)

            # 메트릭 계산 (간단한 시뮬레이션 - 실제는 ground truth 필요)
            result_ids = [str(r.id) for r in results.points]

            # 상위 2개를 관련 문서로 가정 (실제 평가시 ground truth 사용)
            relevant_ids = result_ids[:2] if result_ids else []

            # MRR
            mrr = 0.0
            for i, rid in enumerate(result_ids):
                if rid in relevant_ids:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)

            # Recall@5, Recall@10
            recall_5 = len(set(result_ids[:5]) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0
            recall_10 = len(set(result_ids[:10]) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0
            recall_5_scores.append(recall_5)
            recall_10_scores.append(recall_10)

        return {
            'search_time_avg_ms': np.mean(search_times),
            'search_time_p95_ms': np.percentile(search_times, 95),
            'filter_search_time_avg_ms': np.mean(filter_search_times) if filter_search_times else 0,
            'mrr': np.mean(mrr_scores),
            'recall_at_5': np.mean(recall_5_scores),
            'recall_at_10': np.mean(recall_10_scores),
        }

    def run_experiment(self, output_path: str) -> List[SchemaEvalResult]:
        """전체 실험 실행"""
        results = []

        # 문서 로드
        documents = self._load_sample_documents()
        queries = self._load_test_queries()

        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")

        # 실험 저장소 경로
        exp_storage = self.base_storage_path / "schema_experiment"
        exp_storage.mkdir(parents=True, exist_ok=True)

        # 1. Baseline 스키마
        baseline_storage = exp_storage / "baseline"
        idx_time, idx_size = self.create_baseline_schema(str(baseline_storage), documents)
        eval_metrics = self.evaluate_search(str(baseline_storage), "baseline", queries, use_named_vectors=False)

        baseline_result = SchemaEvalResult(
            schema_name="Baseline",
            description="현재 스키마 (Dense 벡터 + 기본 메타데이터)",
            indexing_time_sec=idx_time,
            index_size_mb=idx_size,
            search_time_avg_ms=eval_metrics['search_time_avg_ms'],
            search_time_p95_ms=eval_metrics['search_time_p95_ms'],
            mrr=eval_metrics['mrr'],
            recall_at_5=eval_metrics['recall_at_5'],
            recall_at_10=eval_metrics['recall_at_10'],
            ndcg_at_5=0.0,  # 추후 계산
            filter_search_time_avg_ms=eval_metrics['filter_search_time_avg_ms'],
            filter_improvement_pct=0.0,  # 기준점
            weighted_score=0.0,
            total_queries=len(queries),
            total_documents=len(documents),
            eval_timestamp=datetime.now().isoformat()
        )
        results.append(baseline_result)

        # 2. Enhanced Metadata 스키마
        enhanced_storage = exp_storage / "enhanced_metadata"
        idx_time, idx_size = self.create_enhanced_metadata_schema(str(enhanced_storage), documents)
        eval_metrics = self.evaluate_search(str(enhanced_storage), "enhanced_metadata", queries, use_named_vectors=False)

        enhanced_result = SchemaEvalResult(
            schema_name="Enhanced Metadata",
            description="확장 메타데이터 + Payload Index (키워드, 부서, 문서유형, 연도)",
            indexing_time_sec=idx_time,
            index_size_mb=idx_size,
            search_time_avg_ms=eval_metrics['search_time_avg_ms'],
            search_time_p95_ms=eval_metrics['search_time_p95_ms'],
            mrr=eval_metrics['mrr'],
            recall_at_5=eval_metrics['recall_at_5'],
            recall_at_10=eval_metrics['recall_at_10'],
            ndcg_at_5=0.0,
            filter_search_time_avg_ms=eval_metrics['filter_search_time_avg_ms'],
            filter_improvement_pct=0.0,
            weighted_score=0.0,
            total_queries=len(queries),
            total_documents=len(documents),
            eval_timestamp=datetime.now().isoformat()
        )
        results.append(enhanced_result)

        # 3. Multi-Vector 스키마
        multi_storage = exp_storage / "multi_vector"
        idx_time, idx_size = self.create_multi_vector_schema(str(multi_storage), documents)
        eval_metrics = self.evaluate_search(str(multi_storage), "multi_vector", queries, use_named_vectors=True)

        multi_result = SchemaEvalResult(
            schema_name="Multi-Vector",
            description="Dense + Sparse Named Vectors (BGE-M3 활용)",
            indexing_time_sec=idx_time,
            index_size_mb=idx_size,
            search_time_avg_ms=eval_metrics['search_time_avg_ms'],
            search_time_p95_ms=eval_metrics['search_time_p95_ms'],
            mrr=eval_metrics['mrr'],
            recall_at_5=eval_metrics['recall_at_5'],
            recall_at_10=eval_metrics['recall_at_10'],
            ndcg_at_5=0.0,
            filter_search_time_avg_ms=eval_metrics['filter_search_time_avg_ms'],
            filter_improvement_pct=0.0,
            weighted_score=0.0,
            total_queries=len(queries),
            total_documents=len(documents),
            eval_timestamp=datetime.now().isoformat()
        )
        results.append(multi_result)

        # 4. Combined 스키마
        combined_storage = exp_storage / "combined"
        idx_time, idx_size = self.create_combined_schema(str(combined_storage), documents)
        eval_metrics = self.evaluate_search(str(combined_storage), "combined", queries, use_named_vectors=True)

        combined_result = SchemaEvalResult(
            schema_name="Combined",
            description="Enhanced Metadata + Multi-Vector (전체 적용)",
            indexing_time_sec=idx_time,
            index_size_mb=idx_size,
            search_time_avg_ms=eval_metrics['search_time_avg_ms'],
            search_time_p95_ms=eval_metrics['search_time_p95_ms'],
            mrr=eval_metrics['mrr'],
            recall_at_5=eval_metrics['recall_at_5'],
            recall_at_10=eval_metrics['recall_at_10'],
            ndcg_at_5=0.0,
            filter_search_time_avg_ms=eval_metrics['filter_search_time_avg_ms'],
            filter_improvement_pct=0.0,
            weighted_score=0.0,
            total_queries=len(queries),
            total_documents=len(documents),
            eval_timestamp=datetime.now().isoformat()
        )
        results.append(combined_result)

        # 필터 개선율 및 종합 점수 계산
        baseline_filter_time = results[0].filter_search_time_avg_ms
        for result in results:
            if baseline_filter_time > 0 and result.filter_search_time_avg_ms > 0:
                result.filter_improvement_pct = (
                    (baseline_filter_time - result.filter_search_time_avg_ms) / baseline_filter_time * 100
                )

        # 종합 점수 계산
        for result in results:
            result.weighted_score = self._compute_weighted_score(result, results)

        # 결과 저장
        self._save_results(results, output_path)
        self._print_comparison_table(results)

        return results

    def _compute_weighted_score(self, result: SchemaEvalResult, all_results: List[SchemaEvalResult]) -> float:
        """종합 점수 계산"""
        # 정규화를 위한 최소/최대값
        search_times = [r.search_time_avg_ms for r in all_results]
        idx_times = [r.indexing_time_sec for r in all_results]

        # 정규화 (낮을수록 좋음 -> 1로 변환)
        norm_search_time = 1 - (result.search_time_avg_ms - min(search_times)) / (max(search_times) - min(search_times) + 1e-6)
        norm_idx_time = 1 - (result.indexing_time_sec - min(idx_times)) / (max(idx_times) - min(idx_times) + 1e-6)

        # 가중 점수
        score = (
            result.mrr * 0.30 +           # 검색 품질 30%
            result.recall_at_5 * 0.20 +   # Recall@5 20%
            norm_search_time * 0.25 +     # 검색 속도 25%
            norm_idx_time * 0.15 +        # 인덱싱 속도 15%
            (result.filter_improvement_pct / 100 + 1) * 0.10  # 필터 개선 10%
        )

        return score

    def _save_results(self, results: List[SchemaEvalResult], output_path: str):
        """결과 저장"""
        results_dict = [asdict(r) for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")

    def _print_comparison_table(self, results: List[SchemaEvalResult]):
        """비교표 출력"""
        sorted_results = sorted(results, key=lambda r: r.weighted_score, reverse=True)

        print("\n" + "=" * 160)
        print("VectorDB 스키마 비교 결과")
        print("=" * 160)
        print(f"{'스키마':<20} {'인덱싱(s)':>10} {'크기(MB)':>10} {'검색(ms)':>10} {'P95(ms)':>10} {'MRR':>8} {'R@5':>8} {'R@10':>8} {'필터(ms)':>10} {'필터개선%':>10} {'종합점수':>10}")
        print("-" * 160)

        for r in sorted_results:
            print(f"{r.schema_name:<20} {r.indexing_time_sec:>10.2f} {r.index_size_mb:>10.2f} {r.search_time_avg_ms:>10.2f} {r.search_time_p95_ms:>10.2f} {r.mrr:>8.4f} {r.recall_at_5:>8.4f} {r.recall_at_10:>8.4f} {r.filter_search_time_avg_ms:>10.2f} {r.filter_improvement_pct:>10.1f} {r.weighted_score:>10.4f}")

        print("=" * 160)

        # 최적 스키마 추천
        best = sorted_results[0]
        print(f"\n최적 스키마: {best.schema_name}")
        print(f"  - 설명: {best.description}")
        print(f"  - 종합 점수: {best.weighted_score:.4f}")
        print(f"  - 검색 속도: {best.search_time_avg_ms:.2f}ms (P95: {best.search_time_p95_ms:.2f}ms)")
        print(f"  - 검색 품질 (MRR): {best.mrr:.4f}")
        print(f"  - 인덱싱 시간: {best.indexing_time_sec:.2f}초")

        # Baseline 대비 개선율
        baseline = next(r for r in results if r.schema_name == "Baseline")
        if best.schema_name != "Baseline":
            mrr_improvement = (best.mrr - baseline.mrr) / baseline.mrr * 100 if baseline.mrr > 0 else 0
            speed_improvement = (baseline.search_time_avg_ms - best.search_time_avg_ms) / baseline.search_time_avg_ms * 100
            print(f"\nBaseline 대비 개선:")
            print(f"  - MRR 개선: {mrr_improvement:+.1f}%")
            print(f"  - 검색 속도 개선: {speed_improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='VectorDB 스키마 비교 실험')
    parser.add_argument('--base-storage', type=str,
                        default='/dais04/DO_NOT_DELETE/HD_AI_Hackathon',
                        help='기본 저장소 경로')
    parser.add_argument('--test-queries', type=str,
                        default='experiments/test_queries.json',
                        help='테스트 쿼리 파일')
    parser.add_argument('--output', type=str,
                        default='experiments/results/schema_comparison_results.json',
                        help='결과 저장 경로')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    test_queries_path = project_root / args.test_queries
    output_path = project_root / args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = SchemaExperiment(
        base_storage_path=args.base_storage,
        test_queries_path=str(test_queries_path) if test_queries_path.exists() else None
    )

    results = experiment.run_experiment(str(output_path))

    logger.info(f"\nExperiment complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
