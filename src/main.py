"""
메인 애플리케이션
Gradio 기반 챗봇 UI + 전체 파이프라인 통합
"""

import sys
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import gradio as gr
import numpy as np
import hashlib
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.multimodal_parser import MultimodalParser
from src.embeddings.embedding_model import BGEM3Embedder
from src.search.vector_store import QdrantVectorStore
from src.search.bm25_search import BM25SearchEngine
from src.search.hybrid_search import HybridSearchEngine
from src.llm.qwen_model import QwenSummarizer, LLMConfig
from src.recommend.recommender import FileRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIAgentPipeline:
    """전체 AI Agent 파이프라인"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 컴포넌트 초기화
        self._initialize_components()

    def _initialize_components(self):
        """모든 컴포넌트 초기화"""
        logger.info("Initializing AI Agent components...")

        self.llm_enabled = bool(self.config.get('llm', {}).get('enabled', False))

        # 1. 임베딩 모델
        logger.info("Loading embedding model...")
        self.embedder = BGEM3Embedder(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device'],
            use_fp16=self.config['embedding']['use_fp16']
        )

        # 2. 벡터 스토어
        logger.info("Connecting to vector store...")
        self.vector_store = QdrantVectorStore(
            storage_path=self.config['data']['qdrant_storage'],
            collection_name=self.config['qdrant']['collection_name']
        )

        # 3. BM25 검색 엔진
        logger.info("Loading BM25 index...")
        self.bm25_engine = BM25SearchEngine(use_korean_tokenizer=True)

        bm25_index_path = Path(self.config['data']['cache_dir']) / 'bm25_index.pkl'
        if bm25_index_path.exists():
            self.bm25_engine.load_index(str(bm25_index_path))
        else:
            logger.warning("BM25 index not found. Please run indexing first.")

        # 4. 하이브리드 검색 엔진
        self.hybrid_engine = HybridSearchEngine(
            vector_store=self.vector_store,
            bm25_engine=self.bm25_engine,
            bm25_weight=self.config['search']['bm25_weight'],
            vector_weight=self.config['search']['semantic_weight']
        )

        # 5. LLM (프로토타입에서는 선택적으로 로드)
        self.summarizer = None
        if self.llm_enabled:
            try:
                logger.info("Loading LLM (this may take a while)...")
                llm_config = LLMConfig(
                    model_name=self.config['llm']['model_name'],
                    device=self.config['llm']['device'],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens'],
                    use_vllm=False  # 프로토타입에서는 transformers 사용
                )
                self.summarizer = QwenSummarizer(llm_config)
            except Exception as e:
                logger.warning(f"LLM loading failed (optional): {e}")
        else:
            logger.info("LLM disabled via config. Skipping LLM load.")

        # 6. 추천 시스템
        self.recommender = FileRecommender(
            temporal_window_hours=self.config['recommendation']['temporal_window_hours']
        )

        logger.info("All components initialized successfully!")

    def search_files(
        self,
        query: str,
        top_k: int = 5,
        include_summary: bool = True,
        include_recommendations: bool = True
    ) -> Dict:
        """
        파일 검색 메인 함수

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            include_summary: 요약 포함 여부
            include_recommendations: 추천 포함 여부

        Returns:
            검색 결과 딕셔너리
        """
        logger.info(f"Searching for: '{query}'")

        # 1. 쿼리 임베딩 생성
        query_embedding = self.embedder.encode_queries(query)

        # 2. 하이브리드 검색
        raw_results = self.hybrid_engine.search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.config['search']['top_k'],
            final_top_k=top_k
        )

        # 2-1. 파일 단위로 결과 집계 (청크 중복 제거)
        results = self._aggregate_results_by_file(raw_results)

        logger.info(f"Found {len(results)} results")

        # 3. 요약 생성 (선택적)
        if include_summary and self.summarizer:
            logger.info("Generating summaries...")
            for result in results:
                text = result.get('text', '') or result.get('metadata', {}).get('text', '')
                if text:
                    try:
                        summary = self.summarizer.summarize(
                            text[:4000],  # 길이 제한
                            style="bullet_points"
                        )
                        result['summary'] = summary
                    except Exception as e:
                        logger.warning(f"Summary generation failed: {e}")
                        result['summary'] = "요약 생성 실패"
        else:
            for result in results:
                result['summary'] = "요약 미사용"

        # 4. 연관 파일 추천 (선택적)
        # 빈 결과 처리: results가 비어있으면 바로 반환
        if not results:
            return {
                'query': query,
                'results': [],
                'total_found': 0
            }

        if include_recommendations:
            logger.info("Generating recommendations...")
            # 첫 번째 결과에 대한 추천만 생성 (프로토타입)
            top_result = results[0]

            # 모든 문서 가져오기 (벡터 포함하여 추천 정확도 향상)
            all_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=50,
                with_vectors=True  # 벡터 포함하여 추천에 활용
            )

            # top_result가 검색 결과에 없으면 포함 (벡터 조회 시도)
            candidate_map = {r['id']: r for r in all_results}
            if top_result['id'] not in candidate_map:
                fetched = self.vector_store.get_document(top_result['id'])
                candidate_map[top_result['id']] = {
                    'id': top_result['id'],
                    'score': top_result.get('score', 0),
                    'payload': (fetched or {}).get('payload', {}) or top_result.get('metadata', {}),
                    'vector': (fetched or {}).get('vector'),
                    'metadata': top_result.get('metadata', {})
                }

            all_results = list(candidate_map.values())

            # 추천 생성
            recommendations = []
            if len(all_results) > 1:
                # 메타데이터 또는 payload에서 정보 추출 (fallback 처리)
                def get_file_info(r):
                    meta = r.get('metadata', {}) or {}
                    payload = r.get('payload', {}) or {}
                    file_id = meta.get('file_id') or payload.get('file_id')
                    if not file_id:
                        rid = r.get('id', '')
                        file_id = rid.split('_chunk_')[0] if '_chunk_' in rid else rid
                    return {
                        'id': r['id'],
                        'file_id': file_id,
                        'path': meta.get('file_path') or payload.get('file_path', ''),
                        'file_type': meta.get('file_type') or payload.get('file_type', ''),
                        'modified_time': meta.get('modified_time') or payload.get('modified_time', ''),
                        'file_name': meta.get('file_name') or payload.get('file_name', '')
                    }

                target_file = get_file_info(top_result)
                candidate_files = [get_file_info(r) for r in all_results]

                # 임베딩 수집 (추천 정확도 향상)
                candidate_embeddings = None
                target_embedding_for_rec = None

                # 타겟 문서 임베딩 조회 (없으면 텍스트 재임베딩)
                target_embedding_for_rec = self._get_vector_for_result(top_result)

                # candidate_embeddings 수집 (with_vectors=True로 검색한 결과에서)
                try:
                    vectors_list = []
                    for r in all_results:
                        if 'vector' in r and r['vector'] is not None:
                            vectors_list.append(np.array(r['vector'], dtype=np.float32))
                        else:
                            vectors_list.append(np.zeros(self.embedder.get_embedding_dim(), dtype=np.float32))

                    if vectors_list:
                        candidate_embeddings = np.vstack(vectors_list)
                except Exception as e:
                    logger.warning(f"Failed to get embeddings for recommendations: {e}")
                    candidate_embeddings = None

                recommendations = self.recommender.recommend_similar_files(
                    target_file=target_file,
                    candidate_files=candidate_files,
                    target_embedding=target_embedding_for_rec,
                    candidate_embeddings=candidate_embeddings,
                    top_k=5
                )

            results[0]['recommendations'] = recommendations
        else:
            results[0]['recommendations'] = []

        return {
            'query': query,
            'results': results,
            'total_found': len(results)
        }

    def _extract_file_id(self, result: Dict) -> str:
        """결과에서 file_id 추출 (없으면 chunk_id 기반 생성)"""
        meta = result.get('metadata', {}) or {}
        payload = result.get('payload', {}) or {}
        file_id = meta.get('file_id') or payload.get('file_id')

        if not file_id:
            rid = result.get('id', '')
            file_id = rid.split('_chunk_')[0] if '_chunk_' in rid else rid

        return file_id

    def _aggregate_results_by_file(self, results: List[Dict]) -> List[Dict]:
        """청크 결과를 파일 단위로 집계 (최고 점수 청크만 유지)"""
        best_by_file = {}

        for res in results:
            file_id = self._extract_file_id(res)
            meta = res.get('metadata', {}) or {}
            payload = res.get('payload', {}) or {}

            merged_meta = {**payload, **meta}
            merged_meta['file_id'] = file_id

            candidate = {
                'id': res.get('id'),
                'score': res.get('score', 0.0),
                'text': res.get('text', '') or merged_meta.get('text', ''),
                'metadata': merged_meta,
                'payload': res.get('payload', {}) or merged_meta,
                'best_chunk_id': res.get('id')
            }

            prev = best_by_file.get(file_id)
            if (prev is None) or (candidate['score'] > prev['score']):
                best_by_file[file_id] = candidate

        aggregated = list(best_by_file.values())
        aggregated.sort(key=lambda r: r['score'], reverse=True)
        return aggregated

    def _get_vector_for_result(self, result: Dict) -> np.ndarray:
        """검색 결과(대표 청크)에 대응하는 벡터 조회 또는 재임베딩"""
        chunk_id = result.get('best_chunk_id') or result.get('id')
        vector = None

        try:
            doc = self.vector_store.get_document(chunk_id)
            if doc and doc.get('vector') is not None:
                vector = np.array(doc['vector'], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Vector fetch failed for {chunk_id}: {e}")

        if vector is None:
            text = result.get('text') or result.get('metadata', {}).get('text', '')
            if text:
                try:
                    emb = self.embedder.encode_documents(
                        [text],
                        include_sparse=False
                    )
                    vector = emb['dense_vecs'][0]
                except Exception as e:
                    logger.debug(f"Re-embedding failed for {chunk_id}: {e}")

        if vector is None:
            vector = np.zeros(self.embedder.get_embedding_dim(), dtype=np.float32)

        return vector

    def detect_duplicates(
        self,
        similarity_threshold: float = 0.95,
        top_k: int = 50
    ) -> List[Dict]:
        """
        중복 문서 탐지 (해시 + 벡터 유사도 기반)

        Args:
            similarity_threshold: 중복 판단 유사도 임계값 (0.95 = 95% 유사)
            top_k: 검사할 문서 수

        Returns:
            중복 문서 그룹 리스트
            [{'original': {...}, 'duplicates': [{...}, ...], 'similarity': float}]
        """
        logger.info(f"Detecting duplicate documents (threshold: {similarity_threshold})")

        # 모든 문서 가져오기
        all_docs = []
        try:
            # 벡터 스토어에서 문서 검색 (임의 벡터로 전체 조회)
            dummy_vector = np.zeros(self.embedder.get_embedding_dim(), dtype=np.float32)
            all_docs = self.vector_store.search(
                query_vector=dummy_vector,
                top_k=top_k,
                with_vectors=True
            )
        except Exception as e:
            logger.error(f"Failed to fetch documents for duplicate detection: {e}")
            return []

        if len(all_docs) < 2:
            return []

        # 파일 단위로 집계 (청크 중복 제거)
        file_docs = {}
        for doc in all_docs:
            file_id = self._extract_file_id(doc)
            if file_id not in file_docs:
                file_docs[file_id] = doc

        docs_list = list(file_docs.values())

        # 콘텐츠 해시 계산
        content_hashes = {}
        for doc in docs_list:
            text = doc.get('text', '') or doc.get('payload', {}).get('text', '')
            if text:
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                content_hashes[doc['id']] = text_hash

        # 중복 그룹 탐지
        duplicate_groups = []
        checked_ids = set()

        for i, doc in enumerate(docs_list):
            if doc['id'] in checked_ids:
                continue

            doc_vector = doc.get('vector')
            if doc_vector is None:
                continue

            doc_vector = np.array(doc_vector, dtype=np.float32)
            doc_hash = content_hashes.get(doc['id'])

            duplicates = []

            for j, other_doc in enumerate(docs_list[i+1:], i+1):
                if other_doc['id'] in checked_ids:
                    continue

                other_vector = other_doc.get('vector')
                if other_vector is None:
                    continue

                other_vector = np.array(other_vector, dtype=np.float32)
                other_hash = content_hashes.get(other_doc['id'])

                # 1. 해시 비교 (완전 일치)
                is_hash_match = doc_hash and other_hash and doc_hash == other_hash

                # 2. 벡터 유사도 계산 (코사인 유사도)
                similarity = 0.0
                norm_a = np.linalg.norm(doc_vector)
                norm_b = np.linalg.norm(other_vector)
                if norm_a > 0 and norm_b > 0:
                    similarity = float(np.dot(doc_vector, other_vector) / (norm_a * norm_b))

                # 중복 판단
                if is_hash_match or similarity >= similarity_threshold:
                    other_meta = other_doc.get('payload', {}) or other_doc.get('metadata', {})
                    duplicates.append({
                        'id': other_doc['id'],
                        'file_name': other_meta.get('file_name', 'Unknown'),
                        'file_path': other_meta.get('file_path', 'N/A'),
                        'similarity': 1.0 if is_hash_match else similarity,
                        'match_type': 'hash' if is_hash_match else 'semantic'
                    })
                    checked_ids.add(other_doc['id'])

            if duplicates:
                doc_meta = doc.get('payload', {}) or doc.get('metadata', {})
                duplicate_groups.append({
                    'original': {
                        'id': doc['id'],
                        'file_name': doc_meta.get('file_name', 'Unknown'),
                        'file_path': doc_meta.get('file_path', 'N/A')
                    },
                    'duplicates': duplicates,
                    'count': len(duplicates)
                })
                checked_ids.add(doc['id'])

        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups

    def answer_question(self, question: str, context: str) -> str:
        """
        LLM을 사용하여 질문에 답변

        Args:
            question: 사용자 질문
            context: 검색 결과 컨텍스트

        Returns:
            LLM 응답
        """
        if not self.summarizer:
            return "LLM이 비활성화되어 있어 질문 답변 기능을 사용할 수 없습니다."

        try:
            # QwenSummarizer.answer_question(context, question) 순서에 맞춤
            response = self.summarizer.answer_question(context, question)
            return response
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return f"답변 생성 중 오류 발생: {str(e)}"


class GradioChatInterface:
    """Gradio 챗봇 UI - 대화형 파일 검색 인터페이스"""

    def __init__(self, pipeline: AIAgentPipeline):
        self.pipeline = pipeline
        self.conversation_history = []
        self.last_search_results = None

    def _parse_command(self, message: str) -> Tuple[str, Dict]:
        """
        사용자 메시지에서 명령어 파싱

        Returns:
            (command_type, params)
            command_type: 'search', 'duplicate', 'recommend', 'help', 'question'
        """
        message_lower = message.lower().strip()

        # 중복 문서 탐지 명령
        if any(kw in message_lower for kw in ['중복', '중복 탐지', '중복 검사', 'duplicate']):
            return 'duplicate', {}

        # 도움말 명령
        if any(kw in message_lower for kw in ['/help', '도움말', '사용법']):
            return 'help', {}

        # 추천 명령 (이전 검색 결과 기반)
        if any(kw in message_lower for kw in ['추천', '유사한 파일', 'similar', '연관']):
            if self.last_search_results:
                return 'recommend', {}

        # LLM 질문 (이전 검색 결과가 있고, '?' 또는 질문형 문장인 경우)
        if self.last_search_results and self.pipeline.summarizer:
            if '?' in message or any(kw in message_lower for kw in ['뭐야', '뭔가요', '알려', '설명', '어떻게']):
                return 'question', {'question': message}

        # 기본: 검색
        return 'search', {'query': message}

    def chat_response(
        self,
        message: str,
        history: List[Dict],
        top_k: int,
        include_summary: bool,
        include_recommendations: bool,
        show_explanation: bool
    ) -> Tuple[str, List[Dict]]:
        """
        챗봇 응답 생성

        Args:
            message: 사용자 메시지
            history: 대화 히스토리 (Gradio 6.x 형식)
            top_k: 검색 결과 수
            include_summary: 요약 포함 여부
            include_recommendations: 추천 포함 여부
            show_explanation: 검색 설명 표시 여부

        Returns:
            (응답 텍스트, 업데이트된 히스토리)
        """
        if not message.strip():
            return "", history

        try:
            command, params = self._parse_command(message)

            if command == 'help':
                response = self._get_help_message()

            elif command == 'duplicate':
                response = self._handle_duplicate_detection()

            elif command == 'recommend':
                response = self._handle_recommendation()

            elif command == 'question':
                response = self._handle_question(params['question'])

            else:  # search
                response = self._handle_search(
                    params['query'],
                    top_k,
                    include_summary,
                    include_recommendations,
                    show_explanation
                )

            # 히스토리 업데이트 (Gradio 6.x 형식)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_response = f"오류가 발생했습니다: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history

    def _handle_search(
        self,
        query: str,
        top_k: int,
        include_summary: bool,
        include_recommendations: bool,
        show_explanation: bool
    ) -> str:
        """검색 처리"""
        search_result = self.pipeline.search_files(
            query=query,
            top_k=top_k,
            include_summary=include_summary,
            include_recommendations=include_recommendations
        )

        # 검색 결과 저장 (후속 질문용)
        self.last_search_results = search_result

        return self._format_search_results(search_result, show_explanation, include_recommendations)

    def _handle_duplicate_detection(self) -> str:
        """중복 문서 탐지 처리"""
        duplicates = self.pipeline.detect_duplicates(
            similarity_threshold=0.95,
            top_k=100
        )

        return self._format_duplicates(duplicates)

    def _handle_recommendation(self) -> str:
        """추천 처리 (마지막 검색 결과 기반)"""
        if not self.last_search_results or not self.last_search_results['results']:
            return "이전 검색 결과가 없습니다. 먼저 검색을 수행해주세요."

        top_result = self.last_search_results['results'][0]
        recommendations = top_result.get('recommendations', [])

        return self._format_recommendations(recommendations)

    def _handle_question(self, question: str) -> str:
        """질문 답변 처리 (LLM 사용)"""
        if not self.last_search_results or not self.last_search_results['results']:
            return "이전 검색 결과가 없습니다. 먼저 검색을 수행해주세요."

        # 컨텍스트 구성
        context_parts = []
        for i, result in enumerate(self.last_search_results['results'][:3]):
            file_name = result.get('metadata', {}).get('file_name', 'Unknown')
            text = result.get('text', '')[:1000]
            context_parts.append(f"[문서 {i+1}: {file_name}]\n{text}")

        context = "\n\n".join(context_parts)

        # LLM 응답 생성
        response = self.pipeline.answer_question(question, context)

        return f"**질문:** {question}\n\n**답변:**\n{response}"

    def _get_help_message(self) -> str:
        """도움말 메시지"""
        return """## 사내 파일 검색 AI Agent 사용법

### 기본 기능
- **파일 검색**: 자연어로 검색어를 입력하세요.
  - 예: "2024년 매출 보고서", "고객 만족도 향상 전략"

### 특수 명령
- **중복 탐지**: "중복 검사" 또는 "중복 탐지"를 입력하면 유사한 문서를 찾아줍니다.
- **추천**: 검색 후 "추천" 또는 "유사한 파일"을 입력하면 연관 파일을 추천합니다.
- **질문**: 검색 후 검색 결과에 대해 질문할 수 있습니다. (LLM 활성화 필요)

### 검색 옵션
- **결과 개수**: 슬라이더로 검색 결과 수를 조절합니다.
- **요약 생성**: 검색된 문서의 요약을 생성합니다. (LLM 활성화 필요)
- **연관 파일 추천**: 검색 결과와 유사한 파일을 추천합니다.
- **검색 설명 표시**: 검색 결과에 대한 상세 설명(매칭 키워드, 점수 분해)을 표시합니다.

### 팁
- 구체적인 키워드를 사용하면 더 정확한 결과를 얻을 수 있습니다.
- 한글과 영어 모두 지원됩니다.
"""

    def _format_search_results(
        self,
        search_result: Dict,
        show_explanation: bool,
        include_recommendations: bool
    ) -> str:
        """검색 결과 포맷팅 (설명 포함)"""
        if not search_result['results']:
            return f"'{search_result['query']}'에 대한 검색 결과가 없습니다."

        output = f"## 검색 결과 (총 {search_result['total_found']}개)\n\n"

        for i, result in enumerate(search_result['results']):
            meta = result.get('metadata', {})
            file_name = meta.get('file_name', 'Unknown')
            file_path = meta.get('file_path', 'N/A')
            file_type = meta.get('file_type', 'N/A')
            score = result.get('score', 0)

            output += f"### {i+1}. {file_name}\n"
            output += f"- **경로:** `{file_path}`\n"
            output += f"- **타입:** {file_type}\n"
            output += f"- **통합 점수:** {score:.4f}\n"

            # 검색 설명 (근거) 표시
            if show_explanation and 'explanation' in result:
                exp = result['explanation']
                output += "\n**검색 근거:**\n"

                # 매칭 타입
                search_types = exp.get('search_type', [])
                type_str = ', '.join(['키워드' if t == 'keyword' else '의미' for t in search_types])
                output += f"- 매칭 방식: {type_str or 'N/A'}\n"

                # 점수 분해
                bm25_score = exp.get('bm25_score', 0)
                vector_score = exp.get('vector_score', 0)
                output += f"- BM25(키워드) 점수: {bm25_score:.4f}\n"
                output += f"- 벡터(의미) 점수: {vector_score:.4f}\n"

                # 매칭 키워드
                matched = exp.get('matched_keywords', [])
                if matched:
                    output += f"- 매칭 키워드: {', '.join(matched[:5])}\n"

            # 요약
            if 'summary' in result and result['summary'] != "요약 미사용":
                output += f"\n**요약:** {result['summary']}\n"

            # 내용 미리보기
            text_preview = result.get('text', '')[:200]
            if not text_preview:
                text_preview = meta.get('text', '')[:200]
            if text_preview:
                output += f"\n**미리보기:** {text_preview}...\n"

            output += "\n---\n"

        # 추천 파일 (첫 번째 결과에 대해)
        if include_recommendations and search_result['results']:
            recommendations = search_result['results'][0].get('recommendations', [])
            if recommendations:
                output += "\n## 연관 파일 추천\n"
                for i, rec in enumerate(recommendations[:3]):
                    output += f"- **{rec.get('file_name', 'Unknown')}** (점수: {rec.get('recommendation_score', 0):.2f})\n"

        return output

    def _format_duplicates(self, duplicate_groups: List[Dict]) -> str:
        """중복 문서 포맷팅"""
        if not duplicate_groups:
            return "중복 문서가 발견되지 않았습니다."

        output = f"## 중복 문서 탐지 결과\n\n"
        output += f"총 **{len(duplicate_groups)}개** 중복 그룹 발견\n\n"

        for i, group in enumerate(duplicate_groups[:10]):  # 상위 10개만 표시
            original = group['original']
            duplicates = group['duplicates']

            output += f"### 그룹 {i+1}: {original['file_name']}\n"
            output += f"- **원본:** `{original['file_path']}`\n"
            output += f"- **중복 수:** {group['count']}개\n\n"

            output += "| 파일명 | 유사도 | 탐지 방식 |\n"
            output += "|--------|--------|----------|\n"

            for dup in duplicates[:5]:  # 각 그룹당 5개까지
                similarity_pct = dup['similarity'] * 100
                match_type = '해시 일치' if dup['match_type'] == 'hash' else '의미 유사'
                output += f"| {dup['file_name'][:30]} | {similarity_pct:.1f}% | {match_type} |\n"

            output += "\n"

        return output

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """추천 파일 포맷팅"""
        if not recommendations:
            return "추천할 연관 파일이 없습니다."

        output = "## 연관 파일 추천\n\n"

        for i, rec in enumerate(recommendations):
            output += f"### {i+1}. {rec.get('file_name', 'Unknown')}\n"
            output += f"- **경로:** `{rec.get('path', 'N/A')}`\n"
            output += f"- **추천 점수:** {rec.get('recommendation_score', 0):.4f}\n"

            # 유사도 세부사항
            breakdown = rec.get('similarity_breakdown', {})
            output += "- **유사도 상세:**\n"
            output += f"  - 내용 유사도: {breakdown.get('vector', 0):.2f}\n"
            output += f"  - 시간 연관성: {breakdown.get('temporal', 0):.2f}\n"
            output += f"  - 경로 유사도: {breakdown.get('path', 0):.2f}\n"
            output += f"  - 타입 일치: {breakdown.get('type', 0):.2f}\n\n"

        return output

    def create_ui(self):
        """Gradio 챗봇 UI 생성"""
        llm_available = self.pipeline.summarizer is not None

        with gr.Blocks(title="사내 파일 검색 AI Agent") as demo:
            gr.Markdown("# 사내 네트워크 드라이브 파일 검색 AI Agent")
            gr.Markdown("자연어로 파일을 검색하고, 관련 문서를 추천받으세요. 대화형 인터페이스로 편리하게 사용할 수 있습니다.")

            # 상태 표시
            status_text = "LLM 활성화" if llm_available else "LLM 비활성화 (요약/질문답변 불가)"
            status_color = "green" if llm_available else "orange"
            gr.Markdown(f"> **상태:** <span style='color:{status_color}'>{status_text}</span>")

            with gr.Row():
                # 메인 채팅 영역
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="대화",
                        height=500
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="메시지 입력",
                            placeholder="검색어를 입력하세요. 예: 2024년 매출 보고서, 중복 검사, /help",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("전송", variant="primary", scale=1)

                # 설정 패널
                with gr.Column(scale=1):
                    gr.Markdown("### 검색 설정")

                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="검색 결과 수"
                    )

                    show_explanation = gr.Checkbox(
                        label="검색 설명 표시 (매칭 근거)",
                        value=True
                    )

                    include_recommendations = gr.Checkbox(
                        label="연관 파일 추천",
                        value=True
                    )

                    summary_label = "요약 생성" if llm_available else "요약 생성 (비활성화)"
                    include_summary = gr.Checkbox(
                        label=summary_label,
                        value=False,
                        interactive=llm_available
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 빠른 명령")

                    help_btn = gr.Button("도움말", size="sm")
                    duplicate_btn = gr.Button("중복 문서 탐지", size="sm")
                    clear_btn = gr.Button("대화 초기화", size="sm", variant="secondary")

            # 예시 질의
            gr.Markdown("### 예시 질의")
            gr.Examples(
                examples=[
                    ["2024년 매출 보고서"],
                    ["고객 만족도 향상 전략"],
                    ["AI 자동화 프로젝트"],
                    ["중복 검사"],
                    ["/help"]
                ],
                inputs=msg_input
            )

            # 이벤트 핸들러
            def submit_message(message, history, top_k, include_summary, include_recommendations, show_explanation):
                return self.chat_response(
                    message, history, top_k, include_summary, include_recommendations, show_explanation
                )

            # 전송 버튼 클릭
            send_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot, top_k_slider, include_summary, include_recommendations, show_explanation],
                outputs=[msg_input, chatbot]
            )

            # Enter 키로 전송
            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot, top_k_slider, include_summary, include_recommendations, show_explanation],
                outputs=[msg_input, chatbot]
            )

            # 도움말 버튼
            def show_help(history):
                help_msg = self._get_help_message()
                history.append(["도움말", help_msg])
                return history

            help_btn.click(
                fn=show_help,
                inputs=[chatbot],
                outputs=[chatbot]
            )

            # 중복 탐지 버튼
            def run_duplicate(history):
                response = self._handle_duplicate_detection()
                history.append(["중복 문서 탐지", response])
                return history

            duplicate_btn.click(
                fn=run_duplicate,
                inputs=[chatbot],
                outputs=[chatbot]
            )

            # 대화 초기화
            def clear_chat():
                self.last_search_results = None
                return []

            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot]
            )

        return demo


# 기존 단일 검색 UI (하위 호환성)
class GradioInterface:
    """Gradio 단일 검색 UI (레거시)"""

    def __init__(self, pipeline: AIAgentPipeline):
        self.pipeline = pipeline

    def search_interface(self, query: str, top_k: int, include_summary: bool, include_recommendations: bool):
        if not query.strip():
            return "검색어를 입력해주세요.", ""

        try:
            search_result = self.pipeline.search_files(
                query=query,
                top_k=top_k,
                include_summary=include_summary,
                include_recommendations=include_recommendations
            )

            output = f"# 검색 결과 (총 {search_result['total_found']}개)\n\n"
            for i, result in enumerate(search_result['results']):
                meta = result.get('metadata', {})
                output += f"## {i+1}. {meta.get('file_name', 'Unknown')}\n"
                output += f"**경로:** `{meta.get('file_path', 'N/A')}`\n"
                output += f"**점수:** {result['score']:.4f}\n\n"

            recommendations_output = ""
            if include_recommendations and search_result['results']:
                recs = search_result['results'][0].get('recommendations', [])
                if recs:
                    recommendations_output = "# 추천 파일\n" + "\n".join([f"- {r['file_name']}" for r in recs])

            return output, recommendations_output

        except Exception as e:
            return f"오류: {e}", ""

    def create_ui(self):
        with gr.Blocks(title="파일 검색") as demo:
            gr.Markdown("# 파일 검색")
            query_input = gr.Textbox(label="검색어")
            top_k = gr.Slider(1, 10, 5, step=1, label="결과 수")
            summary_check = gr.Checkbox(label="요약", value=False)
            recommend_check = gr.Checkbox(label="추천", value=True)
            btn = gr.Button("검색")
            results = gr.Markdown()
            recs = gr.Markdown()
            btn.click(self.search_interface, [query_input, top_k, summary_check, recommend_check], [results, recs])
        return demo


def main():
    # 설정 파일 경로
    config_path = project_root / "config" / "config.yaml"

    # 파이프라인 초기화
    logger.info("Starting AI Agent...")
    pipeline = AIAgentPipeline(str(config_path))

    # 챗봇 UI 생성
    ui = GradioChatInterface(pipeline)
    demo = ui.create_ui()

    # 서버 시작
    logger.info("Launching Gradio Chatbot interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        quiet=False
    )


if __name__ == "__main__":
    main()
