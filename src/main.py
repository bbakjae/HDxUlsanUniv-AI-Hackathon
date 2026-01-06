"""
메인 애플리케이션
Gradio 기반 챗봇 UI + 전체 파이프라인 통합
"""

import os
# Gradio 오프라인 모드 설정 (외부 CDN 의존성 제거)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import sys
from pathlib import Path
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import gradio as gr
import numpy as np
import hashlib
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.multimodal_parser import MultimodalParser
from src.embeddings.embedding_model import BGEM3Embedder
from src.search.vector_store import QdrantVectorStore
from src.search.bm25_search import BM25SearchEngine
from src.search.hybrid_search import HybridSearchEngine
from src.llm.qwen_model import QwenSummarizer, CachedSummarizer, LLMConfig
from src.recommend.recommender import FileRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryParser:
    """
    자연어 쿼리 파싱 - 시간 표현, 필터 조건 추출
    """

    # 시간 표현 패턴
    TIME_PATTERNS = {
        # 상대 시간 표현
        r'작년': ('year', -1),
        r'올해': ('year', 0),
        r'재작년': ('year', -2),
        r'내년': ('year', 1),
        r'지난\s*달': ('month', -1),
        r'이번\s*달': ('month', 0),
        r'저번\s*달': ('month', -1),
        r'다음\s*달': ('month', 1),
        r'지난\s*주': ('week', -1),
        r'이번\s*주': ('week', 0),
        r'저번\s*주': ('week', -1),
        r'다음\s*주': ('week', 1),
        r'어제': ('day', -1),
        r'오늘': ('day', 0),
        r'그제': ('day', -2),
        r'내일': ('day', 1),
        r'최근\s*(\d+)\s*일': ('recent_days', None),
        r'최근\s*(\d+)\s*주': ('recent_weeks', None),
        r'최근\s*(\d+)\s*개월': ('recent_months', None),
        r'(\d+)\s*일\s*전': ('days_ago', None),
        r'(\d+)\s*주\s*전': ('weeks_ago', None),
        r'(\d+)\s*개월\s*전': ('months_ago', None),
        # 절대 연도 (2020~2030)
        r'(20[2-3]\d)년': ('absolute_year', None),
        # 상반기/하반기
        r'(20[2-3]\d)년?\s*상반기': ('half_year_1', None),
        r'(20[2-3]\d)년?\s*하반기': ('half_year_2', None),
        r'상반기': ('current_half_1', None),
        r'하반기': ('current_half_2', None),
        # 분기
        r'(\d)분기': ('quarter', None),
    }

    # 파일 타입 패턴
    FILE_TYPE_PATTERNS = {
        r'pdf\s*(파일|문서)?': 'pdf',
        r'워드\s*(파일|문서)?': 'docx',
        r'docx?\s*(파일|문서)?': 'docx',
        r'엑셀\s*(파일|문서)?': 'xlsx',
        r'xlsx?\s*(파일|문서)?': 'xlsx',
        r'파워포인트\s*(파일|문서)?': 'pptx',
        r'pptx?\s*(파일|문서)?': 'pptx',
        r'ppt\s*(파일|문서)?': 'pptx',
        r'이미지\s*(파일)?': 'image',
        r'사진\s*(파일)?': 'image',
        r'(png|jpg|jpeg)\s*(파일)?': 'image',
    }

    # 부서 패턴
    DEPARTMENT_PATTERNS = {
        r'기획팀': '기획팀',
        r'개발팀': '개발팀',
        r'마케팅팀': '마케팅팀',
        r'영업팀': '영업팀',
        r'인사팀': '인사팀',
        r'재무팀': '재무팀',
        r'디자인팀': '디자인팀',
        r'품질관리팀': '품질관리팀',
        r'품질팀': '품질관리팀',
    }

    def __init__(self):
        self.now = datetime.now()

    def parse(self, query: str) -> Dict:
        """
        쿼리를 파싱하여 필터 조건과 정제된 쿼리 반환

        Args:
            query: 원본 쿼리

        Returns:
            {
                'cleaned_query': str,  # 필터 표현 제거된 쿼리
                'date_filter': {       # 날짜 필터
                    'start_date': datetime,
                    'end_date': datetime
                },
                'file_type': str,      # 파일 타입 필터
                'department': str      # 부서 필터
            }
        """
        result = {
            'cleaned_query': query,
            'date_filter': None,
            'file_type': None,
            'department': None
        }

        cleaned_query = query

        # 1. 시간 표현 파싱
        date_filter = self._parse_time_expression(query)
        if date_filter:
            result['date_filter'] = date_filter
            # 시간 표현 제거
            for pattern in self.TIME_PATTERNS.keys():
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

        # 2. 파일 타입 파싱
        for pattern, file_type in self.FILE_TYPE_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                result['file_type'] = file_type
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
                break

        # 3. 부서 파싱
        for pattern, department in self.DEPARTMENT_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                result['department'] = department
                # 부서명은 검색에 유용하므로 제거하지 않음
                break

        # 정제된 쿼리 (불필요한 공백 제거)
        result['cleaned_query'] = ' '.join(cleaned_query.split()).strip()

        return result

    def _parse_time_expression(self, query: str) -> Optional[Dict]:
        """시간 표현을 파싱하여 날짜 범위 반환"""
        for pattern, (time_type, offset) in self.TIME_PATTERNS.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return self._calculate_date_range(time_type, offset, match)
        return None

    def _calculate_date_range(self, time_type: str, offset: Optional[int], match) -> Dict:
        """시간 타입에 따른 날짜 범위 계산"""
        now = self.now

        if time_type == 'year':
            year = now.year + offset
            return {
                'start_date': datetime(year, 1, 1),
                'end_date': datetime(year, 12, 31, 23, 59, 59)
            }

        elif time_type == 'month':
            target = now + relativedelta(months=offset)
            start = datetime(target.year, target.month, 1)
            end = start + relativedelta(months=1) - timedelta(seconds=1)
            return {'start_date': start, 'end_date': end}

        elif time_type == 'week':
            # 주의 시작을 월요일로 가정
            days_since_monday = now.weekday()
            week_start = now - timedelta(days=days_since_monday) + timedelta(weeks=offset)
            week_start = datetime(week_start.year, week_start.month, week_start.day)
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return {'start_date': week_start, 'end_date': week_end}

        elif time_type == 'day':
            target = now + timedelta(days=offset)
            start = datetime(target.year, target.month, target.day)
            end = start + timedelta(hours=23, minutes=59, seconds=59)
            return {'start_date': start, 'end_date': end}

        elif time_type == 'recent_days':
            days = int(match.group(1))
            return {
                'start_date': now - timedelta(days=days),
                'end_date': now
            }

        elif time_type == 'recent_weeks':
            weeks = int(match.group(1))
            return {
                'start_date': now - timedelta(weeks=weeks),
                'end_date': now
            }

        elif time_type == 'recent_months':
            months = int(match.group(1))
            return {
                'start_date': now - relativedelta(months=months),
                'end_date': now
            }

        elif time_type == 'days_ago':
            days = int(match.group(1))
            target = now - timedelta(days=days)
            return {
                'start_date': datetime(target.year, target.month, target.day),
                'end_date': datetime(target.year, target.month, target.day, 23, 59, 59)
            }

        elif time_type == 'weeks_ago':
            weeks = int(match.group(1))
            target = now - timedelta(weeks=weeks)
            start = target - timedelta(days=target.weekday())
            return {
                'start_date': datetime(start.year, start.month, start.day),
                'end_date': datetime(start.year, start.month, start.day) + timedelta(days=6, hours=23, minutes=59, seconds=59)
            }

        elif time_type == 'months_ago':
            months = int(match.group(1))
            target = now - relativedelta(months=months)
            start = datetime(target.year, target.month, 1)
            end = start + relativedelta(months=1) - timedelta(seconds=1)
            return {'start_date': start, 'end_date': end}

        elif time_type == 'absolute_year':
            year = int(match.group(1))
            return {
                'start_date': datetime(year, 1, 1),
                'end_date': datetime(year, 12, 31, 23, 59, 59)
            }

        elif time_type == 'half_year_1':
            year = int(match.group(1))
            return {
                'start_date': datetime(year, 1, 1),
                'end_date': datetime(year, 6, 30, 23, 59, 59)
            }

        elif time_type == 'half_year_2':
            year = int(match.group(1))
            return {
                'start_date': datetime(year, 7, 1),
                'end_date': datetime(year, 12, 31, 23, 59, 59)
            }

        elif time_type == 'current_half_1':
            return {
                'start_date': datetime(now.year, 1, 1),
                'end_date': datetime(now.year, 6, 30, 23, 59, 59)
            }

        elif time_type == 'current_half_2':
            return {
                'start_date': datetime(now.year, 7, 1),
                'end_date': datetime(now.year, 12, 31, 23, 59, 59)
            }

        elif time_type == 'quarter':
            quarter = int(match.group(1))
            start_month = (quarter - 1) * 3 + 1
            return {
                'start_date': datetime(now.year, start_month, 1),
                'end_date': datetime(now.year, start_month, 1) + relativedelta(months=3) - timedelta(seconds=1)
            }

        return None


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

        # 5. LLM (프로토타입에서는 선택적으로 로드) + 캐싱 적용
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
                base_summarizer = QwenSummarizer(llm_config)
                # CachedSummarizer로 래핑하여 동일 문서 재요약 방지
                self.summarizer = CachedSummarizer(base_summarizer, cache_size=500)
                logger.info("LLM loaded with caching enabled (cache_size=500)")
            except Exception as e:
                logger.warning(f"LLM loading failed (optional): {e}")
        else:
            logger.info("LLM disabled via config. Skipping LLM load.")

        # 6. 추천 시스템
        self.recommender = FileRecommender(
            temporal_window_hours=self.config['recommendation']['temporal_window_hours']
        )

        # 7. 쿼리 파서 (시간 표현, 필터 추출)
        self.query_parser = QueryParser()

        logger.info("All components initialized successfully!")

    def search_files(
        self,
        query: str,
        top_k: int = 5,
        include_summary: bool = True,
        include_recommendations: bool = True,
        file_type_filter: Optional[str] = None,
        sort_by: str = 'relevance'  # 'relevance', 'date_desc', 'date_asc', 'name'
    ) -> Dict:
        """
        파일 검색 메인 함수

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            include_summary: 요약 포함 여부
            include_recommendations: 추천 포함 여부
            file_type_filter: 파일 타입 필터 (None, 'pdf', 'docx', 'pptx', 'xlsx', 'image')
            sort_by: 정렬 기준 ('relevance', 'date_desc', 'date_asc', 'name')

        Returns:
            검색 결과 딕셔너리
        """
        logger.info(f"Searching for: '{query}'")

        # 0. 쿼리 파싱 (시간 표현, 파일 타입 자동 추출)
        parsed = self.query_parser.parse(query)
        search_query = parsed['cleaned_query'] or query

        # UI에서 지정한 필터가 없으면 쿼리에서 추출한 필터 사용
        if not file_type_filter and parsed['file_type']:
            file_type_filter = parsed['file_type']

        # 적용된 필터 정보 (사용자에게 표시용)
        applied_filters = {
            'date_filter': parsed['date_filter'],
            'file_type': file_type_filter,
            'department': parsed['department']
        }

        logger.info(f"Parsed query: '{search_query}', Filters: {applied_filters}")

        # 1. 쿼리 임베딩 생성
        query_embedding = self.embedder.encode_queries(search_query)

        # 2. 하이브리드 검색 (필터 적용)
        filter_conditions = {}
        if file_type_filter:
            if file_type_filter == 'image':
                # 이미지는 여러 확장자 포함
                filter_conditions['file_type'] = ['png', 'jpg', 'jpeg']
            else:
                filter_conditions['file_type'] = file_type_filter

        if filter_conditions:
            raw_results = self.hybrid_engine.search_with_filter(
                query=search_query,
                query_embedding=query_embedding,
                filter_conditions=filter_conditions,
                top_k=self.config['search']['top_k']
            )
        else:
            raw_results = self.hybrid_engine.search(
                query=search_query,
                query_embedding=query_embedding,
                top_k=self.config['search']['top_k'],
                final_top_k=top_k * 2  # 필터링 후 줄어들 수 있으므로 여유있게
            )

        # 2-1. 파일 단위로 결과 집계 (청크 중복 제거)
        results = self._aggregate_results_by_file(raw_results)

        # 2-2. 날짜 필터 적용 (쿼리에서 추출된 시간 표현 기반)
        if parsed['date_filter']:
            results = self._apply_date_filter(results, parsed['date_filter'])

        # 2-3. 정렬 적용
        results = self._apply_sorting(results, sort_by)

        # top_k 제한
        results = results[:top_k]

        logger.info(f"Found {len(results)} results (after filters and sorting)")

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
            'total_found': len(results),
            'applied_filters': applied_filters,
            'sort_by': sort_by
        }

    def _apply_date_filter(self, results: List[Dict], date_filter: Dict) -> List[Dict]:
        """날짜 필터 적용"""
        if not date_filter:
            return results

        start_date = date_filter.get('start_date')
        end_date = date_filter.get('end_date')

        if not start_date or not end_date:
            return results

        filtered = []
        for result in results:
            meta = result.get('metadata', {})
            modified_time_str = meta.get('modified_time', '')

            if not modified_time_str:
                # 날짜 정보 없으면 포함 (필터링하지 않음)
                filtered.append(result)
                continue

            try:
                # ISO 형식 파싱
                modified_time = datetime.fromisoformat(modified_time_str.replace('Z', '+00:00'))
                # timezone 제거하여 비교
                modified_time = modified_time.replace(tzinfo=None)

                if start_date <= modified_time <= end_date:
                    filtered.append(result)
            except (ValueError, TypeError):
                # 파싱 실패 시 포함
                filtered.append(result)

        return filtered

    def _apply_sorting(self, results: List[Dict], sort_by: str) -> List[Dict]:
        """정렬 적용"""
        if not results:
            return results

        if sort_by == 'relevance':
            # 기본: 점수 내림차순 (이미 정렬되어 있음)
            return sorted(results, key=lambda r: r.get('score', 0), reverse=True)

        elif sort_by == 'date_desc':
            # 최신순
            def get_date(r):
                meta = r.get('metadata', {})
                time_str = meta.get('modified_time', '')
                try:
                    return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except:
                    return datetime.min
            return sorted(results, key=get_date, reverse=True)

        elif sort_by == 'date_asc':
            # 오래된순
            def get_date(r):
                meta = r.get('metadata', {})
                time_str = meta.get('modified_time', '')
                try:
                    return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                except:
                    return datetime.max
            return sorted(results, key=get_date, reverse=False)

        elif sort_by == 'name':
            # 파일명순
            def get_name(r):
                meta = r.get('metadata', {})
                return meta.get('file_name', '') or ''
            return sorted(results, key=get_name)

        return results

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
        show_explanation: bool,
        file_type_filter: str,
        sort_by: str
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
            file_type_filter: 파일 타입 필터
            sort_by: 정렬 기준

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
                    show_explanation,
                    file_type_filter,
                    sort_by
                )

            # 히스토리 업데이트 (Gradio 6.x 형식)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            error_response = self._format_user_friendly_error(e)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history

    def _format_user_friendly_error(self, error: Exception) -> str:
        """사용자 친화적인 에러 메시지 생성"""
        error_str = str(error).lower()

        if 'qdrant' in error_str or 'connection' in error_str:
            return "⚠️ **검색 서비스 연결 오류**\n\n검색 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n관리자에게 문의가 필요한 경우 IT 지원팀에 연락해주세요."

        elif 'index' in error_str or 'bm25' in error_str:
            return "⚠️ **검색 인덱스 준비 중**\n\n검색 인덱스가 아직 준비되지 않았습니다. 관리자가 인덱싱을 완료한 후 다시 시도해주세요."

        elif 'memory' in error_str or 'cuda' in error_str or 'gpu' in error_str:
            return "⚠️ **시스템 리소스 부족**\n\n현재 시스템 리소스가 부족합니다. 잠시 후 다시 시도하거나, 요약 기능을 비활성화하고 검색해보세요."

        elif 'timeout' in error_str:
            return "⚠️ **요청 시간 초과**\n\n검색 요청이 시간 초과되었습니다. 검색어를 더 구체적으로 입력하거나, 결과 수를 줄여서 다시 시도해주세요."

        elif 'file' in error_str and 'not found' in error_str:
            return "⚠️ **파일을 찾을 수 없음**\n\n요청한 파일을 찾을 수 없습니다. 파일이 이동되었거나 삭제되었을 수 있습니다."

        else:
            return f"⚠️ **오류가 발생했습니다**\n\n죄송합니다. 예상치 못한 오류가 발생했습니다.\n\n다시 시도해주시고, 문제가 지속되면 IT 지원팀에 문의해주세요.\n\n(오류 코드: {type(error).__name__})"

    def _handle_search(
        self,
        query: str,
        top_k: int,
        include_summary: bool,
        include_recommendations: bool,
        show_explanation: bool,
        file_type_filter: str,
        sort_by: str
    ) -> str:
        """검색 처리"""
        # '전체' 선택 시 None으로 변환
        actual_filter = None if file_type_filter == '전체' else file_type_filter

        search_result = self.pipeline.search_files(
            query=query,
            top_k=top_k,
            include_summary=include_summary,
            include_recommendations=include_recommendations,
            file_type_filter=actual_filter,
            sort_by=sort_by
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
        return """## 📖 사내 파일 검색 AI Agent 사용법

### 🔍 기본 검색
자연어로 검색어를 입력하세요. 시스템이 키워드와 의미를 모두 분석하여 관련 문서를 찾아줍니다.

### ⏰ 시간 표현 (자동 인식)
검색어에 시간 표현을 포함하면 자동으로 날짜 필터가 적용됩니다.
- **상대 시간**: "작년", "지난달", "이번 주", "어제", "최근 3개월"
- **절대 시간**: "2024년", "2023년 상반기", "1분기"
- **예시**: "작년 안전 점검 보고서" → 2025년 문서만 검색

### 📁 파일 타입 (자동 인식)
검색어에 파일 타입을 포함하면 자동으로 필터링됩니다.
- "PDF 파일", "워드 문서", "엑셀", "파워포인트", "이미지"
- **예시**: "마케팅팀 PDF 문서" → PDF 파일만 검색

### 🏢 부서명 (자동 인식)
부서명이 포함되면 해당 부서 관련 문서를 우선 검색합니다.
- 기획팀, 개발팀, 마케팅팀, 영업팀, 인사팀, 재무팀, 디자인팀, 품질관리팀

### ⚙️ 검색 설정 (오른쪽 패널)
- **파일 타입 필터**: 특정 파일 형식만 검색
- **정렬 기준**: 관련도순, 최신순, 오래된순, 파일명순
- **검색 설명 표시**: 왜 이 문서가 검색되었는지 근거 표시
- **연관 파일 추천**: 검색 결과와 유사한 파일 추천
- **요약 생성**: LLM이 문서 내용을 요약 (활성화 필요)

### 🔧 특수 명령
- **"중복 검사"**: 유사하거나 동일한 문서 그룹을 찾아줍니다
- **"추천" / "유사한 파일"**: 이전 검색 결과 기반 연관 파일 추천
- **"/help"**: 이 도움말 표시

### 💬 후속 질문 (LLM 활성화 시)
검색 후 결과에 대해 질문할 수 있습니다.
- "이 문서의 핵심 내용이 뭐야?"
- "ROI가 얼마라고 했어?"

### 💡 검색 팁
1. **구체적으로**: "보고서" 보다 "2024년 상반기 매출 보고서"
2. **시간 활용**: "작년 회의록", "최근 1주일 계획서"
3. **파일 타입 지정**: "엑셀로 된 예산 자료"
4. **부서 언급**: "마케팅팀 캠페인 분석"
"""

    def _highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """
        텍스트에서 키워드를 하이라이트 처리

        Args:
            text: 원본 텍스트
            keywords: 하이라이트할 키워드 리스트

        Returns:
            키워드가 **강조**된 텍스트
        """
        if not keywords or not text:
            return text

        highlighted = text
        for keyword in keywords:
            if len(keyword) < 2:  # 너무 짧은 키워드는 제외
                continue
            # 대소문자 무시하고 매칭, 원본 케이스 유지하면서 강조
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)

        return highlighted

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

            # 매칭 키워드 추출 (하이라이트용)
            matched_keywords = []
            if 'explanation' in result:
                matched_keywords = result['explanation'].get('matched_keywords', [])

            output += f"### {i+1}. {file_name}\n"
            output += f"- **경로:** `{file_path}`\n"
            output += f"- **타입:** {file_type}\n"
            output += f"- **통합 점수:** {score:.4f}\n"

            # 원문 바로가기 버튼 (파일 경로 링크)
            if file_path and file_path != 'N/A':
                # file:// 프로토콜로 로컬 파일 링크 생성
                file_link = f"file://{file_path}"
                output += f"- 📂 [원문 열기]({file_link})\n"

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

                # 매칭 키워드 (하이라이트된 형태로 표시)
                if matched_keywords:
                    highlighted_keywords = [f"**{kw}**" for kw in matched_keywords[:5]]
                    output += f"- 매칭 키워드: {', '.join(highlighted_keywords)}\n"

            # 요약
            if 'summary' in result and result['summary'] != "요약 미사용":
                output += f"\n**요약:** {result['summary']}\n"

            # 내용 미리보기 (키워드 하이라이트 적용)
            text_preview = result.get('text', '')[:300]
            if not text_preview:
                text_preview = meta.get('text', '')[:300]
            if text_preview:
                # 키워드 하이라이트 적용
                highlighted_preview = self._highlight_keywords(text_preview, matched_keywords)
                output += f"\n**미리보기:** {highlighted_preview}...\n"

            output += "\n---\n"

        # 추천 파일 (첫 번째 결과에 대해)
        if include_recommendations and search_result['results']:
            recommendations = search_result['results'][0].get('recommendations', [])
            if recommendations:
                output += "\n## 연관 파일 추천\n"
                for i, rec in enumerate(recommendations[:3]):
                    rec_path = rec.get('path', 'N/A')
                    rec_name = rec.get('file_name', 'Unknown')
                    rec_score = rec.get('recommendation_score', 0)
                    output += f"- **{rec_name}** (점수: {rec_score:.2f})"
                    if rec_path and rec_path != 'N/A':
                        output += f" - 📂 [열기](file://{rec_path})"
                    output += "\n"

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
            gr.Markdown("# 🔍 사내 네트워크 드라이브 파일 검색 AI Agent")
            gr.Markdown("자연어로 파일을 검색하고, 관련 문서를 추천받으세요. 대화형 인터페이스로 편리하게 사용할 수 있습니다.")

            # 상태 표시
            status_text = "✅ LLM 활성화 (요약/질문답변 가능)" if llm_available else "⚠️ LLM 비활성화 (요약/질문답변 불가)"
            status_color = "green" if llm_available else "orange"
            gr.Markdown(f"> **시스템 상태:** <span style='color:{status_color}'>{status_text}</span>")

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
                            placeholder="검색어를 입력하세요. 예: '작년 안전 점검 보고서', '마케팅팀 PDF 파일', '중복 검사'",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("🔍 검색", variant="primary", scale=1)

                # 설정 패널
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 검색 설정")

                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="검색 결과 수"
                    )

                    # 파일 타입 필터 (신규)
                    file_type_filter = gr.Dropdown(
                        choices=["전체", "pdf", "docx", "pptx", "xlsx", "image"],
                        value="전체",
                        label="📁 파일 타입 필터",
                        info="특정 파일 타입만 검색"
                    )

                    # 정렬 옵션 (신규)
                    sort_by = gr.Dropdown(
                        choices=[
                            ("관련도순", "relevance"),
                            ("최신순", "date_desc"),
                            ("오래된순", "date_asc"),
                            ("파일명순", "name")
                        ],
                        value="relevance",
                        label="📊 정렬 기준"
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 📋 표시 옵션")

                    show_explanation = gr.Checkbox(
                        label="검색 설명 표시 (매칭 근거)",
                        value=True
                    )

                    include_recommendations = gr.Checkbox(
                        label="연관 파일 추천",
                        value=True
                    )

                    summary_label = "📝 요약 생성" if llm_available else "📝 요약 생성 (비활성화)"
                    include_summary = gr.Checkbox(
                        label=summary_label,
                        value=False,
                        interactive=llm_available
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 🚀 빠른 명령")

                    help_btn = gr.Button("❓ 도움말", size="sm")
                    duplicate_btn = gr.Button("🔄 중복 문서 탐지", size="sm")
                    clear_btn = gr.Button("🗑️ 대화 초기화", size="sm", variant="secondary")

            # 예시 질의
            gr.Markdown("### 💡 예시 질의")
            gr.Markdown("시간 표현, 파일 타입, 부서명을 포함하면 자동으로 필터링됩니다.")
            gr.Examples(
                examples=[
                    ["작년 안전 점검 보고서"],
                    ["마케팅팀 PDF 문서"],
                    ["2024년 상반기 매출 실적"],
                    ["최근 3개월 회의록"],
                    ["고객 만족도 향상 전략"],
                    ["중복 검사"],
                    ["/help"]
                ],
                inputs=msg_input
            )

            # 이벤트 핸들러
            def submit_message(message, history, top_k, include_summary, include_recommendations, show_explanation, file_type_filter, sort_by):
                return self.chat_response(
                    message, history, top_k, include_summary, include_recommendations, show_explanation, file_type_filter, sort_by
                )

            # 전송 버튼 클릭
            send_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot, top_k_slider, include_summary, include_recommendations, show_explanation, file_type_filter, sort_by],
                outputs=[msg_input, chatbot]
            )

            # Enter 키로 전송
            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot, top_k_slider, include_summary, include_recommendations, show_explanation, file_type_filter, sort_by],
                outputs=[msg_input, chatbot]
            )

            # 도움말 버튼
            def show_help(history):
                help_msg = self._get_help_message()
                history.append({"role": "user", "content": "도움말"})
                history.append({"role": "assistant", "content": help_msg})
                return history

            help_btn.click(
                fn=show_help,
                inputs=[chatbot],
                outputs=[chatbot]
            )

            # 중복 탐지 버튼
            def run_duplicate(history):
                response = self._handle_duplicate_detection()
                history.append({"role": "user", "content": "중복 문서 탐지"})
                history.append({"role": "assistant", "content": response})
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
        share=True,  # Docker 컨테이너 외부 접속을 위해 공개 URL 생성
        quiet=False
    )


if __name__ == "__main__":
    main()
