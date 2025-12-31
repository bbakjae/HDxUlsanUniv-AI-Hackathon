"""
파일 추천 시스템
- 벡터 유사도 기반 추천
- 메타데이터 기반 추천 (시간, 경로, 타입)
- 하이브리드 스코어링
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileRecommender:
    """파일 추천 시스템"""

    def __init__(
        self,
        vector_weight: float = 0.5,
        temporal_weight: float = 0.2,
        path_weight: float = 0.15,
        type_weight: float = 0.15,
        temporal_window_hours: int = 24
    ):
        """
        Args:
            vector_weight: 벡터 유사도 가중치
            temporal_weight: 시간적 연관성 가중치
            path_weight: 경로 유사도 가중치
            type_weight: 파일 타입 일치 가중치
            temporal_window_hours: 시간 윈도우 (시간)
        """
        self.vector_weight = vector_weight
        self.temporal_weight = temporal_weight
        self.path_weight = path_weight
        self.type_weight = type_weight
        self.temporal_window_hours = temporal_window_hours

        # 가중치 정규화
        total_weight = sum([vector_weight, temporal_weight, path_weight, type_weight])
        self.vector_weight /= total_weight
        self.temporal_weight /= total_weight
        self.path_weight /= total_weight
        self.type_weight /= total_weight

    def recommend_similar_files(
        self,
        target_file: Dict,
        candidate_files: List[Dict],
        target_embedding: Optional[np.ndarray] = None,
        candidate_embeddings: Optional[np.ndarray] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        유사 파일 추천

        Args:
            target_file: 타겟 파일 정보
                {
                    'id': str,
                    'path': str,
                    'file_type': str,
                    'modified_time': str (ISO format),
                    'metadata': dict
                }
            candidate_files: 후보 파일 리스트
            target_embedding: 타겟 파일 임베딩
            candidate_embeddings: 후보 파일 임베딩 (n, dim)
            top_k: 추천할 파일 수

        Returns:
            추천 파일 리스트 (점수 포함)
        """
        if not candidate_files:
            return []

        scores = []

        for i, candidate in enumerate(candidate_files):
            # 자기 자신 제외
            if candidate['id'] == target_file['id']:
                scores.append(0.0)
                continue

            total_score = 0.0

            # 1. 벡터 유사도
            if target_embedding is not None and candidate_embeddings is not None:
                vector_sim = self._compute_vector_similarity(
                    target_embedding,
                    candidate_embeddings[i]
                )
                total_score += self.vector_weight * vector_sim

            # 2. 시간적 연관성
            temporal_sim = self._compute_temporal_similarity(
                target_file.get('modified_time'),
                candidate.get('modified_time')
            )
            total_score += self.temporal_weight * temporal_sim

            # 3. 경로 유사도
            path_sim = self._compute_path_similarity(
                target_file.get('path', ''),
                candidate.get('path', '')
            )
            total_score += self.path_weight * path_sim

            # 4. 파일 타입 일치
            type_sim = 1.0 if target_file.get('file_type') == candidate.get('file_type') else 0.0
            total_score += self.type_weight * type_sim

            scores.append(total_score)

        # 상위 k개 선택
        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:  # 점수가 0보다 큰 경우만
                rec = {
                    **candidate_files[idx],
                    'recommendation_score': scores[idx],
                    'similarity_breakdown': {
                        'vector': self._compute_vector_similarity(
                            target_embedding, candidate_embeddings[idx]
                        ) if target_embedding is not None else 0.0,
                        'temporal': self._compute_temporal_similarity(
                            target_file.get('modified_time'),
                            candidate_files[idx].get('modified_time')
                        ),
                        'path': self._compute_path_similarity(
                            target_file.get('path', ''),
                            candidate_files[idx].get('path', '')
                        ),
                        'type': 1.0 if target_file.get('file_type') == candidate_files[idx].get('file_type') else 0.0
                    }
                }
                recommendations.append(rec)

        return recommendations

    def _compute_vector_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        코사인 유사도 계산

        Args:
            vec1: 벡터 1
            vec2: 벡터 2

        Returns:
            유사도 (0-1)
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # 코사인 유사도
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # 0-1 범위로 정규화
        return (similarity + 1) / 2

    def _compute_temporal_similarity(
        self,
        time1: Optional[str],
        time2: Optional[str]
    ) -> float:
        """
        시간적 유사도 계산

        Args:
            time1: 시간 1 (ISO format string)
            time2: 시간 2 (ISO format string)

        Returns:
            유사도 (0-1)
        """
        if not time1 or not time2:
            return 0.0

        try:
            dt1 = datetime.fromisoformat(time1)
            dt2 = datetime.fromisoformat(time2)

            # 시간 차이 (시간 단위)
            time_diff = abs((dt1 - dt2).total_seconds()) / 3600

            # 시간 윈도우 내에 있으면 높은 점수
            if time_diff <= self.temporal_window_hours:
                # 가까울수록 높은 점수 (지수 감소)
                similarity = np.exp(-time_diff / (self.temporal_window_hours / 2))
                return similarity
            else:
                # 윈도우 밖이면 낮은 점수
                return np.exp(-time_diff / (self.temporal_window_hours * 10))

        except Exception as e:
            logger.debug(f"Error computing temporal similarity: {e}")
            return 0.0

    def _compute_path_similarity(self, path1: str, path2: str) -> float:
        """
        경로 유사도 계산

        Args:
            path1: 경로 1
            path2: 경로 2

        Returns:
            유사도 (0-1)
        """
        if not path1 or not path2:
            return 0.0

        # Path 객체로 변환
        p1 = Path(path1)
        p2 = Path(path2)

        # 공통 부모 경로 찾기
        try:
            # 상대 경로 계산
            common_parts = 0
            p1_parts = p1.parts
            p2_parts = p2.parts

            for part1, part2 in zip(p1_parts, p2_parts):
                if part1 == part2:
                    common_parts += 1
                else:
                    break

            # 같은 폴더에 있으면 1.0
            if p1.parent == p2.parent:
                return 1.0

            # 공통 경로 비율 계산
            max_depth = max(len(p1_parts), len(p2_parts))
            if max_depth == 0:
                return 0.0

            similarity = common_parts / max_depth
            return similarity

        except Exception as e:
            logger.debug(f"Error computing path similarity: {e}")
            return 0.0

    def recommend_by_query_results(
        self,
        search_results: List[Dict],
        all_files: List[Dict],
        all_embeddings: Optional[np.ndarray] = None,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        검색 결과 기반 추천
        각 검색 결과에 대한 연관 파일 추천

        Args:
            search_results: 검색 결과 리스트
            all_files: 전체 파일 리스트
            all_embeddings: 전체 파일 임베딩
            top_k: 각 파일당 추천 수

        Returns:
            {file_id: [recommended_files]}
        """
        recommendations = {}

        for result in search_results:
            file_id = result['id']

            # 해당 파일의 임베딩 찾기
            target_embedding = None
            target_idx = None

            for i, file in enumerate(all_files):
                if file['id'] == file_id:
                    target_idx = i
                    if all_embeddings is not None:
                        target_embedding = all_embeddings[i]
                    break

            if target_idx is None:
                continue

            # 추천 생성
            target_file = all_files[target_idx]

            recommended = self.recommend_similar_files(
                target_file=target_file,
                candidate_files=all_files,
                target_embedding=target_embedding,
                candidate_embeddings=all_embeddings,
                top_k=top_k
            )

            recommendations[file_id] = recommended

        return recommendations


def test_recommender():
    """추천 시스템 테스트"""
    logger.info("Testing File Recommender")

    # 추천 시스템 초기화
    recommender = FileRecommender(
        vector_weight=0.5,
        temporal_weight=0.2,
        path_weight=0.15,
        type_weight=0.15,
        temporal_window_hours=24
    )

    # 테스트 데이터
    now = datetime.now()

    target_file = {
        'id': 'file_1',
        'path': '/project/reports/2024_sales.pdf',
        'file_type': 'pdf',
        'modified_time': now.isoformat(),
        'metadata': {'department': '재무팀'}
    }

    candidate_files = [
        {
            'id': 'file_2',
            'path': '/project/reports/2024_marketing.pdf',
            'file_type': 'pdf',
            'modified_time': (now - timedelta(hours=2)).isoformat(),
            'metadata': {'department': '마케팅팀'}
        },
        {
            'id': 'file_3',
            'path': '/project/reports/2024_sales_analysis.docx',
            'file_type': 'docx',
            'modified_time': (now - timedelta(hours=1)).isoformat(),
            'metadata': {'department': '재무팀'}
        },
        {
            'id': 'file_4',
            'path': '/project/archive/old_report.pdf',
            'file_type': 'pdf',
            'modified_time': (now - timedelta(days=30)).isoformat(),
            'metadata': {'department': '재무팀'}
        },
        {
            'id': 'file_5',
            'path': '/project/reports/2024_budget.xlsx',
            'file_type': 'xlsx',
            'modified_time': now.isoformat(),
            'metadata': {'department': '재무팀'}
        }
    ]

    # 임베딩 (랜덤 생성)
    target_embedding = np.random.rand(1024).astype(np.float32)
    candidate_embeddings = np.random.rand(len(candidate_files), 1024).astype(np.float32)

    # 첫 번째 후보를 타겟과 유사하게 설정
    candidate_embeddings[0] = target_embedding + np.random.randn(1024) * 0.1

    # 추천 생성
    logger.info(f"\nTarget file: {target_file['path']}")

    recommendations = recommender.recommend_similar_files(
        target_file=target_file,
        candidate_files=candidate_files,
        target_embedding=target_embedding,
        candidate_embeddings=candidate_embeddings,
        top_k=3
    )

    logger.info(f"\nTop {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations):
        logger.info(f"\n{i+1}. {rec['path']}")
        logger.info(f"   Overall score: {rec['recommendation_score']:.4f}")
        logger.info(f"   Breakdown:")
        for key, value in rec['similarity_breakdown'].items():
            logger.info(f"     - {key}: {value:.4f}")


if __name__ == "__main__":
    test_recommender()
