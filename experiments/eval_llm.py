"""
LLM 모델 비교 평가 스크립트
- ROUGE 점수, 추론 시간, GPU 메모리 사용량 측정
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""
    name: str
    model_id: str
    params: str  # 파라미터 크기 (예: "7B", "14B")
    device: str = "cuda"
    temperature: float = 0.7
    max_tokens: int = 512
    use_fp16: bool = True


@dataclass
class LLMEvalResult:
    """LLM 모델 평가 결과"""
    model_name: str
    model_id: str
    params: str
    rouge_1: float
    rouge_2: float
    rouge_l: float
    korean_rouge_l: float
    english_rouge_l: float
    avg_inference_time_sec: float
    tokens_per_second: float
    gpu_memory_gb: float
    total_samples: int
    eval_timestamp: str


# 평가 대상 LLM 모델 목록
LLM_MODELS = [
    LLMModelConfig(
        name="Qwen2.5-14B-Instruct",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        params="14B"
    ),
    LLMModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        params="7B"
    ),
    LLMModelConfig(
        name="Qwen2.5-3B-Instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        params="3B"
    ),
    LLMModelConfig(
        name="Phi-3.5-mini-instruct",
        model_id="microsoft/Phi-3.5-mini-instruct",
        params="3.8B"
    ),
    LLMModelConfig(
        name="gemma-2-9b-it",
        model_id="google/gemma-2-9b-it",
        params="9B"
    ),
    LLMModelConfig(
        name="Mistral-7B-Instruct-v0.3",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        params="7B"
    ),
]


class LLMModelWrapper:
    """LLM 모델 래퍼"""

    def __init__(self, config: LLMModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """모델 로딩"""
        logger.info(f"Loading LLM model: {self.config.name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )

        # 모델 로드
        dtype = torch.float16 if self.config.use_fp16 else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        logger.info(f"Model loaded: {self.config.name}")

    def generate_summary(self, text: str, style: str = "bullet_points") -> Tuple[str, float, int]:
        """
        요약 생성

        Returns:
            (요약 텍스트, 추론 시간(초), 생성된 토큰 수)
        """
        # 프롬프트 구성
        if style == "bullet_points":
            prompt = f"""다음 문서의 핵심 내용을 3-5개의 불릿 포인트로 요약해주세요.

문서:
{text[:2000]}

요약:"""
        else:
            prompt = f"""다음 문서를 간결하게 요약해주세요.

문서:
{text[:2000]}

요약:"""

        # 모델별 프롬프트 포맷 조정
        if "Qwen" in self.config.model_id:
            messages = [
                {"role": "system", "content": "당신은 문서 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "Phi" in self.config.model_id:
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "gemma" in self.config.model_id.lower():
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "Mistral" in self.config.model_id:
            formatted_prompt = f"[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt

        # 토큰화
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        input_length = inputs['input_ids'].shape[1]

        # 생성
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        inference_time = time.time() - start_time

        # 디코딩
        generated_tokens = outputs[0][input_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return summary.strip(), inference_time, len(generated_tokens)

    def get_gpu_memory_usage(self) -> float:
        """현재 GPU 메모리 사용량 (GB)"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0

    def cleanup(self):
        """모델 메모리 해제"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class ROUGEScorer:
    """ROUGE 점수 계산기"""

    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=False  # 한국어는 stemmer 미사용
            )
            self.available = True
        except ImportError:
            logger.warning("rouge_score not available. Installing...")
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'rouge-score'], check=True)
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=False
            )
            self.available = True

    def compute_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        if not reference or not hypothesis:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        scores = self.scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }


class LLMEvaluator:
    """LLM 모델 평가기"""

    def __init__(self, test_queries_path: str, qdrant_storage_path: str):
        self.test_queries_path = test_queries_path
        self.qdrant_storage_path = qdrant_storage_path
        self.test_data = self._load_test_queries()
        self.rouge_scorer = ROUGEScorer()
        self.sample_documents = self._load_sample_documents()

    def _load_test_queries(self) -> Dict:
        """테스트 쿼리 로드"""
        with open(self.test_queries_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_sample_documents(self) -> List[Dict]:
        """샘플 문서 로드 (요약 평가용)"""
        from qdrant_client import QdrantClient

        try:
            client = QdrantClient(path=self.qdrant_storage_path)
            collection_name = "company_files"

            # 다양한 문서 샘플링 (10개)
            result = client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            documents = []
            for point in result[0]:
                text = point.payload.get('text', '')
                if text and len(text) > 200:  # 최소 길이 필터
                    documents.append({
                        'id': str(point.id),
                        'text': text,
                        'file_name': point.payload.get('file_name', ''),
                        'language': 'ko' if any('\uac00' <= c <= '\ud7a3' for c in text[:100]) else 'en'
                    })

            logger.info(f"Loaded {len(documents)} sample documents for summarization")
            return documents

        except Exception as e:
            logger.error(f"Error loading sample documents: {e}")
            return []

    def _create_reference_summary(self, text: str) -> str:
        """
        참조 요약 생성 (간단한 추출 기반)
        실제 실험에서는 수동 레이블링된 참조 요약 사용 권장
        """
        # 문장 분리 (간단한 방법)
        sentences = []
        for sep in ['. ', '.\n', '다. ', '요. ', '음. ']:
            if sep in text:
                sentences = text.split(sep)
                break

        if not sentences:
            sentences = [text]

        # 상위 3개 문장 추출 (길이 기준)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        sentences.sort(key=len, reverse=True)

        reference = ' '.join(sentences[:3])
        return reference[:500]  # 최대 500자

    def evaluate_model(self, model_config: LLMModelConfig, num_samples: int = 10) -> LLMEvalResult:
        """단일 모델 평가"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_config.name}")
        logger.info(f"{'='*60}")

        # GPU 메모리 초기화
        torch.cuda.reset_peak_memory_stats()

        # 모델 로드
        model = LLMModelWrapper(model_config)

        # GPU 메모리 측정 (모델 로딩 후)
        gpu_memory = model.get_gpu_memory_usage()

        # 평가 데이터 준비
        samples = self.sample_documents[:num_samples]
        if not samples:
            logger.error("No sample documents available for evaluation")
            model.cleanup()
            return None

        # 평가 수행
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        korean_rouge_l = []
        english_rouge_l = []
        inference_times = []
        total_tokens = 0

        for sample in tqdm(samples, desc="Generating summaries"):
            try:
                # 참조 요약 (실제 실험에서는 수동 레이블 사용)
                reference = self._create_reference_summary(sample['text'])

                # 요약 생성
                summary, inf_time, num_tokens = model.generate_summary(sample['text'])

                inference_times.append(inf_time)
                total_tokens += num_tokens

                # ROUGE 점수 계산
                scores = self.rouge_scorer.compute_scores(reference, summary)

                rouge_scores['rouge1'].append(scores['rouge1'])
                rouge_scores['rouge2'].append(scores['rouge2'])
                rouge_scores['rougeL'].append(scores['rougeL'])

                # 언어별 분류
                if sample.get('language', 'ko') == 'ko':
                    korean_rouge_l.append(scores['rougeL'])
                else:
                    english_rouge_l.append(scores['rougeL'])

                logger.debug(f"Sample {sample['id']}: ROUGE-L={scores['rougeL']:.4f}, Time={inf_time:.2f}s")

            except Exception as e:
                logger.error(f"Error processing sample {sample['id']}: {e}")
                continue

        # 결과 집계
        total_time = sum(inference_times)
        avg_time = np.mean(inference_times) if inference_times else 0
        tps = total_tokens / total_time if total_time > 0 else 0

        result = LLMEvalResult(
            model_name=model_config.name,
            model_id=model_config.model_id,
            params=model_config.params,
            rouge_1=np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            rouge_2=np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            rouge_l=np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            korean_rouge_l=np.mean(korean_rouge_l) if korean_rouge_l else 0,
            english_rouge_l=np.mean(english_rouge_l) if english_rouge_l else 0,
            avg_inference_time_sec=avg_time,
            tokens_per_second=tps,
            gpu_memory_gb=gpu_memory,
            total_samples=len(samples),
            eval_timestamp=datetime.now().isoformat()
        )

        # 정리
        model.cleanup()

        return result

    def evaluate_all_models(self, output_path: str, num_samples: int = 10) -> List[LLMEvalResult]:
        """모든 모델 평가"""
        results = []

        for config in LLM_MODELS:
            try:
                result = self.evaluate_model(config, num_samples)
                if result:
                    results.append(result)

                    # 중간 결과 저장
                    self._save_results(results, output_path)

                    logger.info(f"\nResults for {config.name}:")
                    logger.info(f"  ROUGE-1: {result.rouge_1:.4f}")
                    logger.info(f"  ROUGE-2: {result.rouge_2:.4f}")
                    logger.info(f"  ROUGE-L: {result.rouge_l:.4f}")
                    logger.info(f"  Avg Inference Time: {result.avg_inference_time_sec:.2f}s")
                    logger.info(f"  Tokens/sec: {result.tokens_per_second:.2f}")
                    logger.info(f"  GPU Memory: {result.gpu_memory_gb:.2f}GB")

            except Exception as e:
                logger.error(f"Error evaluating {config.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def _save_results(self, results: List[LLMEvalResult], output_path: str):
        """결과 저장"""
        results_dict = [asdict(r) for r in results]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

    def compute_weighted_score(self, result: LLMEvalResult, all_results: List[LLMEvalResult]) -> float:
        """가중 종합 점수 계산"""
        # 정규화를 위한 최소/최대값 계산
        times = [r.avg_inference_time_sec for r in all_results]
        memories = [r.gpu_memory_gb for r in all_results]

        min_time, max_time = min(times), max(times)
        min_mem, max_mem = min(memories), max(memories)

        # 정규화 (시간/메모리는 낮을수록 좋음)
        norm_time = 1 - (result.avg_inference_time_sec - min_time) / (max_time - min_time + 1e-6)
        norm_mem = 1 - (result.gpu_memory_gb - min_mem) / (max_mem - min_mem + 1e-6)

        # 가중 점수 계산
        score = (
            result.rouge_l * 0.25 +
            norm_time * 0.30 +
            result.rouge_1 * 0.15 +
            result.rouge_2 * 0.10 +
            norm_mem * 0.10 +
            result.korean_rouge_l * 0.10
        )

        return score

    def print_comparison_table(self, results: List[LLMEvalResult]):
        """비교표 출력"""
        # 종합 점수 계산
        scores = [(r, self.compute_weighted_score(r, results)) for r in results]
        scores.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 140)
        print("LLM 모델 비교 결과표")
        print("=" * 140)
        print(f"{'모델':<30} {'파라미터':>8} {'ROUGE-L':>10} {'ROUGE-1':>10} {'ROUGE-2':>10} {'추론시간(s)':>12} {'TPS':>10} {'메모리(GB)':>10} {'종합점수':>10}")
        print("-" * 140)

        for result, score in scores:
            print(f"{result.model_name:<30} {result.params:>8} {result.rouge_l:>10.4f} {result.rouge_1:>10.4f} {result.rouge_2:>10.4f} {result.avg_inference_time_sec:>12.2f} {result.tokens_per_second:>10.2f} {result.gpu_memory_gb:>10.2f} {score:>10.4f}")

        print("=" * 140)
        print(f"\n최고 성능 모델: {scores[0][0].model_name} (종합점수: {scores[0][1]:.4f})")

        # 추천 요약
        print("\n" + "-" * 60)
        print("추천:")
        print(f"  - 품질 우선: {max(results, key=lambda r: r.rouge_l).model_name}")
        print(f"  - 속도 우선: {min(results, key=lambda r: r.avg_inference_time_sec).model_name}")
        print(f"  - 균형 (종합점수): {scores[0][0].model_name}")


def main():
    parser = argparse.ArgumentParser(description='LLM 모델 비교 평가')
    parser.add_argument('--test-queries', type=str,
                        default='experiments/test_queries.json',
                        help='테스트 쿼리 JSON 파일 경로')
    parser.add_argument('--qdrant-storage', type=str,
                        default='qdrant_storage_gdrive',
                        help='Qdrant 스토리지 경로')
    parser.add_argument('--output', type=str,
                        default='experiments/results/llm_results.json',
                        help='결과 저장 경로')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='평가할 샘플 문서 수')
    parser.add_argument('--models', type=str, nargs='+',
                        help='평가할 모델 이름 목록 (기본: 전체)')

    args = parser.parse_args()

    # 경로 설정
    project_root = Path(__file__).parent.parent
    test_queries_path = project_root / args.test_queries
    qdrant_storage_path = project_root / args.qdrant_storage
    output_path = project_root / args.output

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 평가 실행
    evaluator = LLMEvaluator(
        str(test_queries_path),
        str(qdrant_storage_path)
    )

    # 특정 모델만 평가할 경우 필터링
    if args.models:
        global LLM_MODELS
        LLM_MODELS = [m for m in LLM_MODELS if m.name in args.models]

    results = evaluator.evaluate_all_models(str(output_path), args.num_samples)

    # 비교표 출력
    if results:
        evaluator.print_comparison_table(results)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
