"""
Qwen2.5-14B 모델을 활용한 문서 요약 및 LLM 추론
"""

import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available, falling back to transformers")

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM 설정"""
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    device: str = "cuda"
    max_model_len: int = 32768
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 500
    use_vllm: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85


class QwenSummarizer:
    """
    Qwen2.5-14B 기반 문서 요약 및 LLM 추론
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Args:
            config: LLM 설정
        """
        self.config = config or LLMConfig()

        # vLLM 사용 가능 여부 확인
        if self.config.use_vllm and not VLLM_AVAILABLE:
            logger.warning("vLLM not available, using transformers instead")
            self.config.use_vllm = False

        self.model = None
        self.tokenizer = None
        self.sampling_params = None

        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화"""
        logger.info(f"Loading model: {self.config.model_name}")

        if self.config.use_vllm:
            self._initialize_vllm()
        else:
            self._initialize_transformers()

        logger.info("Model loaded successfully")

    def _initialize_vllm(self):
        """vLLM 모델 초기화"""
        try:
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens
            )

            logger.info("vLLM model initialized")

        except Exception as e:
            logger.error(f"Error initializing vLLM: {e}")
            raise

    def _initialize_transformers(self):
        """Transformers 모델 초기화"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            self.model.eval()

            logger.info("Transformers model initialized")

        except Exception as e:
            logger.error(f"Error initializing Transformers: {e}")
            raise

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "bullet_points"
    ) -> str:
        """
        텍스트 요약

        Args:
            text: 요약할 텍스트
            max_length: 최대 요약 길이
            style: 요약 스타일 (bullet_points, paragraph, key_points)

        Returns:
            요약 텍스트
        """
        # 스타일별 프롬프트
        style_prompts = {
            "bullet_points": "다음 문서를 3-5개의 핵심 bullet point로 요약해주세요.",
            "paragraph": "다음 문서를 2-3개 문단으로 요약해주세요.",
            "key_points": "다음 문서의 핵심 내용만 간단히 요약해주세요.",
            "executive": "다음 문서를 경영진을 위한 요약본으로 작성해주세요."
        }

        instruction = style_prompts.get(style, style_prompts["bullet_points"])

        prompt = f"""{instruction}

문서 내용:
{text}

요약:"""

        return self._generate(prompt, max_length)

    def batch_summarize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        style: str = "bullet_points"
    ) -> List[str]:
        """
        배치 요약

        Args:
            texts: 요약할 텍스트 리스트
            max_length: 최대 요약 길이
            style: 요약 스타일

        Returns:
            요약 텍스트 리스트
        """
        prompts = []

        style_prompts = {
            "bullet_points": "다음 문서를 3-5개의 핵심 bullet point로 요약해주세요.",
            "paragraph": "다음 문서를 2-3개 문단으로 요약해주세요.",
            "key_points": "다음 문서의 핵심 내용만 간단히 요약해주세요."
        }

        instruction = style_prompts.get(style, style_prompts["bullet_points"])

        for text in texts:
            prompt = f"""{instruction}

문서 내용:
{text}

요약:"""
            prompts.append(prompt)

        return self._batch_generate(prompts, max_length)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        키워드 추출

        Args:
            text: 입력 텍스트
            top_k: 추출할 키워드 수

        Returns:
            키워드 리스트
        """
        prompt = f"""다음 문서에서 가장 중요한 키워드 {top_k}개를 추출해주세요.
쉼표로 구분하여 키워드만 나열해주세요.

문서:
{text}

키워드:"""

        result = self._generate(prompt, max_length=100)

        # 키워드 파싱
        keywords = [kw.strip() for kw in result.split(',')]
        return keywords[:top_k]

    def answer_question(self, context: str, question: str) -> str:
        """
        문서 기반 질의응답

        Args:
            context: 참조 문서
            question: 질문

        Returns:
            답변
        """
        prompt = f"""다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:"""

        return self._generate(prompt)

    def _generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_length: 최대 생성 길이

        Returns:
            생성된 텍스트
        """
        if self.config.use_vllm:
            return self._generate_vllm(prompt, max_length)
        else:
            return self._generate_transformers(prompt, max_length)

    def _generate_vllm(self, prompt: str, max_length: Optional[int] = None) -> str:
        """vLLM을 사용한 생성"""
        # 샘플링 파라미터 업데이트
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_length or self.config.max_tokens
        )

        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_transformers(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Transformers를 사용한 생성"""
        # 메시지 포맷 (Qwen 스타일)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length or self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True
            )

        # 입력 제거하고 생성된 부분만 추출
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def _batch_generate(self, prompts: List[str], max_length: Optional[int] = None) -> List[str]:
        """배치 생성"""
        if self.config.use_vllm:
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=max_length or self.config.max_tokens
            )

            outputs = self.model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        else:
            # Transformers는 배치 처리가 느리므로 순차 처리
            results = []
            for prompt in prompts:
                result = self._generate_transformers(prompt, max_length)
                results.append(result)
            return results


class CachedSummarizer:
    """캐싱 기능이 있는 요약기"""

    def __init__(self, summarizer: QwenSummarizer, cache_size: int = 1000):
        """
        Args:
            summarizer: 베이스 요약기
            cache_size: 캐시 크기
        """
        self.summarizer = summarizer
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size

    def summarize(self, text: str, **kwargs) -> str:
        """캐싱을 사용한 요약"""
        # 캐시 키 생성 (텍스트 해시)
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # 캐시 확인
        if cache_key in self.cache:
            logger.debug("Cache hit")
            return self.cache[cache_key]

        # 신규 요약 생성
        summary = self.summarizer.summarize(text, **kwargs)

        # 캐시 저장
        self.cache[cache_key] = summary

        # 캐시 크기 제한
        if len(self.cache) > self.cache_size:
            # 가장 오래된 항목 제거
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        return summary


def test_qwen_summarizer():
    """Qwen 요약기 테스트"""
    logger.info("Testing Qwen Summarizer")

    # 설정
    config = LLMConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        use_vllm=False,  # 테스트용 - Transformers 사용
        temperature=0.1,
        max_tokens=300
    )

    # 요약기 초기화
    summarizer = QwenSummarizer(config)

    # 테스트 문서
    test_document = """
2024년 상반기 우리 회사의 매출은 전년 동기 대비 25% 증가한 500억원을 기록했습니다.
이는 신제품 출시와 마케팅 강화에 따른 결과입니다.
특히 AI 기반 자동화 솔루션이 큰 인기를 끌면서 주요 수익원으로 자리잡았습니다.

고객 만족도 조사 결과, 전체 만족도는 4.5/5.0으로 나타났으며,
특히 고객 지원 서비스 부문에서 높은 평가를 받았습니다.

하반기에는 글로벌 시장 진출을 계획하고 있으며,
북미와 유럽 지역을 1차 타겟으로 설정했습니다.
예상 투자 금액은 약 100억원이며, 현지 파트너사와의 협력을 통해
안정적인 시장 진입을 도모할 예정입니다.
"""

    # 1. 요약 테스트
    logger.info("\n1. Document summarization (bullet points)")
    summary = summarizer.summarize(test_document, style="bullet_points")
    logger.info(f"Summary:\n{summary}")

    # 2. 키워드 추출
    logger.info("\n2. Keyword extraction")
    keywords = summarizer.extract_keywords(test_document, top_k=5)
    logger.info(f"Keywords: {keywords}")

    # 3. 질의응답
    logger.info("\n3. Question answering")
    question = "상반기 매출은 얼마인가요?"
    answer = summarizer.answer_question(test_document, question)
    logger.info(f"Q: {question}")
    logger.info(f"A: {answer}")


if __name__ == "__main__":
    test_qwen_summarizer()
