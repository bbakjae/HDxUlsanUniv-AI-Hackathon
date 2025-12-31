# LLM 요약 기능 활성화 안내

## 현재 상태
LLM(Large Language Model)은 기본적으로 **비활성화** 상태입니다. 이는 GPU 메모리 사용량이 높기 때문입니다.

## 활성화 방법

### 1. config.yaml 수정
`config/config.yaml` 파일에서 LLM 설정을 변경합니다:

```yaml
# LLM 설정
llm:
  enabled: true  # false → true로 변경
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  device: "cuda"
  ...
```

### 2. 시스템 요구사항
- **GPU 메모리**: 최소 24GB 이상 권장 (RTX 4090, A100 등)
- **모델 크기**: Qwen2.5-14B는 약 28GB 메모리 필요
- 메모리 부족 시 `gpu_memory_utilization` 값을 낮추거나 더 작은 모델 사용

### 3. 대안: 작은 모델 사용
GPU 메모리가 부족한 경우 더 작은 모델을 사용할 수 있습니다:

```yaml
llm:
  enabled: true
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # 7B 버전
  # 또는
  model_name: "Qwen/Qwen2.5-3B-Instruct"  # 3B 버전
```

## 활성화 후 기능
LLM이 활성화되면 다음 기능을 사용할 수 있습니다:

1. **문서 요약**: 검색된 문서의 핵심 내용을 요약
2. **질문 답변**: 검색 결과를 바탕으로 질문에 답변
3. **키워드 추출**: 문서에서 중요 키워드 자동 추출

## 확인 방법
Gradio UI 상단에서 상태를 확인할 수 있습니다:
- `LLM 활성화` (녹색): LLM 기능 사용 가능
- `LLM 비활성화` (주황색): LLM 기능 사용 불가
