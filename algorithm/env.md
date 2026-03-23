# Environment Setup & Compatibility Fixes

## 목표 환경

| 항목 | 버전 |
|------|------|
| Python | 3.11 |
| torch | 2.8.0+ |
| CUDA | 12.8 |
| transformers | 4.57.0 |
| accelerate | 1.0.0+ |
| bitsandbytes | 0.45.0+ |

---

## uv 환경 설정

### 권장: uv managed Python 사용

conda Python은 시스템 라이브러리(`libstdc++`, `libicu`) 버전 충돌을 유발할 수 있다.
uv managed Python은 libstdc++를 정적 링크하므로 이 문제가 없다.

```bash
# uv managed Python 3.11 설치
uv python install 3.11

# venv 생성 (conda Python이 아닌 uv managed Python 사용)
cd algorithm/focus
uv venv --python 3.11

# Python 버전 고정
echo "3.11" > .python-version

# 패키지 설치
uv pip install -e .
```

### 가상환경 활성화

```bash
source algorithm/focus/.venv/bin/activate
```

---

## 수정된 파일 목록

### 1. `algorithm/focus/pyproject.toml`

의존성 버전을 torch 2.8.0 + CUDA 12.8 환경에 맞게 업데이트.

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| torch | `>=2.1.0` | `>=2.8.0` |
| transformers | `>=4.48.2,<4.50.0` | `==4.57.0` |
| accelerate | `>=0.29.1` | `>=1.0.0` |
| bitsandbytes | `==0.48.1` | `>=0.45.0` |
| qwen-vl-utils | `==0.0.10` | `>=0.0.10` |

---

### 2. transformers 4.57 API 변경 대응

transformers 4.57에서 아래 심볼들이 제거되었다.

| 심볼 | 기존 위치 | 해결 방법 |
|------|----------|----------|
| `QWEN2_INPUTS_DOCSTRING` | `transformers.models.qwen2.modeling_qwen2` | 임포트 제거, 데코레이터에 `""` 사용 |
| `logger` | `transformers.models.qwen2.modeling_qwen2` | `transformers.utils.logging`에서 직접 생성 |
| `Qwen2SdpaAttention` | `transformers.models.qwen2.modeling_qwen2` | `Qwen2Attention as Qwen2SdpaAttention`으로 대체 |
| `apply_chunking_to_forward` | `transformers.modeling_utils` | `transformers.pytorch_utils`로 이동 |
| `find_pruneable_heads_and_indices` | `transformers.modeling_utils` | `transformers.pytorch_utils`로 이동 |
| `prune_linear_layer` | `transformers.modeling_utils` | `transformers.pytorch_utils`로 이동 |

**수정된 파일:**

```
algorithm/focus/models/qwen2/modeling_qwen2.py
algorithm/focus/models/qwen2/modeling_qwen2_CMC.py
algorithm/focus/models/qwen2/modeling_qwen2_adaptiv.py
algorithm/focus/models/qwen2_5_vl/modeling_qwen2_5_vl.py
algorithm/focus/models/qwen2_5_vl/modeling_qwen2_5_vl_adaptiv.py
algorithm/lmms-eval/lmms_eval/framefusion/models/qwen2/modeling_qwen2.py
algorithm/lmms-eval/lmms_eval/framefusion/models/qwen2/modeling_qwen2_baseline.py
```

**패턴 (logger + QWEN2_INPUTS_DOCSTRING):**

```python
# Before
from transformers.models.qwen2.modeling_qwen2 import (
    QWEN2_INPUTS_DOCSTRING,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)
from transformers.utils.doc import add_start_docstrings_to_model_forward

@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def SomeModel_forward(...):

# After
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging as transformers_logging
from transformers.utils.doc import add_start_docstrings_to_model_forward

logger = transformers_logging.get_logger(__name__)

@add_start_docstrings_to_model_forward("")
def SomeModel_forward(...):
```

---

### 3. `3rd_party/LLaVA-NeXT/llava/model/multimodal_resampler/qformer.py`

pruning 유틸리티가 `transformers.modeling_utils` → `transformers.pytorch_utils`로 이동.

```python
# Before
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

# After
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

---

## 시스템 라이브러리 충돌 (conda Python 사용 시)

### 증상

```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6:
version `CXXABI_1.3.15' not found
(required by /opt/conda/lib/python3.11/lib-dynload/../.././libicui18n.so.78)
```

### 원인

conda Python이 의존하는 `libicui18n.so.78`이 `CXXABI_1.3.15`를 요구하는데,
시스템(`/usr/lib`)과 conda(`/opt/conda/lib`) 모두 최대 `CXXABI_1.3.14`만 지원한다.

### 해결책 A: uv managed Python으로 venv 재생성 (권장)

위의 [uv 환경 설정](#uv-환경-설정) 참고.

### 해결책 B: LD_PRELOAD (conda Python 유지 시 임시방편)

NVIDIA nsight-compute에 번들된 libstdc++ (`CXXABI_1.3.15` 포함)를 preload한다.
`run_focus_image.sh` 상단에 추가되어 있음:

```bash
export LD_PRELOAD=/opt/nvidia/nsight-compute/2025.3.0/host/linux-desktop-glibc_2_11_3-x64/libstdc++.so.6
```

> **주의**: nsight-compute 버전이 바뀌면 경로를 업데이트해야 한다.
> 근본적인 해결은 uv managed Python 사용이다.
