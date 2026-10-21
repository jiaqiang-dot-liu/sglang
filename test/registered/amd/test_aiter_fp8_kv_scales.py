"""GSM8K with an fp8 KV cache and a checkpoint that ships calibrated KV scales.

``amd/Llama-3.1-8B-Instruct-FP8-KV`` carries a per-layer ``kv_scale`` far from 1, so
the cache holds ``K / k_scale`` and every attention kernel has to dequantize with the
layer's scales and quantize q with a scale of its own. A model without KV scales runs
with ``k_scale == 1``, where those mistakes are invisible.

The default case covers the NHD ``mha_batch_prefill_func`` prefill and the
``paged_attention_ragged`` decode; the unified case (page size 16) covers the
``unified_attention`` decode.
"""

import contextlib
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=420, suite="stage-b-test-1-gpu-small-amd")

FP8_KV_SCALES_MODEL = "amd/Llama-3.1-8B-Instruct-FP8-KV"


class _AiterFp8KvScalesBase(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.6
    use_unified_attention = False
    extra_server_args: list[str] = []

    @classmethod
    def setUpClass(cls):
        cls.model = FP8_KV_SCALES_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._env_stack = contextlib.ExitStack()
        cls._env_stack.enter_context(envs.SGLANG_USE_AITER.override(True))
        cls._env_stack.enter_context(
            envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(cls.use_unified_attention)
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "aiter",
                "--kv-cache-dtype",
                "fp8_e4m3",
                *cls.extra_server_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()


class TestAiterFp8KvScales(_AiterFp8KvScalesBase):
    use_unified_attention = False


class TestAiterFp8KvScalesUnifiedAttention(_AiterFp8KvScalesBase):
    use_unified_attention = True
    # Without a sliding-window KV pool, the CUDA-graph decode page table of
    # unified attention is only filled when page_size > 1.
    extra_server_args = ["--page-size", "16"]


del _AiterFp8KvScalesBase


if __name__ == "__main__":
    unittest.main()
