from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_device_name, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()
_LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES = 128 * 1024


@functools.lru_cache(maxsize=None)
def _get_cuda_shared_memory_per_block_optin() -> Optional[int]:
    if _is_hip or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        return None
    return getattr(props, "shared_memory_per_block_optin", None)


def _use_low_smem_fp8_default() -> bool:
    smem_limit = _get_cuda_shared_memory_per_block_optin()
    return smem_limit is not None and smem_limit < _LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES


def get_config_file_name(
    E: int,
    N: int,
    dtype: Optional[str],
    block_shape: Optional[int] = None,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    down_moe_selector = "_down" if down_moe else ""
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    if get_exec().deterministic.enable_deterministic_inference:
        logger.warning(
            "Deterministic inference is enabled, using default MoE kernel config."
        )
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(
        E,
        N,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with Triton 3.2.0 results in negative performance gains,
    # so we also include the Triton version as a key for finding the fused_moe_kernel config to achieve the best performance.
    config_dir = os.environ.get(
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # Discover available triton config dirs on disk and search newest-first.
    configs_root = os.path.join(config_dir, "configs")
    available_versions = sorted(
        (
            d.removeprefix("triton_").replace("_", ".")
            for d in os.listdir(configs_root)
            if d.startswith("triton_")
        ),
        key=lambda v: tuple(int(x) for x in v.split(".")),
        reverse=True,
    )

    for try_triton_version in available_versions:
        if try_triton_version == triton_version:
            continue
        try_config_file_path = os.path.join(
            configs_root,
            f"triton_{try_triton_version.replace('.', '_')}",
            json_file_name,
        )
        if os.path.exists(try_config_file_path):
            with open(try_config_file_path) as f:
                logger.warning(
                    f"Config file not found at {config_file_path}. Fallback to triton version {try_triton_version} and use MoE kernel config from {try_config_file_path}. Performance might be sub-optimal!",
                )
                # If a configuration has been found, return it
                return {int(key): val for key, val in json.load(f).items()}

    if down_moe:
        # A separate down-projection config enables the TMA path, but it is
        # optional. Reuse a tuned up-projection config when it is absent so
        # the second GEMM does not silently fall back to the heuristic.
        up_configs = get_moe_configs(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=False,
        )
        if up_configs is not None:
            logger.warning(
                "Down MoE config file not found at %s; reusing the tuned "
                "up-projection config without TMA. Performance might be sub-optimal.",
                config_file_path,
            )
            return up_configs
        logger.warning(
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    else:
        logger.warning(
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    return None


# ---------------------------------------------------------------------------
# Padding-aware BLOCK_SIZE_M for the ROCm fp8_w8a8 fused-MoE default config.
#
# ``config["BLOCK_SIZE_M"]`` is not just the GEMM M-tile. fused_moe.py hands it
# straight to ``moe_align_block_size`` (fused_moe.py, _prepare_moe_launch), which
# rounds *every touched expert's* row segment up to a full block, and the
# kernel's only early exit is ``pid_m * BLOCK_SIZE_M >= num_tokens_post_padded``
# (kernels/ops/moe/fused_moe_triton_kernels.py) -- an all-padding block still
# runs the whole K contraction. Below one block of routed rows per expert the
# block count is pinned to the touched-expert count whatever the alignment is,
# so the padded row count -- and with it the work -- tracks
# ``touched_experts * BLOCK_SIZE_M`` and not the batch. A 32-token decode step
# on a 128-expert / top_k 8 model routes 256 rows, touches ~108 experts, and
# pads to ~6.9k rows at the alignment 64 that ``M <= E`` currently selects.
#
# The shipped heuristic cannot see this: it branches only on ``M`` vs ``E`` and
# never reads ``topk``, so it does not know how many rows an expert actually
# receives. The table below keys on exactly that quantity.
#
# Ported from ROCm/vllm#1069 ("M-aware moe_align block size", W4A16 prefill on
# gfx1151) and re-derived against SGLang's own fused_experts on MI355X (gfx950)
# for Gemma4 fp8_w8a8, E=128 / top_k=8 / N=704 / K=2816: a 540-point
# BLOCK_SIZE_M x BLOCK_SIZE_N x BLOCK_SIZE_K x num_warps x num_stages sweep at
# 7 batch sizes, hipGraph-captured because decode runs under
# --cuda-graph-max-bs-decode and eager timing is host-launch bound.
#
# Tiers were then re-selected for worst case rather than best case across three
# routing distributions (uniform, and two increasingly skewed routers that drop
# the touched-expert count from 128 to 58 and to 31). Skew only ever raises the
# rows an individual *touched* expert receives above the M*topk/E average this
# table can see, so the rule can only ever under-estimate; the tiers below are
# the ones whose worst case over that uncertainty stays flat. A more aggressive
# table (BLOCK_SIZE_M 16 out to M=128) is 5-12% faster under uniform routing but
# gives back 13% under the skewed one, so it is deliberately not used.
#
# Measured fused_experts time, shipped default -> this table (median of 30 graph
# replays, MI355X gfx950, triton 3.6.0), uniform / skewed routing:
#     M=   16   175.3 -> 156.4us  1.120x  |  132.4 -> 125.5us  1.055x
#     M=   32   200.1 -> 184.6us  1.084x  |  168.8 -> 150.6us  1.121x
#     M=   64   213.9 -> 208.6us  1.025x  |  189.0 -> 183.5us  1.030x
#     M=  128   224.5 -> 218.2us  1.029x  |  213.0 -> 210.9us  1.010x
#     M=  192   278.0 -> 224.9us  1.236x  |  281.1 -> 227.0us  1.238x
#     M=  512   303.1 -> 253.4us  1.196x  |  349.4 -> 284.9us  1.226x
#     M= 1024   353.2 -> 318.3us  1.110x  |  400.4 -> 369.3us  1.084x
#     M= 4096   845.7 -> 776.4us  1.089x  |  915.6 -> 805.3us  1.137x
# Over M = 1..8192 x 3 routers the total is 1.100x / 1.115x / 1.106x and the
# single worst point is 0.980x (M=24 under the most skewed router).
#
# Every tier keeps BLOCK_SIZE_K=128, which is what both branches this bypasses
# already used, so the K reduction order per output element is unchanged and the
# results are bit-identical -- measured max_abs_err 0 against the old configs at
# every batch size and both routers.
#
# Scope: ROCm only, and only the no-tuned-config fallback. Devices that ship a
# tuned JSON (MI300X / MI325X) never reach get_default_config, so this affects
# exactly the untuned-device case it was measured on.
#
# Entries are (minimum routed rows per expert, tile), highest tier first. The
# top two tiers are the actual alignment decision -- 128 once every expert fills
# a padding block anyway, 64 below that, which is where the shipped ``M <= E``
# test puts the boundary in the wrong place. The bottom two only re-tune the
# tile at a fixed alignment.
_HIP_FP8_MOE_ALIGN_TIERS: Tuple[Tuple[int, Dict[str, int]], ...] = (
    (
        64,
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        16,
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    # Same alignment as the tier above and as the shipped ``M <= E`` branch;
    # only num_warps moves, which is what the sub-block-per-expert regime wants.
    (
        4,
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 2,
        },
    ),
    # Under ~4 routed rows per expert even a fully skewed router cannot fill a
    # 64-row block, so the smallest alignment is unconditionally safe here.
    (
        0,
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        },
    ),
)


def _hip_padding_aware_fp8_config(
    M: int, E: int, topk: Optional[int]
) -> Optional[Dict[str, int]]:
    """Pick the fp8_w8a8 tile from routed rows per expert, not from ``M`` alone.

    ``BLOCK_SIZE_M`` doubles as the ``moe_align_block_size`` alignment, so an
    oversized tile pads every touched expert up to a full block of work that the
    kernel then actually executes. See ``_HIP_FP8_MOE_ALIGN_TIERS``.

    Returns None off ROCm, or when ``topk`` / ``E`` are not usable, so the
    caller falls back to the previous heuristic.
    """
    if not _is_hip or not topk or E <= 0:
        return None
    rows_per_expert = (M * topk) / E
    for min_rows_per_expert, config in _HIP_FP8_MOE_ALIGN_TIERS:
        if rows_per_expert >= min_rows_per_expert:
            return dict(config)
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if get_exec().deterministic.enable_deterministic_inference:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        return config
    if dtype == "fp8_w8a8":
        if block_shape is None:
            if _use_low_smem_fp8_default():
                config = {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 256,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                }
                if M > E:
                    config = {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 256,
                        "GROUP_SIZE_M": 64,
                        "num_warps": 4,
                        "num_stages": 2,
                    }
            else:
                padding_aware_config = _hip_padding_aware_fp8_config(M, E, topk)
                if padding_aware_config is not None:
                    return padding_aware_config
                config = {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 32,
                    "num_warps": 8,
                    "num_stages": 2 if _is_hip else 4,
                }
                if M <= E:
                    config = {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 128,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                        "num_stages": 2 if _is_hip else 4,
                    }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_hip else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    per_channel_quant: bool = False,
    return_down_config: bool = False,
):
    from sglang.srt.layers.moe.moe_runner.triton_utils import get_config

    down_config = None
    max_block_m = None
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=False,
        )

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(
                M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
            )
        if return_down_config:
            down_configs = get_moe_configs(
                E,
                N,
                dtype,
                block_n,
                block_k,
                per_channel_quant=per_channel_quant,
                down_moe=True,
            )
            if down_configs:
                down_config = down_configs[
                    min(down_configs.keys(), key=lambda x: abs(x - M))
                ]
                down_config = dict(**down_config)
                max_block_m = max(
                    [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
                )
    if return_down_config:
        if (
            down_config is not None
            and config["BLOCK_SIZE_M"] != down_config["BLOCK_SIZE_M"]
        ):
            # Both kernels share one moe_align_block_size sort, so the down
            # config must use the up config's BLOCK_SIZE_M.
            logger.warning_once(
                "down_moe config BLOCK_SIZE_M=%d does not match up config "
                "BLOCK_SIZE_M=%d at M=%d; overriding down BLOCK_SIZE_M to match.",
                down_config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_M"],
                M,
            )
            down_config["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]
        return config, (down_config, max_block_m)
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None
