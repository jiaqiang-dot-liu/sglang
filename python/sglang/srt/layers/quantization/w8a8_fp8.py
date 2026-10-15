from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_padding_size,
    get_moe_runner_backend,
)
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()


def _aiter_moe_opt_out() -> bool:
    """``SGLANG_W8A8_FP8_MOE_AITER=0`` pins the historical Triton MoE runner."""
    return not get_bool_env_var("SGLANG_W8A8_FP8_MOE_AITER", "1")


@functools.lru_cache(maxsize=8)
def _aiter_moe_kernel_available(
    padded_inter: int, hidden_size: int, activation: str
) -> bool:
    """Probe once whether aiter's asm fused-MoE has a kernel for this shape.

    ``asm_fmoe`` picks its kernel by ``inter_dim % subGU_n == 0`` and raises when
    no tile divides the (padded) intermediate size -- e.g. Gemma4's 704 has no
    divisor among the shipped 128/192/256/320/384/448/512 tiles, while the
    128-aligned 768 does.  Probing with an 8-expert dummy at load time turns an
    unsupported shape into a silent Triton fallback instead of a crash on the
    first decode.
    """
    try:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe
        from aiter.ops.shuffle import shuffle_weight

        num_experts, num_tokens, topk = 8, 4, 2
        device = torch.cuda.current_device()
        w13 = torch.zeros(
            num_experts, 2 * padded_inter, hidden_size, dtype=fp8_dtype, device=device
        )
        w2 = torch.zeros(
            num_experts, hidden_size, padded_inter, dtype=fp8_dtype, device=device
        )
        fused_moe(
            torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device),
            shuffle_weight(w13, layout=(16, 16)),
            shuffle_weight(w2, layout=(16, 16)),
            torch.ones(num_tokens, topk, dtype=torch.float32, device=device),
            torch.zeros(num_tokens, topk, dtype=torch.int32, device=device),
            quant_type=QuantType.per_Token,
            activation=getattr(
                ActivationType, "Gelu" if activation == "gelu" else "Silu"
            ),
            w1_scale=torch.ones(
                num_experts, 2 * padded_inter, 1, dtype=torch.float32, device=device
            ),
            w2_scale=torch.ones(
                num_experts, hidden_size, 1, dtype=torch.float32, device=device
            ),
        )
        return True
    except Exception as e:  # noqa: BLE001 - any failure means "stay on Triton"
        logger.warning(
            "w8a8_fp8: aiter MoE probe failed for inter=%d hidden=%d act=%s (%s); "
            "keeping the Triton MoE runner.",
            padded_inter,
            hidden_size,
            activation,
            e,
        )
        return False
    finally:
        torch.cuda.empty_cache()


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading
    - If CUTLASS is supported: Per-channel weight quantization is used
    - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = (
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
        )
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return W8A8Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8FP8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Fp8LinearMethod(LinearMethodBase):
    def __init__(self, quantization_config: W8A8Fp8Config):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = layer.weight_scale.detach()
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            if _is_fp8_fnuz:
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=weight_scale
                )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            # If checkpoint not offline quantized, quantize the weights with per-channel quantization.
            if self.cutlass_fp8_supported:
                # if cutlass supported, we use cutlass_scaled_mm
                # which requires per-channel quantization on weight
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
                )
                weight_scale = weight_scale.t().contiguous()
            else:
                # if cutlass not supported, we fall back to use torch._scaled_mm
                # which requires per tensor quantization on weight
                qweight, weight_scale = input_to_float8(layer.weight, dtype=fp8_dtype)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)
        else:
            layer.weight_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )


class W8A8FP8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Fp8Config):
        self.quant_config = quant_config
        self.use_aiter_moe = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=fp8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=fp8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_aiter_moe and self._prepare_aiter_moe_weights(layer):
            return
        layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    @staticmethod
    def _aiter_moe_supported(moe_runner_config: MoeRunnerConfig) -> bool:
        """Static preconditions for routing this layer to aiter's asm MoE.

        Mirrors the guards ``Fp8MoEMethod`` / ``CompressedTensorsW8A8Fp8MoEMethod``
        already apply, plus the ones specific to what ``AiterMoeQuantInfo`` can
        express (no EP expert_mask, no router-weight folding, no combine skip).
        """
        from sglang.srt.runtime_context import get_parallel

        if not _is_hip or _aiter_moe_opt_out():
            return False
        if not get_moe_a2a_backend().supports_aiter():
            return False
        # The aiter runner needs `expert_mask` from the dispatcher for EP, which
        # StandardDispatcher only builds when it thinks the runner is AITER.
        # Keep this bridge to the non-EP case.
        if get_parallel().moe_ep_size > 1:
            return False
        if moe_runner_config.no_combine:
            return False
        if not moe_runner_config.is_gated:
            return False
        if moe_runner_config.activation not in ("silu", "gelu"):
            return False
        if moe_runner_config.apply_router_weight_on_input:
            return False
        # aiter.fused_moe has no routed_scaling_factor / gemm1 alpha-limit knob.
        if moe_runner_config.routed_scaling_factor not in (None, 1.0):
            return False
        if (
            moe_runner_config.gemm1_alpha is not None
            or moe_runner_config.gemm1_clamp_limit is not None
        ):
            return False
        return True

    def _prepare_aiter_moe_weights(self, layer: torch.nn.Module) -> bool:
        """Pad the intermediate dim to the aiter alignment and pre-shuffle.

        Returns False (leaving the weights untouched) when the probe says aiter
        has no kernel for the padded shape; the caller then keeps the Triton
        path and ``create_moe_runner``'s AITER choice is rolled back.
        """
        from aiter.ops.shuffle import shuffle_weight

        num_experts, two_inter, hidden_size = layer.w13_weight.shape
        inter = two_inter // 2
        align = get_moe_padding_size(True)
        padded = ((inter + align - 1) // align) * align

        if not _aiter_moe_kernel_available(
            padded, hidden_size, self.moe_runner_config.activation
        ):
            self.use_aiter_moe = False
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, self.moe_runner_config)
            return False

        if padded != inter:
            # w13 is [gate; up] concatenated on dim 1, so each half is padded
            # separately: [gate | 0 | up | 0]. Zero rows are numerically inert
            # (act(0) * 0 == 0) and the matching zero columns of w2 drop out of
            # the down GEMM; `intermediate_pad` lets the kernel skip them.
            w13 = layer.w13_weight.data
            padded_w13 = torch.zeros(
                num_experts, 2 * padded, hidden_size, dtype=w13.dtype, device=w13.device
            )
            padded_w13[:, :inter] = w13[:, :inter]
            padded_w13[:, padded : padded + inter] = w13[:, inter:]
            layer.w13_weight = Parameter(padded_w13, requires_grad=False)
            del w13, padded_w13
            torch.cuda.empty_cache()

            w13_scale = layer.w13_weight_scale.data
            padded_scale = torch.zeros(
                num_experts,
                2 * padded,
                1,
                dtype=w13_scale.dtype,
                device=w13_scale.device,
            )
            padded_scale[:, :inter] = w13_scale[:, :inter]
            padded_scale[:, padded : padded + inter] = w13_scale[:, inter:]
            layer.w13_weight_scale = Parameter(padded_scale, requires_grad=False)

            w2 = layer.w2_weight.data
            padded_w2 = torch.zeros(
                num_experts, hidden_size, padded, dtype=w2.dtype, device=w2.device
            )
            padded_w2[:, :, :inter] = w2
            layer.w2_weight = Parameter(padded_w2, requires_grad=False)
            del w2, padded_w2
            torch.cuda.empty_cache()
        else:
            layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
            layer.w13_weight_scale = Parameter(
                layer.w13_weight_scale.data, requires_grad=False
            )
            layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.intermediate_pad = padded - inter
        layer.w13_weight.data = shuffle_weight(
            layer.w13_weight.data.contiguous(), layout=(16, 16)
        )
        layer.w2_weight.data = shuffle_weight(
            layer.w2_weight.data.contiguous(), layout=(16, 16)
        )
        torch.cuda.empty_cache()
        return True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        # NOTE: this used to be a hard `MoeRunnerBackend.TRITON` literal, so
        # `--moe-runner-backend` was silently discarded on the w8a8_fp8 MoE path
        # (unlike Fp8MoEMethod / CompressedTensorsW8A8Fp8MoEMethod, which both
        # resolve it). Honour the flag, and on ROCm prefer aiter's asm MoE --
        # including when the inherited flag says `triton`, because that value
        # was never read here and therefore never expressed a kernel choice.
        # `SGLANG_W8A8_FP8_MOE_AITER=0` restores the old behaviour.
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto() or moe_runner_backend.is_triton():
            moe_runner_backend = (
                MoeRunnerBackend.AITER
                if self._aiter_moe_supported(moe_runner_config)
                else MoeRunnerBackend.TRITON
            )
        elif not moe_runner_backend.is_aiter():
            # Any other explicit backend is unsupported by this quant scheme.
            moe_runner_backend = MoeRunnerBackend.TRITON

        self.use_aiter_moe = moe_runner_backend.is_aiter()
        self.runner = MoeRunner(moe_runner_backend, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def get_aiter_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        return AiterMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            # Weights are per-output-channel fp8; activations are dynamically
            # quantized per token inside the kernel (a13/a2 scales are None).
            quant_type=AiterQuantType.PER_TOKEN,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            intermediate_pad=getattr(layer, "intermediate_pad", 0),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        if self.use_aiter_moe:
            return self.runner.run(dispatch_output, self.get_aiter_quant_info(layer))
        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)
