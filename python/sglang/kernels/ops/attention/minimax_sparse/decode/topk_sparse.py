# Copyright 2025 XunhaoLai. All rights reserved.

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)

# --------------------------------------------------------------------------- #
# gfx950 stage-1 geometry for the sparse decode kernel.
#
# The stock autotune space (num_warps in {4, 8} x num_stages in {2, 3, 4, 5})
# does not contain the gfx950 optimum, and which member of it the autotuner
# picks depends on whatever `seq_lens` happen to be live when the cuda graph
# for a given batch rung is captured -- so a server can spend the whole run on
# a config ~20% slower than num_warps=4. On gfx950 we derive the geometry from
# the launch shape instead of benchmarking it.
#
# Measured on MI355X (256 CU) at the real *per-rank* MiniMax-M3 decode shape.
# Note this is not the config-file shape: num_key_value_heads=4 at TP=8 gives
# num_kv_heads = max(1, 4 // 8) = 1 and num_q_heads = 64 // 8 = 8 per rank
# (models/minimax_m3.py:489), so the K/V pool is [max_slots, 1, 128] and the
# grid's second dimension is 1. Measured cold (rotating KV pools, so the
# gathers miss the 256 MB LLC exactly as they do when 57 layers each stream
# their own multi-GB KV slice). num_warps=4 wins at every batch measured;
# num_stages=2 wins while the grid is at most about one wave deep, and past
# that occupancy already hides the gather latency so the extra stage only
# costs registers.
#
# TARGET_GRID: upstream aims the split count at 256 total workgroups. With
# num_kv_heads == 1 the grid is (batch * NUM_TOPK_CHUNKS, 1), so at batch 64
# that yields NUM_TOPK_CHUNKS=4 and exactly one workgroup per CU with nothing
# to interleave against. Aiming at two workgroups per CU (chunks=8 here)
# measured 17.7 % faster at seq 1024 and 5.6 % at seq 2048.
# --------------------------------------------------------------------------- #
_GFX95_NUM_WARPS = 4
_GFX95_ONE_WAVE_SLACK = 1.25  # workgroups/CU below which num_stages=2 pays off
_GFX95_TARGET_WAVES = 2  # split target, in waves of the CU count
_TUNE_ENV = "SGLANG_MINIMAX_SPARSE_DECODE_TUNE"


@functools.lru_cache(maxsize=None)
def _gfx95_props(device_index: int):
    """(is_gfx95, core_count) for a device; cheap and cached."""
    try:
        props = torch.cuda.get_device_properties(device_index)
    except Exception:  # pragma: no cover - defensive
        return False, 0
    is_gfx95 = bool(torch.version.hip) and "gfx95" in getattr(
        props, "gcnArchName", ""
    )
    return is_gfx95, props.multi_processor_count


def _sparse_decode_tune(device_index: int) -> bool:
    """Use the fixed gfx950 geometry instead of autotuning?

    Unset -> on for gfx95x, off everywhere else. ``0``/``1`` force it.
    """
    override = os.environ.get(_TUNE_ENV)
    if override is not None:
        return override.strip() not in ("", "0", "false", "False")
    return _gfx95_props(device_index)[0]


def _gfx95_stage1_geometry(workgroups: int, device_index: int):
    """(num_warps, num_stages) for a stage-1 launch of ``workgroups`` blocks."""
    cores = _gfx95_props(device_index)[1]
    num_stages = 2
    if cores > 0 and workgroups > _GFX95_ONE_WAVE_SLACK * cores:
        num_stages = 1
    return _GFX95_NUM_WARPS, num_stages


_DECODE_HEURISTICS = {
    "BLOCK_SIZE_H": lambda args: max(
        16, triton.next_power_of_2(args["gqa_group_size"])
    ),
    "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    "BATCH_SIZE_BUCKET": lambda args: triton.next_power_of_2(args["batch_size"]),
}


@triton.heuristics(_DECODE_HEURISTICS)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4, 5]
    ],
    # NUM_TOPK_CHUNKS and BLOCK_SIZE_N set the launch geometry and the inner
    # loop trip count, so a config tuned for one must not be reused for
    # another. ("block_size" was never a kernel argument, so the autotuner
    # silently dropped it from the key.)
    key=[
        "BATCH_SIZE_BUCKET",
        "gqa_group_size",
        "head_dim",
        "HAS_SINK",
        "NUM_TOPK_CHUNKS",
        "BLOCK_SIZE_N",
    ],
)
@triton.jit
def _gqa_share_sparse_decode_kernel(
    q_ptr,  # Q: b x qh x d
    sink_ptr,  # Sink: qh x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    idx_ptr,  # topk index: qh x b x topk
    o_ptr,  # O partial: c x b x qh x d
    lse_ptr,  # lse partial: c x b x qh
    seq_lens,
    slot_ids,
    # shape
    max_slots,
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # META parameters
    BATCH_SIZE_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    IS_FP8: tl.constexpr,
    # Streaming hint for the KV gathers. Every (BLOCK_SIZE_D, BLOCK_SIZE_N)
    # tile is read by exactly one program exactly once, so the per-CU vector L1
    # can never serve a hit; ".cg" keeps the gather out of it. Empty string
    # (the default) reproduces the original loads bit-for-bit.
    KV_CACHE_MODIFIER: tl.constexpr = "",
    # Guard the gathered slot ids by clamping instead of a 64-bit modulo. Both
    # forms are identity for the in-range slot ids req_to_token actually holds;
    # the modulo lowers to a software int64 division in the hot loop.
    CLAMP_SLOT_GUARD: tl.constexpr = False,
):
    # decode program ids: split-K over the topk dimension to give every SM
    # something to do at small batch. pid(0) folds (batch, chunk) together so
    # the grid size = batch_size * NUM_TOPK_CHUNKS.
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    # per-chunk topk range. chunk_size is *runtime* (depends on max_topk which
    # is a runtime arg, not constexpr), so don't annotate as tl.constexpr —
    # doing so produces undefined behavior in Triton.
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk_compiletime = chunk_start_topk + chunk_size_topk
    # get q k start and len after rmpad
    seq_len = tl.minimum(tl.load(seq_lens + pid_b), max_kv_len)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # to avoid bugs when slot_ids is negative
    # get real topk
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum(valid_idx != -1, axis=0)
    chunk_end_topk = tl.minimum(chunk_end_topk_compiletime, real_topk)
    # init pointer
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    # init statistics — kept at -inf so empty chunks (chunk_start >= real_topk)
    # naturally fall out as weight=0 in the merge step.
    if HAS_SINK and pid_c == 0:
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sink_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_sink_h, stride_sink_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale  # (BLOCK_SIZE_H,)
        m_i = qsink
        lse_i = qsink
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
            shape=(gqa_group_size, head_dim),
            strides=(stride_q_h, stride_q_d),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # only iterate over this chunk's topk slice. the load must respect the
    # per-chunk start offset.
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        # load index
        c = tl.load(cur_idx_ptr).to(tl.int32) * BLOCK_SIZE_N
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        # resolve slots for this block via req_to_token
        pos = c + off_n
        pos_mask = pos < seq_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        if CLAMP_SLOT_GUARD:
            # same out-of-range guard, without a per-iteration int64 division
            slots = tl.minimum(tl.maximum(slots, 0), max_slots - 1)
        else:
            slots = (slots + max_slots) % max_slots  # safety against negative
        # load K as (head_dim, BLOCK_SIZE_N) via indirect addressing
        k_off = (
            slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_off,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
            cache_modifier=KV_CACHE_MODIFIER,
        )
        if IS_FP8:
            # fp8 KV cache: with bf16/fp16 Q this widens K to the compute dtype
            # (unit-scaled cache -> exact inverse dequant; k_scale covers
            # calibrated caches). With fp8 Q (fp8 attn-GEMM mode) the cast is a
            # no-op and tl.dot below runs fp8x8 on tensor cores. Matches the
            # bf16 path bit-for-bit when the cache is bf16 (IS_FP8 False ->
            # this branch is compiled out).
            k = k.to(q.dtype)
        # load V as (BLOCK_SIZE_N, head_dim) via indirect addressing
        v_off = (
            slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_off,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
            cache_modifier=KV_CACHE_MODIFIER,
        )
        if IS_FP8:
            # Cast V to the compute dtype. With bf16/fp16 Q this widens (so the
            # `p.to(v.dtype)` below keeps P in the compute dtype); with fp8 Q it
            # is a no-op and P is quantized to e4m3 for the fp8 PV MMA — the
            # same accuracy contract as fmha_sm100's fp8 kernel.
            v = v.to(q.dtype)
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < seq_len - c, 0, float("-inf"))
        # [H, D], [D, N] -> [H, N]
        qk += tl.dot(q, k) * (sm_scale * k_scale)
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        # [H, N], [N, D] -> [H, D]
        acc_o += tl.dot(p.to(v.dtype), v) * v_scale
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    # final scale (matches the old non-split kernel for chunks where lse_i>-inf).
    # For empty chunks (chunk_start_topk >= real_topk) the inner loop never
    # runs, so m_i = lse_i = -inf and naive `tl.exp(m_i - lse_i)` would compute
    # exp(-inf - (-inf)) = exp(NaN) = NaN, then 0 * NaN = NaN poisons o_partial
    # and the merge result. Gate the scale with tl.where so empty chunks emit a
    # clean zero (lse_i stays -inf which the merge correctly turns into weight=0).
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]
    # save partial output and lse for the merge step
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


# Same kernel and same heuristics, minus the autotuner, so the gfx950 path can
# hand num_warps/num_stages straight to the launch. Autotuner.run rejects an
# explicit num_warps kwarg (it would collide with the one from its own config),
# hence the separate entry point rather than a per-call override.
# Decorators apply bottom-up, so the module symbol is
# Heuristics(Autotuner(JITFunction)) and `.fn.fn` is the bare JITFunction. If a
# future Triton reshapes that stack, fall back to the autotuned kernel rather
# than failing at import (which would take the server down).
try:
    _gqa_share_sparse_decode_kernel_fixed = triton.heuristics(_DECODE_HEURISTICS)(
        _gqa_share_sparse_decode_kernel.fn.fn
    )
except Exception:  # pragma: no cover - defensive
    _gqa_share_sparse_decode_kernel_fixed = None


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH, D] — partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    # standard flash-decoding merge in linear (not log2) space, matching the
    # decode kernel which uses tl.exp / tl.log.
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)


@torch.no_grad()
def flash_decode_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="decode")
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    assert topk_idx.shape[0] == num_kv_heads
    assert triton.next_power_of_2(block_size) == block_size, (
        f"block_size must be a power of 2, but got {block_size}"
    )
    # assert slot_ids.max() < max_slots, f"get slot_ids {slot_ids}, but kv_cache shape is {kv_cache.shape}"
    max_kv_len = req_to_token.shape[1]
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    # Pick NUM_TOPK_CHUNKS so total grid ≈ TARGET_GRID. Same constraints as
    # flash_decode_with_topk_idx: must be power of 2 (Triton arange) and must
    # only depend on shape constants (so grid is fixed within a cuda graph).
    # Capped by max_topk because chunks beyond real_topk early-fall-through to
    # the merge-as-zero path; capping avoids wasting blocks at tiny topk.
    device_index = q.device.index if q.device.index is not None else 0
    tune = _gqa_share_sparse_decode_kernel_fixed is not None and _sparse_decode_tune(
        device_index
    )
    TARGET_GRID = 256
    if tune:
        cores = _gfx95_props(device_index)[1]
        if cores > 0:
            TARGET_GRID = cores * _GFX95_TARGET_WAVES
    target = max(
        1,
        min(max_topk, TARGET_GRID // max(1, batch_size * num_kv_heads)),
    )
    NUM_TOPK_CHUNKS = 1 << (target.bit_length() - 1)
    # output tensor: split-K partials, merged into chunk 0 by the merge kernel
    o_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=sparse_out_dtype(q),
        device=q.device,
    )
    lse_partial = torch.empty(
        NUM_TOPK_CHUNKS,
        batch_size,
        num_q_heads,
        dtype=torch.float32,
        device=q.device,
    )
    # launch attention kernel
    grid = (batch_size * NUM_TOPK_CHUNKS, num_kv_heads)
    kernel = (
        _gqa_share_sparse_decode_kernel_fixed
        if tune
        else _gqa_share_sparse_decode_kernel
    )
    extra_kwargs = {}
    if tune:
        num_warps, num_stages = _gfx95_stage1_geometry(grid[0] * grid[1], device_index)
        extra_kwargs = dict(
            num_warps=num_warps,
            num_stages=num_stages,
            KV_CACHE_MODIFIER=".cg",
            CLAMP_SLOT_GUARD=True,
        )
    kernel[grid](
        q,
        sink,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        slot_ids,
        max_slots,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size,
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        IS_FP8=is_fp8,
        **extra_kwargs,
    )
    # merge partials into chunk 0.
    # With a single chunk there is nothing to merge: the merge computes
    # weights = exp(lse - max(lse)) / sum(...) = 1 over a length-1 axis and
    # stores o[0] back onto itself, i.e. it is an identity copy. Skipping it
    # removes one kernel launch (and one round-trip over the whole output) per
    # sparse layer per decode step; at the served shape (batch 64, 4 kv heads)
    # TARGET_GRID always yields NUM_TOPK_CHUNKS == 1, so this is every step.
    if NUM_TOPK_CHUNKS > 1:
        merge_grid = (batch_size, num_q_heads)
        _merge_topk_attn_out_kernel[merge_grid](
            o_partial,
            lse_partial,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        )
    return o_partial[0].contiguous()
