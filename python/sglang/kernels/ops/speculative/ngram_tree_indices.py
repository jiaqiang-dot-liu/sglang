# Copyright (C) 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triton port of sgl_kernel's ``reconstruct_indices_from_tree_mask``.

The NGRAM speculative worker needs, from the draft tree mask alone:

* ``positions[b, i]``        = depth of node i (ancestor count) + seq_len[b]
* ``retrieve_index[b, i]``   = b * draft_token_num + i  (flat row index)
* ``retrieve_next_token[b, i]``  = index of i's first (earliest) child, else -1
* ``retrieve_next_sibling[b, i]`` = index of i's next sibling under the same
  parent, else -1

The upstream kernel is one thread per node: for node i, scan row i of the
mask for ancestors j < i (each ancestor -> depth++, the largest such j is the
parent), scan column i downward for the first descendant -> next_token, and
scan the parent's column downward for the first node whose only ancestor in
(i's row range) is the parent itself -> next_sibling.

This Triton version uses one program per node and mirrors that logic exactly
(scans are O(draft_token_num) register loops, ~12-64 iterations, trivially
cheap next to the verify forward). It matches the CUDA kernel's outputs
node-for-node, including the sibling rule: candidate c > i is a sibling of i
iff parent(i) is an ancestor of c AND no node in (parent(i), c) -- i.e. in
columns parent+1..c-1 of c's row -- is an ancestor of c.

The ROCm build of sgl_kernel does not register
``reconstruct_indices_from_tree_mask`` (the ngram_utils.cu source is absent
from setup_rocm.py, and ``common_extension_rocm.cc`` never declares the op),
which made ``--speculative-algorithm NGRAM`` fail at warmup with
AttributeError. This Triton port restores the op on HIP without requiring a
sgl_kernel rebuild.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct_indices_kernel(
    tree_mask_ptr,
    seq_len_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    draft_token_num,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    bs = tl.num_programs(0)
    # (batch_size, draft_token_num) -> flat node id
    bid = pid // draft_token_num
    tid = pid % draft_token_num
    n = draft_token_num

    mask_base = (bid * n * n).to(tl.int64)
    token_base = bid * n

    # ---- depth + parent: scan row `tid` for ancestors j < tid -------------
    offs = tl.arange(0, BLOCK)
    row_off = tid * n
    anc_mask = offs < tid
    anc = tl.load(
        tree_mask_ptr + mask_base + row_off + offs, mask=anc_mask, other=0
    ).to(tl.int32)
    # ancestor j must be an ancestor of tid: mask[b, tid, j] == 1
    anc_cnt = tl.sum(tl.where(anc_mask, anc, 0), axis=0)
    depth = anc_cnt
    # parent = max j < tid with mask[b, tid, j] == 1 (or -1)
    anc_idx = tl.where(anc_mask & (anc != 0), offs, -1)
    parent = tl.max(anc_idx, axis=0)

    seq_len = tl.load(seq_len_ptr + bid)
    tl.store(positions_ptr + token_base + tid, (depth + seq_len).to(tl.int64))
    tl.store(retrive_index_ptr + token_base + tid, (token_base + tid).to(tl.int64))

    # ---- next_token: first i > tid with mask[b, i, tid] == 1 --------------
    # Scan candidate children in ascending order; take the first hit.
    col_off = tl.arange(0, BLOCK)
    child_cand = col_off + tid + 1
    child_valid = child_cand < n
    child_bit = tl.load(
        tree_mask_ptr + mask_base + child_cand * n + tid,
        mask=child_valid,
        other=0,
    ).to(tl.int32)
    # first child = smallest candidate with bit set
    child_hit = tl.where(child_valid & (child_bit != 0), child_cand, n)
    next_token = tl.min(child_hit, axis=0)
    if next_token >= n:
        next_token = -1
    tl.store(
        retrive_next_token_ptr + token_base + tid, next_token.to(tl.int64)
    )

    # ---- next_sibling: first c > tid whose parent == parent(tid) ----------
    # candidate c qualifies iff:
    #   (a) parent(tid) is an ancestor of c:  mask[b, c, parent] == 1
    #   (b) no j in (parent, c) is an ancestor of c  (i.e. c's parent is
    #       exactly parent(tid), not a deeper descendant)
    # Only nodes with the same parent are siblings; a node's ancestors include
    # itself? (mask[b,i,i] is the node itself; the CUDA kernel treats the
    # ancestor scan strictly below i for parent finding, and sibling check
    # scans j in [parent+1, c) exclusive of c.)
    if parent >= 0:
        sib_cand = col_off + tid + 1
        sib_valid = sib_cand < n
        # (a) parent is an ancestor of candidate
        par_bit = tl.load(
            tree_mask_ptr + mask_base + sib_cand * n + parent,
            mask=sib_valid,
            other=0,
        ).to(tl.int32)
        # (b) count ancestors of candidate strictly between parent and cand
        # Vectorized over candidates is awkward in a 1-D program; do a small
        # scalar-style loop over candidates using tl.sum over a (CAND, J)
        # 2-D load instead.
        j_off = tl.arange(0, BLOCK)
        # 2-D gather: for each candidate c (BLOCK) and each j (BLOCK):
        #   bit[c, j] = mask[b, c, j] if parent < j < c else 0
        c_idx = sib_cand[:, None]  # (BLOCK, 1)
        j_idx = j_off[None, :]  # (1, BLOCK)
        in_range = (j_idx > parent) & (j_idx < c_idx) & (sib_valid[:, None])
        bits2d = tl.load(
            tree_mask_ptr + mask_base + c_idx * n + j_idx,
            mask=in_range,
            other=0,
        ).to(tl.int32)
        deeper = tl.sum(bits2d, axis=1)  # (BLOCK,) ancestors strictly between
        qualifies = sib_valid & (par_bit != 0) & (deeper == 0)
        sib_hit = tl.where(qualifies, sib_cand, n)
        next_sibling = tl.min(sib_hit, axis=0)
        if next_sibling >= n:
            next_sibling = -1
    else:
        next_sibling = -1
    tl.store(
        retrive_next_sibling_ptr + token_base + tid, next_sibling.to(tl.int64)
    )


def reconstruct_indices_from_tree_mask_triton(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,  # mutable, out
    retrive_index: torch.Tensor,  # mutable, out
    retrive_next_token: torch.Tensor,  # mutable, out
    retrive_next_sibling: torch.Tensor,  # mutable, out
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Triton replacement for the CUDA-only sgl_kernel op (missing on ROCm).

    Same contract as ``sgl_kernel.speculative.reconstruct_indices_from_tree_mask``:
    in-place fills ``positions`` / ``retrive_index`` / ``retrive_next_token`` /
    ``retrive_next_sibling`` from the flattened [bs, n, n] boolean tree mask.
    """
    BLOCK = triton.next_power_of_2(max(draft_token_num, 1))
    # The 2-D sibling gather materializes a (BLOCK, BLOCK) tile; keep BLOCK
    # bounded (draft trees are <= 64 nodes in practice).
    assert BLOCK <= 256, (
        f"draft_token_num={draft_token_num} too large for the Triton "
        "tree-mask port (BLOCK={BLOCK} > 256)"
    )
    grid = (batch_size * draft_token_num,)
    _reconstruct_indices_kernel[grid](
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_token_num,
        BLOCK=BLOCK,
    )
