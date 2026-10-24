"""
Set m=128, n=256, k=64 to target the m128n256k64 WGMMA instruction.
Tune num_warps and mainloop stage count.
Also tune TB swizzle group count as sanity check.

To print detailed autotuning logs, use env var TRITON_PRINT_AUTOTUNING=1.
"""

import torch
import triton
import triton.language as tl
from utils import bench, check_err
import argparse
from typing import Optional


autotune_configs = [
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 2}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 3}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 4}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 5}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 2}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 16,
                   'mainloop_stages': 3}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 16,
                   'mainloop_stages': 4}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 16,
                   'mainloop_stages': 5}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 8,
                   'mainloop_stages': 2}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 32,
                   'mainloop_stages': 3}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 32,
                   'mainloop_stages': 4}, num_warps=8),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'tb_swizzle_group_size': 32,
                   'mainloop_stages': 5}, num_warps=8),
]


@triton.autotune(autotune_configs, key=["M", "N", "K"])
@triton.jit
def kernel(
        a_ptr, b_ptr, c_ptr,
        lda, ldb, ldc,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        mainloop_stages: tl.constexpr,
        tb_swizzle_group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks_m, n_blocks_n = tl.cdiv(M, BM), tl.cdiv(N, BN)

    if tb_swizzle_group_size > 0:
        blocks_per_group = n_blocks_n * tb_swizzle_group_size
        gid_m, idx_within_group = (
            pid // blocks_per_group, pid % blocks_per_group
        )
        this_group_size = min(
            # Handle case where TB swizzle size doesn't divide n_blocks_m
            tb_swizzle_group_size,
            n_blocks_m - gid_m * tb_swizzle_group_size
        )
        bid_n, bid_m_within_group = (
            idx_within_group // this_group_size,
            idx_within_group % this_group_size
        )
        bid_m = gid_m * tb_swizzle_group_size + bid_m_within_group
    else:
        bid_m, bid_n = pid // n_blocks_n, pid % n_blocks_n

    a_desc = tl.make_tensor_descriptor(
        base=a_ptr, shape=[M, K], strides=[lda, 1], block_shape=[BM, BK],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=[K, N], strides=[ldb, 1], block_shape=[BK, BN],
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_start in tl.range(0, K, BK, num_stages=mainloop_stages):
        a_block = a_desc.load([bid_m * BM, k_start])
        b_block = b_desc.load([k_start, bid_n * BN])
        acc = tl.dot(a_block, b_block, acc)

    # NOTE: TMA store for output tile makes the kernel slower,
    # so we don't use it.

    rows = bid_m * BM + tl.arange(0, BM)
    cols = bid_n * BN + tl.arange(0, BN)
    tl.store(c_ptr + ldc * rows[:, None] + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def gemm(a, b, c, args):
    assert (
        a.shape[1] == b.shape[0] and c.shape[0] == a.shape[0]
        and c.shape[1] == b.shape[1]
    )
    M, N, K = a.shape[0], b.shape[1], a.shape[1]

    def grid(meta): return (
        triton.cdiv(c.shape[0], meta['BM']) *
        triton.cdiv(c.shape[1], meta['BN']),
    )
    kernel[grid](
        a, b, c, a.stride(0), b.stride(0), c.stride(0), M, N, K,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, nargs="?", default=8192)
    parser.add_argument("n", type=int, nargs="?", default=8192)
    parser.add_argument("k", type=int, nargs="?", default=8192)

    args = parser.parse_args()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    ours_tflops, torch_tflops = bench(args, gemm)
    print(f"Our TFLOP/s: {ours_tflops}")
    print(f"Torch TFLOP/s: {torch_tflops}")

    max_err = check_err(args, gemm)
    print(f"Abs max diff: {max_err}")


if __name__ == "__main__":
    main()
