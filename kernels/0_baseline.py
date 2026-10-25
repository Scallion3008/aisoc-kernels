import triton
import triton.language as tl
import argparse
from utils import bench, check_err

"""
Expected FLOP/s: 263.236747406829
Torch: 417.2575151985287
"""


@triton.jit
def kernel(
        a_ptr, b_ptr, c_ptr,
        lda, ldb, ldc,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    pid = tl.program_id(0)
    n_blocks_n = tl.cdiv(N, BN)
    bid_m, bid_n = pid // n_blocks_n, pid % n_blocks_n

    rows = bid_m * BM + tl.arange(0, BM)
    cols = bid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_start in tl.range(0, K, BK):
        k_curr = k_start + tl.arange(0, BK)
        a_block = tl.load(a_ptr + lda * rows[:, None] + k_curr[None, :],
                          mask=(rows[:, None] < M) & (k_curr[None, :] < K),
                          other=0.0)
        b_block = tl.load(b_ptr + ldb * k_curr[:, None] + cols[None, :],
                          mask=(k_curr[:, None] < K) & (cols[None, :] < N),
                          other=0.0)
        acc = tl.dot(a_block, b_block, acc)

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

        # NOTE: setting BM=BN=128 makes kernel 20x slower on T4
        # (2 -> 0.09 TFLOP/s), likely due to SMEM exhaustion.

        BM=128, BN=128, BK=64, num_warps=8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, nargs="?", default=16384)
    parser.add_argument("n", type=int, nargs="?", default=16384)
    parser.add_argument("k", type=int, nargs="?", default=16384)

    args = parser.parse_args()

    ours_tflops, torch_tflops = bench(args, gemm)
    print(f"Our TFLOP/s: {ours_tflops}")
    print(f"Torch TFLOP/s: {torch_tflops}")

    max_err = check_err(args, gemm)
    print(f"Abs max diff: {max_err}")


if __name__ == "__main__":
    main()
