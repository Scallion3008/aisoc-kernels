import triton
import argparse
import torch
from collections.abc import Callable

type MatmulFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, argparse.Namespace]
]


def tflops(M, N, K, ms): return 2*M*N*K / (ms * 1e-3) / 1e12


def bench_one(fn):
    return triton.testing.do_bench(fn, warmup=1000, rep=1000)


def bench(args: argparse.Namespace, matmul_fn: MatmulFn):
    a = torch.randn((args.m, args.k), device='cuda', dtype=torch.float16) - 0.5
    b = torch.randn((args.k, args.n), device='cuda', dtype=torch.float16) - 0.5
    c = torch.randn((args.m, args.n), device='cuda', dtype=torch.float16) - 0.5

    ours_ms = bench_one(lambda: matmul_fn(a, b, c, args))
    torch_ms = bench_one(lambda: a @ b)

    ours_tflops = tflops(args.m, args.n, args.k, ours_ms)
    torch_tflops = tflops(args.m, args.n, args.k, torch_ms)
    return ours_tflops, torch_tflops


def check_err(args: argparse.Namespace, matmul_fn: MatmulFn):
    a = torch.randn((args.m, args.k), device='cuda', dtype=torch.float16) - 0.5
    b = torch.randn((args.k, args.n), device='cuda', dtype=torch.float16) - 0.5
    c = torch.randn((args.m, args.n), device='cuda', dtype=torch.float16) - 0.5

    matmul_fn(a, b, c, args)
    torch.cuda.synchronize()
    c_torch = a @ b

    return (c - c_torch).abs().max()
