# GPU kernels for AI/ML: supporting repo

## Structure

`kernels/` directory contains Triton kernels.

Optimization ladder:

0. Baseline tiled matmul
1. Threadblock swizzle
2. Pipelining
3. TMA
4. Autotuning

## Running

`uv sync` then `uv run path/to/kernel.py`
