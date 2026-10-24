# GPU kernels for AI/ML: supporting repo

## Structure

`kernels/` directory contains Triton kernels.

## Notes

Increasing pipeline stages seems to work better after enabling TB swizzle; maybe swap the order?
