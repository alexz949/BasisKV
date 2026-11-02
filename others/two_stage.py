"""Compile the WMMA candidate-page router used by the Ampere benchmark."""

from others.register_router import compile_router


def compile_fine(output):
    return compile_router(output, rank=16, warps=0, indexed=True, value_dim=96)
