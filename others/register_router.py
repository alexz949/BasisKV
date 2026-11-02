"""Historical router builds (candidate pages, warp sweeps, wmma fallback) from the archived sources in others/csrc."""

from pathlib import Path

from kernels.build import load_cuda_extension

CSRC = Path(__file__).resolve().parent / 'csrc'


def compile_router(root, *, rank, warps, indexed=False, value_dim=80, unroll=1):
    flags = [f'-DBASIS_VALUE_DIM={value_dim}', '-DBASIS_GQA=4', '-DBASIS_PAGE_SIZE=32',
             f'-DBASIS_BASE_RANK={rank}', f'-DBASIS_RESIDUAL_RANK={rank}',
             f'-DBASIS_ROUTER_INDEXED={int(indexed)}', f'-DBASIS_REGISTER_UNROLL={unroll}', '--use_fast_math']
    flags.append(f'-DBASIS_REGISTER_WARPS={warps}' if warps else '-DBASIS_DISABLE_REGISTER_ROUTER')
    return load_cuda_extension(f'basiskv_router_b{rank}_w{warps}_i{int(indexed)}_v{value_dim}_u{unroll}',
        [CSRC / name for name in ('mapped_host_paged_attention.cpp', 'mapped_host_paged_attention.cu', 'conditional_router_page32.cu')],
        cuda_flags=flags, build_directory=root)
