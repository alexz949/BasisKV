"""Extensions for the system decode benchmarks: paged attention and the Page4 router.

Both compile per Value width on first use. The paged attention runs for any GQA ratio;
the router uses Page4 with four query heads per KV head
and Base/residual rank 16 (rank 8 also compiles). Argument names and shape contracts are
documented in the CUDA sources, which check them with asserts.
"""

from functools import lru_cache

from kernels.build import CSRC, load_cuda_extension
from kernels.routing_config import PAGE_SIZE, RECENT_TOKENS, ROUTED_TOKENS

QUERIES_PER_KV = 4
RANK = 16


@lru_cache(maxsize=None)
def load_paged_attention(value_dim, queries_per_kv):
    flags = [f'-DBASIS_VALUE_DIM={value_dim}', f'-DBASIS_GQA={queries_per_kv}', f'-DBASIS_PAGE_SIZE={PAGE_SIZE}',
             '--use_fast_math']
    return load_cuda_extension(f'basiskv_paged_attention_p{PAGE_SIZE}_v{value_dim}_g{queries_per_kv}',
                               [CSRC / 'paged_attention.cu'], cuda_flags=flags)


@lru_cache(maxsize=None)
def load_router(value_dim, rank=RANK):
    flags = [f'-DBASIS_VALUE_DIM={value_dim}', f'-DBASIS_GQA={QUERIES_PER_KV}', f'-DBASIS_PAGE_SIZE={PAGE_SIZE}',
             f'-DBASIS_ROUTED_TOKENS={ROUTED_TOKENS}', f'-DBASIS_RECENT_TOKENS={RECENT_TOKENS}',
             f'-DBASIS_BASE_RANK={rank}', f'-DBASIS_RESIDUAL_RANK={rank}', '--use_fast_math']
    return load_cuda_extension(f'basiskv_router_p{PAGE_SIZE}_v{value_dim}_b{rank}', [CSRC / 'router.cu'], cuda_flags=flags)
