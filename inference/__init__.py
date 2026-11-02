"""Optional vLLM registration; ordinary HF inference imports no vLLM code."""

COMPACT_ARCHITECTURE = 'BasisKVQwen3ForCausalLM'
DENSE_ARCHITECTURE = 'BasisKVDenseQwen3ForCausalLM'
FOLDED_ARCHITECTURE = 'BasisKVFoldedQwen3ForCausalLM'
QWEN35_ARCHITECTURE = 'BasisKVQwen35ForCausalLM'


def register():
    from vllm import ModelRegistry

    for architecture in (COMPACT_ARCHITECTURE, DENSE_ARCHITECTURE, FOLDED_ARCHITECTURE, QWEN35_ARCHITECTURE):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f'inference.vllm:{architecture}')
