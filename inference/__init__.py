"""Optional vLLM registration; ordinary HF inference imports no vLLM code."""

# Compact-Value tensor-parallel adapters, keyed by the Hugging Face `model_type` of the checkpoint.
COMPACT_ARCHITECTURES = {'llama': 'BasisKVLlamaForCausalLM', 'qwen3': 'BasisKVQwen3ForCausalLM'}


def register():
    from vllm import ModelRegistry

    for architecture in COMPACT_ARCHITECTURES.values():
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f'inference.vllm:{architecture}')
