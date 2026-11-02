"""Load a local model and install current BasisKV Value and router factors."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from core.factor_io import DECODER_KEY, ENCODER_KEY, FORMAT, read_layer
from core.router import conditional_routing_query_projector
from core.router_io import load_record, read_json
from inference.attention import ValueAttention

def attention_layers(model):
    return {i: block.self_attn for (i, block) in enumerate(model.model.layers) if hasattr(block, 'self_attn')}

def position_embeddings(config, sequence_length, device):
    """FP32 calibration phases from the same native rotary configuration as inference."""
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Rotary
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Rotary
    with torch.device('cpu'):
        rotary = Rotary(config)
    rotary = rotary.to(device)
    return rotary(torch.empty(1, device=device, dtype=torch.float32), torch.arange(sequence_length, device=device)[None])

def load_model(path, *, config=None, dtype='bfloat16', device='cuda:0', device_map='none', attention_backend='sdpa'):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = config if config is not None else AutoConfig.from_pretrained(path, local_files_only=True)
    text = getattr(config, 'text_config', config)
    loader = AutoModelForCausalLM
    if text.model_type.startswith('qwen3_5'):
        from transformers import Qwen3_5ForCausalLM
        (loader, config) = (Qwen3_5ForCausalLM, text)
    options = dict(config=config, dtype=getattr(torch, dtype), local_files_only=True, attn_implementation=attention_backend, device_map={'': device} if device_map == 'none' else device_map)
    model = loader.from_pretrained(path, **options).eval().requires_grad_(False)
    return model

@torch.no_grad()
def install_value_factors(model, directory):
    """Read core.fit_value results directly; None installs the dense endpoint."""
    (config, layers) = (model.config, attention_layers(model))
    assert config.model_type in ('llama', 'qwen3'), f'Value factors are implemented for Llama and Qwen3 attention, not {config.model_type}'
    manifest = read_json(Path(directory) / 'results.json') if directory is not None else None
    if manifest is not None:
        assert manifest['format'] == FORMAT and manifest['status'] == 'complete', f'{directory} is not a complete Value-factor bank'
    for index, native in layers.items():
        if manifest is None:
            encoder = torch.eye(native.head_dim).expand(config.num_key_value_heads, -1, -1)
            decoder = native.o_proj.weight.detach().T.reshape(config.num_attention_heads, native.head_dim, config.hidden_size)
        else:
            record = read_layer(Path(directory), index, None)
            assert record is not None and record['artifact'] == manifest['artifacts'][str(index)], f'layer {index} of {directory} is missing or corrupt'
            tensors = load_file(str(Path(directory) / record['artifact']['file']))
            (encoder, decoder) = (tensors[ENCODER_KEY], tensors[DECODER_KEY])
        model.model.layers[index].self_attn = ValueAttention(native, encoder, decoder)
    return model

def install_factors(model, *, value_factors=None, gated_value_bank=None, gdn_bank=None):
    """The same installation path is used by calibration and quality evaluation."""
    if model.config.model_type.startswith('qwen3_5'):
        from inference.qwen35 import GatedVRuntime, Qwen35PrivateAGRuntime, load_bank
        assert value_factors is None, 'Qwen3.5 uses its separate gated-V objective'
        if gated_value_bank is not None:
            GatedVRuntime(model, load_bank(gated_value_bank)['layers']).install()
        if gdn_bank is not None:
            Qwen35PrivateAGRuntime(model, gdn_bank).install()
    else:
        assert gated_value_bank is None and gdn_bank is None, 'gated-V and GDN banks are Qwen3.5 artifacts'
        if value_factors is not None:
            install_value_factors(model, value_factors)
    return model

def router_factors(directory, layer):
    """Router tensors and calibration protocol of one layer."""
    payload, metadata = load_record(Path(directory) / f'layer_{layer:03d}.safetensors')
    return payload, metadata['protocol']

def install_router(model, directory, *, page_size=4, routed_tokens=2048, pinned_pages=0):
    """Page size and pinned prefix pages should match the router's calibration protocol; the recent tail comes from it."""
    assert routed_tokens % page_size == 0 and 0 <= pinned_pages < routed_tokens // page_size, 'the routed budget must be whole pages beyond the pinned prefix'
    if not model.config.model_type.startswith('qwen3_5') and not all(isinstance(layer, ValueAttention) for layer in attention_layers(model).values()):
        install_value_factors(model, None)
    for index, attention in attention_layers(model).items():
        payload, protocol = router_factors(directory, index)
        base, residual = protocol['base_rank'], protocol['residual_rank']
        device = next(attention.parameters()).device
        attention.router = {name: payload[f'{name}_b{base}'].to(device=device, dtype=torch.float32)
                            for name in ('base_left', 'base_right', 'base_bias')}
        attention.router['residual_encoder'] = payload[f'residual_encoder_b{base}_r{residual}'].to(device=device, dtype=torch.float32)
        attention.router['projector'] = conditional_routing_query_projector(payload[f'residual_query_b{base}_r{residual}']).to(device=device, dtype=torch.float32)
        attention.page_size, attention.routed_tokens, attention.recent_tokens = page_size, routed_tokens, protocol['excluded_recent_tokens']
        attention.pinned_pages = pinned_pages
    return model
