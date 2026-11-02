"""Locate attention modules without treating recurrent blocks as attention."""

def c1_attention_layers(model):
    kind = model.config.model_type
    assert kind in ('llama', 'qwen3')
    result = [(index, block.self_attn) for (index, block) in enumerate(model.model.layers)]
    assert result
    return result
