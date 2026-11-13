"""Offline CPU checks of data contracts, fitting, loading, and cached routed inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]


class StandaloneTest(unittest.TestCase):
    def run_cli(self, *args):
        environment = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
        subprocess.run([sys.executable, '-m', *map(str, args)], cwd=self.directory,
                       env=environment, check=True, stdout=subprocess.DEVNULL)

    def test_value_router_and_cache(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        from inference.model import load_model, install_factors, install_router
        for kind, Config, Model in [('llama', LlamaConfig, LlamaForCausalLM), ('qwen3', Qwen3Config, Qwen3ForCausalLM)]:
            with self.subTest(model=kind), tempfile.TemporaryDirectory() as directory:
                self.directory = directory
                root = Path(directory)
                model_path, windows = root/'model', root/'windows.safetensors'
                config = Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
                    bos_token_id=1, eos_token_id=2, pad_token_id=0)
                Model(config).save_pretrained(model_path)
                tokenizer = Tokenizer(models.WordLevel({f't{i}': i for i in range(64)}, unk_token='t0'))
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='t1', eos_token='t2',
                    unk_token='t0', pad_token='t0').save_pretrained(model_path)
                save_file({'input_ids': torch.randint(3, 64, (2, 2048))}, str(windows))
                (root/'manifest.json').write_text(json.dumps(dict(status='complete', fit_ids=[0], validation_ids=[1],
                    config={'dataset_split': 'validation'})))
                stats, factors, capture, router = (root/name for name in ('stats', 'factors', 'capture', 'router'))
                self.run_cli('calibration.value', '--model', model_path, '--windows', windows, '--output', stats,
                    '--device', 'cpu', '--model-dtype', 'float32')
                self.run_cli('core.fit_value', 'fit-shard', '--snapshot-dir', stats, '--output-dir', factors,
                    '--rank', 4, '--device', 'cpu', '--encoder-sweeps', 1, '--factor-dtype', 'float32')
                self.run_cli('core.fit_value', 'merge', '--output-dir', factors)
                self.run_cli('eval.ppl', '--model', model_path, '--value-factors', factors, '--dataset', 'c4',
                    '--c4-windows', windows, '--sequence-length', 2048, '--dtype', 'float32', '--device', 'cpu', '--output', root/'ppl')
                common = ['--model', model_path, '--value-factors', factors, '--windows', windows,
                    '--output', capture, '--device', 'cpu', '--model-dtype', 'float32', '--base-rank', 2, '--residual-rank', 2]
                self.run_cli('calibration.router', 'moments', *common)
                self.run_cli('core.fit_router', 'base', '--statistics-dir', capture, '--output-dir', capture/'base', '--base-rank', 2)
                self.run_cli('calibration.router', 'fisher', *common)
                self.run_cli('core.fit_router', 'fit', '--statistics-dir', capture, '--output-dir', router,
                    '--residual-rank', 2, '--device', 'cpu', '--sweeps', 1, '--pcg-iterations', 4)
                model = load_model(model_path, dtype='float32', device='cpu')
                install_factors(model, value_factors=factors)
                install_router(model, router, routed_tokens=16)
                out = model(torch.randint(3, 64, (1, 128)), use_cache=True)
                decoded = model(torch.tensor([[3]]), past_key_values=out.past_key_values, use_cache=True)
                self.assertTrue(torch.isfinite(decoded.logits).all())
                self.assertEqual(decoded.past_key_values.layers[0].values.shape[-1], 4)

    def test_fp8_keys_enter_cache_after_rope(self):
        from inference.model import install_value_factors, attention_layers
        from appendix.kv_quantization import install_hooks, e4m3_roundtrip
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        model = LlamaForCausalLM(config).eval()
        install_value_factors(model, None)
        before = []
        attention_layers(model)[0].key_cache_transform.register_forward_pre_hook(
            lambda module, inputs: before.append(inputs[0].detach().clone()))
        install_hooks(model, 'fp8')
        with torch.inference_mode():
            output = model(torch.randint(0, 64, (1, 32)), use_cache=True)
        torch.testing.assert_close(output.past_key_values.layers[0].keys, e4m3_roundtrip(before[0], (-1,)))


if __name__ == '__main__':
    unittest.main()
