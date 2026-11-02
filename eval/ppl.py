"""WikiText-2 and document-window C4 perplexity with FP32 cross entropy."""

import argparse
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from core.router_io import read_json, sha256
from eval.common import load_model, model_arguments, protocol, save_result


CHUNK = 512


@torch.inference_mode()
def perplexity(model, sequences, batch_size):
    nlls = []
    for start in range(0, len(sequences), batch_size):
        ids = sequences[start:start + batch_size].to(model.get_input_embeddings().weight.device)
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
        targets = ids[:, 1:].to(logits.device)
        values = torch.zeros(len(ids), dtype=torch.float64, device=logits.device)
        for left in range(0, logits.shape[1], CHUNK):  # FP32 cross entropy CHUNK positions at a time; the full FP32 logits are 1.2 GB per window
            chunk = logits[:, left:left + CHUNK].float()
            losses = F.cross_entropy(chunk.reshape(-1, chunk.shape[-1]), targets[:, left:left + CHUNK].reshape(-1), reduction='none')
            values += losses.reshape(len(ids), -1).double().sum(-1)
        assert torch.isfinite(values).all(), 'non-finite PPL loss'
        nlls.extend(values.tolist())
        print(f'[ppl] windows={min(start + batch_size, len(sequences))}/{len(sequences)}', flush=True)
    tokens = len(sequences) * (sequences.shape[1] - 1)
    return dict(ppl=math.exp(sum(nlls) / tokens), mean_nll=sum(nlls) / tokens, nll_sum=sum(nlls),
                tokens=tokens, windows=len(sequences), sequence_length=sequences.shape[1],
                window_nll=[value / (sequences.shape[1] - 1) for value in nlls], loss_dtype='float32')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--dataset', choices=('wikitext2', 'c4', 'both'), default='both')
    parser.add_argument('--wikitext-tokens', type=Path, help='optional safetensors input_ids; otherwise load WikiText-2 test')
    parser.add_argument('--c4-windows', type=Path, help='calibration.windows --split validation output (windows.safetensors with manifest.json)')
    parser.add_argument('--sequence-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--limit', type=int, help='window limit for a partial evaluation')
    args = parser.parse_args(argv)
    model, tokenizer = load_model(args)
    stamp, metrics = protocol(args, 'ppl'), {}
    if args.dataset in ('wikitext2', 'both'):
        if args.wikitext_tokens is None:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            ids = torch.tensor(tokenizer('\n\n'.join(dataset['text']))['input_ids'])
            stamp['wikitext2'] = dict(dataset='wikitext/wikitext-2-raw-v1', split='test', fingerprint=dataset._fingerprint)
        else:
            ids = load_file(str(args.wikitext_tokens))['input_ids'].long().flatten()
            stamp['wikitext2'] = dict(tokens_sha256=sha256(args.wikitext_tokens), split='test')
        count = ids.numel() // args.sequence_length
        sequences = ids[:count * args.sequence_length].reshape(count, args.sequence_length)
        metrics['wikitext2'] = perplexity(model, sequences[:args.limit], args.batch_size)
    if args.dataset in ('c4', 'both'):
        metadata = read_json(args.c4_windows.with_name('manifest.json'))
        assert metadata['config']['dataset_split'] == 'validation', 'C4 quality uses calibration.windows --split validation, not training windows'
        sequences = load_file(str(args.c4_windows))['input_ids'].long()
        assert sequences.shape[1] == args.sequence_length, 'the C4 windows were built for another sequence length'
        metrics['c4'] = perplexity(model, sequences[:args.limit], args.batch_size)
        stamp['c4'] = dict(windows_sha256=sha256(args.c4_windows), manifest_sha256=sha256(args.c4_windows.with_name('manifest.json')))
    save_result(args, dict(protocol=stamp, metrics=metrics))


if __name__ == '__main__':
    main()
