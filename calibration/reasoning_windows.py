"""Table 6: build Mixed and Reasoning windows from the target model's own responses."""
import argparse
import hashlib
from pathlib import Path
import random
import torch
from safetensors.torch import load_file
from core.router_io import read_json, save_tensors, write_json

DATASET = 'open-thoughts/OpenThoughts3-1.2M'
SOURCES = {'math': 'ai2-adapt-dev/openmath-2-math', 'code': 'nvidia/OpenCodeReasoning'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--c4-windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device-map', choices=('none','auto','balanced'), default='none')
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--max-new-tokens', type=int, default=6144)
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from inference.model import load_model
    revision = HfApi().dataset_info(DATASET).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args.model, device=args.device, device_map=args.device_map)
    dataset = load_dataset(DATASET, split='train', streaming=True, revision=revision).shuffle(seed=args.seed, buffer_size=10000)
    windows, records, seen = {name: [] for name in SOURCES}, [], set()
    rng = random.Random(args.seed)
    for row in dataset:
        domain = next((key for key, source in SOURCES.items() if row['source'] == source), None)
        if domain is None or len(windows[domain]) == 128:
            continue
        prompt = row['conversations'][0]['value']
        fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rendered = tokenizer.apply_chat_template([dict(role='user',content=prompt)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids
        if ids.shape[1] > 1536:
            continue
        with torch.inference_mode():
            response = model.generate(ids.to(model.get_input_embeddings().weight.device),
                max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)[0].cpu()
        if response.numel() < 2048:
            continue
        start = rng.randint(0,response.numel()-2048)
        windows[domain].append(response[start:start+2048].to(torch.int32))
        records.append(dict(domain=domain,source=row['source'],prompt_sha256=fingerprint,start=start,
                            response_tokens=response.numel(),window=len(windows[domain])-1))
        print(f'{domain}: {len(windows[domain])}/128',flush=True)
        if all(len(items)==128 for items in windows.values()):
            break
    assert all(len(items)==128 for items in windows.values()), 'not enough target-model responses for the requested recipe'
    manifest = read_json(args.c4_windows.with_name('manifest.json'))
    c4 = load_file(str(args.c4_windows))['input_ids']
    assert c4.shape[1] == 2048 and len(manifest['fit_ids']) >= 128 and len(manifest['validation_ids']) >= 64
    heldout = c4[manifest['validation_ids'][:64]].to(torch.int32)
    math, code = (torch.stack(windows[name]) for name in ('math','code'))
    recipes = dict(mixed=torch.cat([c4[manifest['fit_ids'][:128]].to(torch.int32),math[:64],code[:64],heldout]),
                   reasoning=torch.cat([math,code,heldout]))
    for name,tokens in recipes.items():
        output=args.output/name
        save_tensors(output/'windows.safetensors',{'input_ids':tokens.contiguous()})
        write_json(output/'manifest.json',dict(status='complete',fit_ids=list(range(256)),validation_ids=list(range(256,320)),
            model=str(args.model),dataset=DATASET,dataset_revision=revision,seed=args.seed,records=records,
            config=dict(sequence_length=2048,fit_windows=256,heldout_windows=64,dataset_split='train')))


if __name__ == '__main__':
    main()
