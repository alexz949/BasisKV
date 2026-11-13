"""Assemble the routing bank: 16 C4 fit + 16 retrieval fit + 16 C4 held-out windows.

    python -m calibration.mix_windows --c4 C4 --synthetic RETRIEVAL --output MIXED
"""

import argparse
from pathlib import Path

from calibration.retrieval import FAMILIES, SEQUENCE_LENGTH


def window_sources(c4_manifest, synthetic_manifest):
    """Preserve the C4 diagnostic set and replace the second half of its fit set."""
    fit, heldout = c4_manifest['fit_ids'], c4_manifest['validation_ids']
    assert len(fit) == 32 and len(heldout) == 16
    assert len(set(fit + heldout)) == 48
    assert c4_manifest['shape'] == [48, SEQUENCE_LENGTH]
    assert sorted(fit + heldout) == list(range(48))
    assert synthetic_manifest['shape'] == [16, SEQUENCE_LENGTH]
    assert synthetic_manifest['families'] == FAMILIES
    return ([('c4', i) for i in fit[:16]] + [('synthetic', i) for i in range(16)]
            + [('c4', i) for i in heldout])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c4', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import json
    import torch
    from safetensors.torch import load_file
    from core.router_io import read_json, save_tensors, sha256, write_json

    manifests, banks = {}, {}
    for name, directory in (('c4', args.c4), ('synthetic', args.synthetic)):
        manifest = read_json(directory / 'manifest.json')
        assert manifest['status'] == 'complete'
        assert manifest['sha256'] == sha256(directory / 'windows.safetensors')
        bank = load_file(str(directory / 'windows.safetensors'))['input_ids']
        assert list(bank.shape) == manifest['shape']
        manifests[name], banks[name] = manifest, bank
    c4, synthetic = manifests['c4'], manifests['synthetic']
    for key in ('model_config_sha256', 'tokenizer_sha256'):
        assert c4[key] == synthetic[key]
    assert c4['config']['dataset_split'] == 'train'
    assert c4['config']['piece_length'] == 4096
    assert sha256(args.c4 / 'manifest.json') in {m['sha256'] for m in synthetic['excluded_manifests']}
    metadata_path = args.synthetic / 'metadata.jsonl'
    assert synthetic['metadata_sha256'] == sha256(metadata_path)
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert len(rows) == 16
    c4_documents = {r['document_sha256'] for r in c4['records']}
    synthetic_documents = {r['document_sha256'] for row in rows for r in row['c4_documents']}
    assert c4_documents.isdisjoint(synthetic_documents)
    sources = window_sources(c4, synthetic)
    packed = torch.stack([banks[source][index].to(torch.int32) for source, index in sources])
    save_tensors(args.output / 'windows.safetensors', {'input_ids': packed})
    write_json(args.output / 'manifest.json', dict(
        status='complete', condition='c4_retrieval_50_50', shape=list(packed.shape),
        sha256=sha256(args.output / 'windows.safetensors'),
        model=c4['model'], model_config_sha256=c4['model_config_sha256'],
        tokenizer_sha256=c4['tokenizer_sha256'],
        fit_ids=list(range(32)), validation_ids=list(range(32, 48)),
        composition='fit 0-15: C4; fit 16-31: synthetic retrieval; held-out 32-47: C4 diagnostics',
        windows=[dict(window=i, source=source, source_window=index) for i, (source, index) in enumerate(sources)],
        sources={name: dict(manifest_sha256=sha256(directory / 'manifest.json'),
                            windows_sha256=manifests[name]['sha256'])
                 for name, directory in (('c4', args.c4), ('synthetic', args.synthetic))},
        source_sha256=sha256(Path(__file__))))
    print('mixed routing windows complete', tuple(packed.shape), args.output, flush=True)


if __name__ == '__main__':
    main()
