"""Sample C4 windows: one excerpt per distinct document, packed into fixed-length windows.

    python -m calibration.windows --model MODEL --output WINDOWS --sequence-length 2048 --fit-windows 256 --heldout-windows 64
    python -m calibration.windows --model MODEL --output C4_VALIDATION --split validation --fit-windows 128 --heldout-windows 0

Fit and held-out windows come from disjoint documents. The manifest lists both id ranges
and the sampled documents; rerunning with the same settings is a no-op. Profile windows
for rank allocation use another --seed for a separate profiling sample.
The second form builds the 128 fixed C4 validation windows that eval.ppl reports on.

"""

import argparse
import hashlib
import logging
import random
from pathlib import Path

import torch

from calibration.common import setup
from core.router_io import read_json, save_tensors, sha256, write_json

log = logging.getLogger("basiskv.windows")
DATASET, DATASET_CONFIG, TEXT_FIELD = "allenai/c4", "en", "text"
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SHUFFLE_BUFFER = 10000
PIECE_LENGTH = 4096  # tokens per excerpt; a window holds sequence_length // PIECE_LENGTH excerpts, one when shorter


def main(argv=None):
    setup()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="tokenizer source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260821, help="stream shuffle and excerpt offsets; 20260821 is the paper's Table 1 calibration bank")
    parser.add_argument("--split", choices=("train", "validation"), default="train", help="train for calibration, validation for eval.ppl")
    args = parser.parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model, output = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    length = args.sequence_length
    piece_length = min(length, PIECE_LENGTH)
    config = dict(model=str(model), dataset=DATASET, dataset_config=DATASET_CONFIG, dataset_revision=REVISION, dataset_split=args.split,
                  seed=args.seed, sequence_length=length, piece_length=piece_length, fit_windows=args.fit_windows,
                  heldout_windows=args.heldout_windows)
    identity = dict(model_config_sha256=sha256(model / 'config.json'),
                    tokenizer_sha256=sha256(model / 'tokenizer.json'))
    path, manifest_path = output / "windows.safetensors", output / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        assert existing["config"] == config, "window inputs changed"
        assert all(existing[key] == value for key, value in identity.items())
        assert existing['sha256'] == sha256(path)
        log.info("windows already complete: %s", path)
        return
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset = load_dataset(DATASET, DATASET_CONFIG, split=args.split, revision=REVISION, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=SHUFFLE_BUFFER)
    rng, seen, pieces, records = random.Random(args.seed), set(), [], []
    pieces_per_window = length // piece_length
    total = args.fit_windows + args.heldout_windows
    for index, row in enumerate(dataset):
        text = row[TEXT_FIELD]
        document = hashlib.sha256(text.encode()).hexdigest()
        if document in seen:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < piece_length:
            continue
        start = rng.randint(0, len(ids) - piece_length)
        pieces.append(torch.tensor(ids[start : start + piece_length], dtype=torch.int32))
        seen.add(document)
        records.append(dict(document_sha256=document, stream_index=index, start=start))
        if len(pieces) % pieces_per_window == 0:
            log.info("windows %d/%d", len(pieces) // pieces_per_window, total)
        if len(pieces) == total * pieces_per_window:
            break
    save_tensors(path, {"input_ids": torch.stack(pieces).reshape(total, length)})
    write_json(manifest_path, dict(
        status="complete", config=config, shape=[total, length], model=str(model),
        sha256=sha256(path), **identity,
        fit_ids=list(range(args.fit_windows)), validation_ids=list(range(args.fit_windows, total)),
        packing="one excerpt per distinct document; no separators or special tokens", records=records))


if __name__ == "__main__":
    main()
