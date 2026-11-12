"""Table 5: WikiText-2 training windows in the format of `calibration.windows`.

To pack the short WikiText-2 articles into fixed-length windows, the train
split is joined with blank lines, tokenized once, and cut into contiguous
windows. The output feeds `calibration.value` unchanged.

    python -m appendix.wikitext_windows --model MODEL --output WINDOWS \\
        --sequence-length 2048 --fit-windows 256 --heldout-windows 64
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from core.router_io import save_tensors, sha256, write_json

log = logging.getLogger("basiskv.wikitext_windows")
DATASET, CONFIG, SPLIT = "Salesforce/wikitext", "wikitext-2-raw-v1", "train"


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="local Hugging Face model directory (tokenizer)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--dataset-revision")
    args = parser.parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = load_dataset(DATASET, CONFIG, split=SPLIT, revision=args.dataset_revision)
    ids = tokenizer("\n\n".join(rows["text"]), add_special_tokens=False)["input_ids"]
    total, length = args.fit_windows + args.heldout_windows, args.sequence_length
    windows = torch.tensor(ids[: total * length], dtype=torch.int32).reshape(total, length)
    config = dict(model_config_sha256=sha256(model / "config.json"), dataset=DATASET, dataset_config=CONFIG,
                  dataset_split=SPLIT, dataset_revision=args.dataset_revision, sequence_length=length,
                  fit_windows=args.fit_windows, heldout_windows=args.heldout_windows)
    path = output / "windows.safetensors"
    save_tensors(path, {"input_ids": windows})
    write_json(output / "manifest.json", dict(
        status="complete", config=config, sha256=sha256(path), shape=[total, length], model=str(model),
        model_config_sha256=config["model_config_sha256"], fit_ids=list(range(args.fit_windows)),
        validation_ids=list(range(args.fit_windows, total)),
        packing="contiguous windows of the blank-line-joined train split; no special tokens", records=[]))
    log.info("wrote %d windows of %d tokens to %s", total, length, output)


if __name__ == "__main__":
    main()
