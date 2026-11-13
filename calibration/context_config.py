"""Write the static YaRN factor-4 Qwen3 configuration for 128K routing."""
import argparse
from pathlib import Path
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    assert config.model_type == 'qwen3'
    config.rope_parameters = dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768,
                                 rope_theta=config.rope_parameters['rope_theta'])
    config.max_position_embeddings = 131072
    config.save_pretrained(args.output)


if __name__ == '__main__':
    main()
