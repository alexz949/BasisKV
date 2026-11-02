#!/usr/bin/env python3
"""Qwen3-32B R80 generation comparison using the shared lm-eval protocol."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appendix import eval_qwen3_8b_r80_generation_vllm as evaluator

evaluator.FORMAT = "basisserve.qwen3_32b.r80_generation_vllm.v1"
evaluator.STAGE_FORMAT = "basisserve.qwen3_32b.r80_generation_vllm.task.v1"
evaluator.CHECKPOINT_FORMAT = "basisserve.qwen3_32b.iclr_v_factors.v1"
evaluator.EXPECTED_GPU = "NVIDIA H200 NVL"
evaluator.ATTENTION_CONFIG = {"backend": "TRITON_ATTN"}
evaluator.FOLDED_ARCHITECTURE = "BasisKVAppendixFoldedQwen3ForCausalLM"
evaluator.TASKS = ("gsm8k", "ifeval")
evaluator.EXPECTED_ARMS = {
    "Q3-32B-Dense": {"method": "dense"},
    "Q3-32B-C1-R80": {"method": "c1-two-sided-kl", "equivalent_rank_target": 80},

}
evaluator.SUPPORTED_ARMS = tuple(evaluator.EXPECTED_ARMS)

if __name__ == "__main__":
    sys.exit(evaluator.evaluate(evaluator.parse_args()))
