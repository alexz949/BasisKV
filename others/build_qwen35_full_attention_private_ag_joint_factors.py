#!/usr/bin/env python3
"""Build decoder-closed ALS factors for Qwen3.5 full-attention Wo wires."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from others.qwen35_full_attention_private_ag_runtime import FACTOR_FORMAT
from others.build_qwen35_gdn_private_ag_joint_factors import Qwen35PrivateAGBuildSpec, main
from others.collect_qwen35_full_attention_wo_activations import FORMAT as MOMENT_FORMAT


FULL_ATTENTION_BUILD_SPEC = Qwen35PrivateAGBuildSpec(
    description=__doc__,
    block_type="full_attention",
    moment_format=MOMENT_FORMAT,
    factor_format=FACTOR_FORMAT,
    tensor_name_template=(
        "model.language_model.layers.{layer_index}.self_attn.o_proj.weight"
    ),
    log_label="FullAttentionPrivateAGBuild",
    objective="decoder_closed_c1_als_full_attention_o_proj_reconstruction",
)


if __name__ == "__main__":
    main(FULL_ATTENTION_BUILD_SPEC)
