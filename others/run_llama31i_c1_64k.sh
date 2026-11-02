#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basis
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONPATH=.
run=results/llama31i_v96_64k
mkdir -p "$run/logs"
python -u -m others.llama31_instruct_c1_64k prepare > "$run/logs/prepare.log" 2>&1
pids=()
for shard in 0 1 2 3; do
    srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:1 \
        python -u -m others.llama31_instruct_c1_64k capture --shard "$shard" \
        > "$run/logs/capture_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u -m others.llama31_instruct_c1_64k merge > "$run/logs/merge_capture.log" 2>&1
pids=()
for shard in 0 1 2 3; do
    srun --exclusive --exact -N1 -n1 -c2 --gres=gpu:1 \
        python -u -m core.fit_value fit-shard \
        --snapshot-dir "$run/covariance" --output-dir "$run/c1" \
        --fit-windows 64 --validation-windows 16 --rank 96 \
        --work-dtype float32 --factor-dtype float32 \
        --layer-shard-index "$shard" --layer-shard-count 4 --torch-num-threads 2 --resume \
        > "$run/logs/fit_${shard}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
python -u -m core.fit_value merge --output-dir "$run/c1" \
    > "$run/logs/merge_c1.log" 2>&1
