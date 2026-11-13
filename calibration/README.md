# Calibration

The root Quick Start covers Llama and Qwen3, including adaptive rank allocation
and the mixed C4/synthetic Page-Fisher calibration bank.

## Qwen3.5

Install its CUDA sequence kernels in the base environment, with `CUDA_HOME`
set as in the benchmark guide:

```bash
pip install ninja
uv pip install -r requirements-qwen35.txt --no-build-isolation --torch-backend=cu129
```


Qwen3.5 fits gated full-attention Values with twelve ALS sweeps. Its
Gated-DeltaNet output projections retain 75% of their width.

```bash
python -m calibration.windows --model QWEN35 --sequence-length 2048 \
  --fit-windows 256 --heldout-windows 64 --output outputs/qwen35/windows
python -m calibration.qwen35 capture --model QWEN35 \
  --windows outputs/qwen35/windows/windows.safetensors --output outputs/qwen35/statistics
python -m calibration.qwen35 fit --statistics outputs/qwen35/statistics \
  --rank 192 --sweeps 12 --output outputs/qwen35/V192
python -m eval.ppl --model QWEN35 --gated-value-bank outputs/qwen35/V192/value.pt \
  --gdn-bank outputs/qwen35/V192/gdn.pt --dataset wikitext2 --output outputs/qwen35/ppl
```

The V192 bank is the routing configuration. For Page-Fisher capture, replace
`--value-factors` in the root workflow with
`--gated-value-bank outputs/qwen35/V192/value.pt --gdn-bank outputs/qwen35/V192/gdn.pt`.
Use the same bank pair for both capture stages and for evaluation.

## Qwen3.5 adaptive Value budgets

Fit candidate full-attention ranks from the same captures, profile their
single-layer KL changes, and use the shared allocation solver. The GDN bank
stays fixed across Value-rank candidates.

```bash
banks=()
for rank in 64 96 128 160 192 224 256; do
  python -m calibration.qwen35 fit --statistics outputs/qwen35/statistics \
    --rank "$rank" --sweeps 12 --output "outputs/qwen35/R${rank}"
  banks+=("${rank}=outputs/qwen35/R${rank}")
done
python -m calibration.windows --model QWEN35 --sequence-length 2048 \
  --fit-windows 128 --heldout-windows 0 --seed 20260828 --output outputs/qwen35/profile
python -m calibration.probe_kl --model QWEN35 \
  --windows outputs/qwen35/profile/windows.safetensors \
  --banks 64=outputs/qwen35/R64 128=outputs/qwen35/R128 192=outputs/qwen35/R192 \
  --gdn-bank outputs/qwen35/R192/gdn.pt --anchor-rank 128 --low-rank 64 --high-rank 192 \
  --output outputs/qwen35/probes
for target in 128 192; do
  python -m core.allocate --banks "${banks[@]}" --probes outputs/qwen35/probes/probes.json \
    --target-average-rank "$target" --output "outputs/qwen35/adaptive${target}"
done
```

Evaluate each allocated `value.pt` together with the fixed
`outputs/qwen35/R192/gdn.pt`. Mean ranks 128 and 192 correspond to 50% and
75% retention of the 256-dimensional full-attention Values.
