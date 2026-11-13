"""Reproducibility metadata shared by the L40S system benchmarks."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
import torch


def command(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def metadata():
    root = Path(__file__).resolve().parents[1]
    suffixes = {'.py', '.cu', '.cpp', '.h', '.cuh', '.sh', '.json', '.toml', '.txt'}
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in suffixes
              and p.relative_to(root).parts[0] in
              {'core', 'calibration', 'inference', 'kernels', 'benchmarks', 'eval', 'appendix',
               'pyproject.toml', 'requirements.txt', 'requirements-tp.txt', 'requirements-qwen35.txt'}}
    snapshot=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
    tree=command(['ps','-eo','pid=,ppid='])['stdout'].splitlines()
    parents={int(v[0]):int(v[1]) for line in tree if len(v:=line.split())==2}
    own={os.getpid(),os.getppid()}
    while True:
        children={pid for pid,parent in parents.items() if parent in own}
        if children<=own:break
        own.update(children)
    external=[line for line in snapshot['stdout'].splitlines() if line.strip() and int(line.split(',')[1]) not in own]
    return dict(source_sha256=hashes, hostname=socket.gethostname(), command_line=sys.argv,
                pytorch=torch.__version__, cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version() if torch.cuda.is_available() else None,
                conda_environment=os.environ.get('CONDA_DEFAULT_ENV',sys.prefix),pid=os.getpid(),gpu_process_snapshot=snapshot,external_gpu_processes=external,
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                gpu_inventory=command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,pci.bus_id','--format=csv']),
                allocation={name:os.environ.get(name) for name in ['SLURM_JOB_ID','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES']},
                torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32)


def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(data,indent=2)+'\n')


def distribution(values):
    import math
    import statistics
    ordered = sorted(values)
    return dict(mean=statistics.fmean(ordered), median=statistics.median(ordered),
                p95=ordered[math.ceil(0.95 * len(ordered)) - 1], min=ordered[0], max=ordered[-1])


def arguments(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def add_model_arguments(parser):
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='extended-context Hugging Face configuration')
    parser.add_argument('--arm', choices=('dense', 'basis'), required=True)
    parser.add_argument('--value-factors', type=Path)
    parser.add_argument('--router-dir', type=Path)
    parser.add_argument('--tokens', type=Path, required=True, help='Safetensors with a [samples, tokens] input_ids tensor')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-steps', type=int, default=16)
    parser.add_argument('--decode-steps', type=int, default=128)


def prompts(path, batch, length, row=0):
    from safetensors.torch import load_file
    values = load_file(str(path))['input_ids']
    assert values.shape[0] >= row + batch and values.shape[1] >= length, f'{path} holds too few or too short prompts'
    return values[row:row + batch, :length].long().contiguous()


def load_model(path, world=1, config_path=None):
    from transformers import AutoConfig, AutoModelForCausalLM
    options = {}
    if config_path is not None:
        options['config'] = AutoConfig.from_pretrained(config_path, local_files_only=True)
    if world > 1:
        from transformers.distributed import DistributedConfig
        options['distributed_config'] = DistributedConfig(tp_size=world)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, **options).eval().requires_grad_(False)
    return model if world > 1 else model.cuda()
