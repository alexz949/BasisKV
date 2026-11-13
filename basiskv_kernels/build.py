"""One lazy CUDA/NCCL extension builder for the release kernels."""

import fcntl
import os
from pathlib import Path
import tempfile

import torch


CSRC = Path(__file__).resolve().parent / 'csrc'


def load_cuda_extension(name, sources, *, cuda_flags=(), nccl=False, build_directory=None, verbose=None):
    from torch.utils import cpp_extension

    assert torch.cuda.is_available(), 'CUDA kernel builds require a GPU allocation'
    configured = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    assert configured is not None, 'Set CUDA_HOME to the selected CUDA toolkit'
    toolkit = Path(configured).expanduser().resolve()
    assert (toolkit / 'bin/nvcc').is_file()
    includes = [CSRC, *[path for path in (toolkit / 'include', toolkit / 'targets/x86_64-linux/include') if path.is_dir()]]
    assert any((path / 'cuda_runtime.h').is_file() for path in includes)
    sources = [Path(path).resolve() for path in sources]
    assert all(path.is_file() for path in sources)
    major, minor = torch.cuda.get_device_capability()
    os.environ['CUDA_HOME'] = str(toolkit)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}')
    cpp_extension.CUDA_HOME = str(toolkit)

    # Match the C++ runtime already loaded by Torch when using the system compiler.
    maps = Path('/proc/self/maps')
    if 'CXX' not in os.environ and maps.is_file():
        system_runtime = any('/usr/lib/' in line and 'libstdc++.so' in line for line in maps.read_text().splitlines())
        if system_runtime and Path('/usr/bin/g++').is_file():
            os.environ['CXX'] = '/usr/bin/g++'
            os.environ.setdefault('CC', '/usr/bin/g++')
    libraries = []
    if nccl:
        package = Path(torch.__file__).resolve().parent.parent / 'nvidia/nccl'
        assert (package / 'include/nccl.h').is_file() and (package / 'lib/libnccl.so.2').is_file()
        includes.append(package / 'include')
        libraries = [f'-L{package / "lib"}', '-l:libnccl.so.2', f'-Wl,-rpath,{package / "lib"}']
    selected = build_directory or os.environ.get('BASISKV_EXT_BUILD_DIR')
    directory = Path(selected).expanduser().resolve() / name if selected else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    verbose = os.environ.get('BASISKV_VERBOSE_BUILD', '0') == '1' if verbose is None else verbose
    lock = Path(tempfile.gettempdir()) / f'{name}.{os.getuid()}.lock'
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return cpp_extension.load(name=f'{name}_sm{major}{minor}', sources=list(map(str, sources)),
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-std=c++17', '--threads', '4', *cuda_flags],
            extra_include_paths=list(map(str, includes)), extra_ldflags=libraries,
            with_cuda=True, build_directory=str(directory) if directory is not None else None, verbose=verbose)
