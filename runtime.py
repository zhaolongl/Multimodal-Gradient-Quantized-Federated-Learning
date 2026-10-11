"""Process-local single OpenMP runtime loader; installed files are untouched."""
from pathlib import Path
import ctypes
import glob
import os
import runpy
import sys

runtime = Path(sys.prefix) / 'Library' / 'bin' / 'libiomp5md.dll'
torch_dlls = str(Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib' / '*.dll')
runtime_handle = ctypes.WinDLL(str(runtime))
original_glob = glob.glob

def single_runtime_glob(pattern, *args, **kwargs):
    paths = original_glob(pattern, *args, **kwargs)
    if os.path.normcase(str(pattern)) == os.path.normcase(torch_dlls):
        paths = [p for p in paths if Path(p).name.lower() != 'libiomp5md.dll']
    return paths

glob.glob = single_runtime_glob
try:
    import torch
finally:
    glob.glob = original_glob
import numpy as np
import sklearn
import psutil
loaded = sorted({m.path for m in psutil.Process().memory_maps() if 'libiomp5md.dll' in m.path.lower()})
if len(loaded) != 1:
    raise RuntimeError(f'Expected exactly one OpenMP runtime, got {loaded}')
print('Single OpenMP runtime:', loaded, flush=True)
print('Versions:', torch.__version__, np.__version__, sklearn.__version__, flush=True)

if len(sys.argv) > 1:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name='__main__')
else:
    # Check that both numerical libraries execute using the shared runtime.
    a = np.array([[2., 1.], [1., 3.]], dtype=np.float64)
    np.testing.assert_allclose(np.linalg.solve(a, np.array([3., 4.])), [1., 1.])
    tensor = torch.tensor(a, requires_grad=True)
    (tensor @ torch.ones(2, dtype=torch.float64)).sum().backward()
    np.testing.assert_allclose(tensor.grad.numpy(), np.ones((2, 2)))
    print('NumPy linear solve and PyTorch autograd checks passed.', flush=True)
