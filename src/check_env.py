# check_env.py
import torch
import ultralytics
import sys

print("🐍 Python Version:", sys.version)
print("🔥 PyTorch Version:", torch.__version__)
print("🎯 Ultralytics Version:", ultralytics.__version__)

# Check device
if torch.cuda.is_available():
    print("✅ CUDA Available")
    print("🔧 CUDA Version:", torch.version.cuda)
    print("💾 GPU:", torch.cuda.get_device_name(0))
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("🍎 MPS Available (Apple Silicon)")
else:
    print("💻 Using CPU")

# Check ROCm specifically
if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
    print("🟣 ROCm Version:", torch.version.hip)