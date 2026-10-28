# Native WSL inference compatibility

This fork retains the upstream MGPC architecture and checkpoints. Tested with Python 3.10.22, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, transformers 4.54.1, numpy 1.26.4, CUDA compiler 12.0 and NVIDIA RTX 4000 Ada Laptop (SM 8.9).

- PointNet++ setup respects TORCH_CUDA_ARCH_LIST (default 8.9), rather than forcing 8.6.
- MGPC_ENCODER_INIT_ONLY=1 constructs DINOv2-base (518-pixel learned positional grid, interpolated for the input) and CLIP ViT-L/14 without downloading redundant backbone weights. Use this ONLY with a full official MGPC checkpoint and strict=True state loading before any inference.
- Without this environment variable, encoder initialization retains upstream behavior.
- No changes to prediction equations or decoder weights.

The complete standalone capture, benchmark and uv setup wrapper lives in WMAM/MGPC alongside this submodule. Model files, environments and generated clouds are excluded from Git. Preserve the upstream MIT license.
