#!/bin/bash
set -e
cd "$(dirname "$0")"

NVCC=/usr/lib/nvidia-cuda-toolkit/bin/nvcc
TORCH_INC=$(python3 -c "import torch; p=torch.utils.cmake_prefix_path; print(p+'/../../include')")
TORCH_LIB=$(python3 -c "import torch; p=torch.utils.cmake_prefix_path; print(p+'/../../lib')")
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
CACHE=~/.cache/torch_extensions/py310_cu128

echo "NVCC: $NVCC"
echo "TORCH_INC: $TORCH_INC"
echo "TORCH_LIB: $TORCH_LIB"
echo "PY_INC: $PY_INC"
echo "CACHE: $CACHE"
echo

for kernel in ggm_kernel fmn_mixture_kernel fmn_mixture_multilevel_kernel; do
    name=${kernel//_kernel/}_cuda
    mkdir -p "$CACHE/$name"
    echo "Building $name from ${kernel}.cu ..."
    $NVCC --allow-unsupported-compiler -ccbin=/usr/bin/g++-11 \
        -gencode=arch=compute_75,code=sm_75 -O3 -std=c++17 \
        --compiler-options '-fPIC' \
        -DTORCH_EXTENSION_NAME=$name -DTORCH_API_INCLUDE_EXTENSION_H \
        -isystem "$TORCH_INC" -isystem "$TORCH_INC/torch/csrc/api/include" \
        -isystem "$PY_INC" --shared \
        -o "$CACHE/$name/$name.so" "${kernel}.cu" \
        -L"$TORCH_LIB" -lc10 -ltorch -ltorch_cpu -ltorch_python
    echo "  → $CACHE/$name/$name.so ($(stat -c%s "$CACHE/$name/$name.so") bytes)"
done

echo
echo "Done. All kernels built."
