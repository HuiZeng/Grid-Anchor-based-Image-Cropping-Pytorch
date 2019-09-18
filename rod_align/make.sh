#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o rod_align_kernel.cu.o rod_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
