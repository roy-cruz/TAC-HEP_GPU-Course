# !/bin/bash
# Script compiles and times different implementations of 2D stencil matrix multiplication

echo "Compiling CPU"
g++ ex1-cpu/stencil_mult.cpp -o ./objs/stencil_mult_cpu -I./hh -O3
echo "Compiling CUDA Explicit"
nvcc ex2-cuda/stencil_mult_explicit.cu -o ./objs/stencil_mult_explicit -I./hh -O3
echo "Compiling CUDA Managed"
nvcc ex2-cuda/stencil_mult_managed.cu -o ./objs/stencil_mult_managed -I./hh -O3
echo "Compiling CUDA Optimized"
nvcc ex3-cudaopt/stencil_mult_opt.cu -o ./objs/stencil_mult_opt -I./hh -O3
# echo "Compiling Alpaka"
# 

echo "--------------------------"

# Start timemings
echo "Running CPU"
time ./objs/stencil_mult_cpu
echo "--------------------------"
echo "Running CUDA Explicit"
time ./objs/stencil_mult_explicit
echo "--------------------------"
echo "Running CUDA Managed"
time ./objs/stencil_mult_managed
echo "--------------------------"
echo "Running CUDA Optimized"
time ./objs/stencil_mult_opt
# echo "Running Alpaka"
# ./objs/stencil_mult_alpaka