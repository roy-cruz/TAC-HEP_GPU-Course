# **Final Project**

This project seeks to incorporate what has been discussed in the course by implementing various versions of a 2D stencil operation and matrix multiplication algorithm. In all versions, two input square matrices, `A` and `B`, of size `N` and initialized so that all of their elements are 1. After initialization, a 2D stencil operation with radius `RAD`[^1] is applied to both matrices. The stenciled version of both matrices are then multiplied together. The results from these computations are validated using a pair of utility functions defined in [utils.h](./hh/utils.h). More information about how these functions work is provided in the Appendix. In accordance with the project instructions, we used `DSIZE =  N = 1024` and `RAD = 3`.

All of the implementations are profiled using appropriate tools for their implementation. For the CPU version, we use `vtune`, while for the CUDA and the Alpaka versions of the code we use `nsys`. This is done through the `profile` rule in [Makefile](./Makefile), which stores the output of these profilings in corresponding `.csv` files which can be found in the [`profiling/`](./profiling/) directory.

## C++ and CPU Profiling
> - Start by writing a code in C++ that :
>   - Creates two 2-dimensional square matrices A and B of size DSIZE >= 512 and fill them in with arbitrary integer values.
>   - Performs a 2-d stencil operation on each matrix. You can use any radius size, but keep it > 2.
>   - Performs a matrix multiplication of the matrices after the stencil application
>   - Make sure that you also add utility functions to check your results. 
> - Profile your C++ code using the VTune profiler and identify the compute intensive parts.

The CPU implementation of the matrix stencil and multiplication C++ script can be found [here](./ex1-cpu/stencil_mult.cpp). This serial code was profiled using `vtune`, the resutls of which can be found in [profile_cpu.csv](./profiling/profile_cpu.csv). From the results displayed in this file, it is clear that `matrix_mult` composed the slowest and most computationally costly part of the algorithm. In fact, most of the time spent running this implementation is spent running this function, which takes up ~99.4% of the computing time. Moreover, looking at the effective CPU utilization histogram offered available in the VTune GUI interphase, we can see that, as expected for this fully serial baseline implementation, the average effective CPU utilization is very low.

![Effective CPU Utilization Histogram](./assets/effcpuutil.png)

## Porting to CUDA

> - Write the same application in CUDA: 
>   - You should write a CUDA kernel that performs the stencil operation and one for the matrix multiplication.
>   - Initially make use of explicit memory copies from host to device and vice-versa and make use only of the default CUDA stream.
>   - Make sure to add utility functions for error checking and for verifying your results.
> - Profile your code using nsys and document/comment on the time spent in each CUDA API call. Also, make note on the time spent on host and device.
> - Try switching from explicit memory copies to managed memory. 
>    - Profile again using either nsys on ncu and comment on the performance of your application. 

Given that the two implementations made for the CUDA port of the algorithm were only meant to differ by how the memory was handled (i.e. explicit copies vs. managed memory), the `compute_stencil` and `matrix_mult` kernels were implemented in a separate header file, namely [compute_funcs_ex2.h](./hh/compute_funcs_ex2.h) which was included in both implementations. These implementations are relatively simple, as they don't use any shared memory, and only the default CUDA stream is used when they were called.

Profiling was performed using `nsys`, as can be seen in the `profile` rule in [Makefile](./Makefile). The results were


## Optimizing performance in CUDA

> - Optimize the performance of your code making use of non-default CUDA streams and shared memory. 
> - Once you have decided on the best approach, profile your application and compare the time spent in each API call and the overall timing of your application with your initial CUDA implementation.

## Making use of Alpaka

> - Re-write your application making use of the Alpaka portability library.
> - Describe the steps you had to follow to re-write your code.


## Appendix
### Setup
This project was developed and run on a GPU node on the Wisconsin Analysis Facility. In this computing system, setting up CUDA for compilation consisted of running

```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib
```

To compile the implementation of the code that uses Alpaka, we followed the setup instructions provided in the Alpaka lecture, which consisted of simply cloning the official Alpaka repo by running the following command.

```bash
git clone https://github.com/alpaka-group/alpaka.git -b 2.0.0 ${HOME}/public/alpaka
```

In order conveniently re-run the compilation of the code during development, a Make file was constructed. In this file, the compilation of the different versions of the matrix stenciling and multiplication code were compiled using the following commands.

```bash
g++ ex1-cpu/stencil_mult.cpp -o objs/stencil_mult_cpu -O3 -g -I ./hh
nvcc ex2-cuda/stencil_mult_explicit.cu -o objs/stencil_mult_explicit -O3 -I ./hh
nvcc ex2-cuda/stencil_mult_managed.cu -o objs/stencil_mult_managed -O3 -I ./hh
nvcc ex3-cudaopt/stencil_mult_opt.cu -o objs/stencil_mult_opt -O3 -I ./hh
nvcc ex4-alpaka/stencil_mult_alpaka.cpp -o objs/stencil_mult_alpaka -x cu -expt-relaxed-constexpr -std=c++20 -O3 -g -I /mnt/ceph/home/rcruzcan/public/alpaka/include -D ALPAKA_ACC_GPU_CUDA_ENABLED -I ./hh -Wno-deprecated-declarations
```

Note that the `hh` directory contained a collection of header files with utilities used by the different version of the implemented algorithm. Of particular note are \texttt{config.h} and \texttt{WorkDiv.hpp}, which contain helpful utilities for the Alpaka version of the code, and which were fetched from [this](https://github.com/fwyzard/intro_to_alpaka/tree/master/alpaka) repository that was part of the instructional material for the Alpaka lecture. 

### Results Validation


---
[^1]: `N` and `RAD` are specified in [utils.h](./hh/utils.h), which is included in all versions of the algorithm so that they all share these same values.