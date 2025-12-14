# **Final Project**

This project seeks to incorporate what has been discussed in the course by implementing various versions of a 2D stencil operation and matrix multiplication algorithm. In all versions, two input square matrices, `A` and `B`, of size `N` and initialized so that all of their elements are 1. After initialization, a 2D stencil operation with radius `RAD`[^1] is applied to both matrices. The stenciled version of both matrices are then multiplied together. The results from these computations are validated using a pair of utility functions defined in [utils.h](./hh/utils.h). More information about how these functions work is provided in the Appendix. In accordance with the project instructions, we used `DSIZE =  N = 1024` and `RAD = 3`.

All of the implementations are profiled using appropriate tools for their implementation. For the CPU version, we use `vtune`, while for the CUDA and the Alpaka versions of the code we use `nsys`. This is done through the `profile` rule in [`Makefile`](./Makefile), which stores the output of these profilings in corresponding `.csv` files which can be found in the [`profiling/`](./profiling/) directory.

## C++ and CPU Profiling
> - Start by writing a code in C++ that :
>   - Creates two 2-dimensional square matrices A and B of size DSIZE >= 512 and fill them in with arbitrary integer values.
>   - Performs a 2-d stencil operation on each matrix. You can use any radius size, but keep it > 2.
>   - Performs a matrix multiplication of the matrices after the stencil application
>   - Make sure that you also add utility functions to check your results. 
> - Profile your C++ code using the VTune profiler and identify the compute intensive parts.

The CPU implementation of the matrix stencil and multiplication C++ script can be found [here](./ex1-cpu/stencil_mult.cpp). This serial code was profiled using `vtune`, the resutls of which can be found in [profile_cpu_hostots.csv](./profiling/profile_cpu_hotspots.csv) and in [profile_cpu_summary.csv](./profiling/profile_cpu_summary.csv), which show that the total runtime was of 4.9 seconds. From the results displayed in this file, it is clear that `matrix_mult` was the slowest and most computationally costly part of the algorithm. In fact, most of the time spent running this implementation is spent running this function, which takes up ~99.0% of the computing time. The second slowest part of this implementation was the stencil computation, but that took a comparatively miniscule 0.6% of the total runtime. Moreover, looking at the effective CPU utilization histogram available in the VTune GUI interphase, we can see that, as expected for this fully serial baseline implementation, the average effective CPU utilization is very low.

![Effective CPU Utilization Histogram](./assets/effcpuutil.png)

## Porting to CUDA

> - Write the same application in CUDA: 
>   - You should write a CUDA kernel that performs the stencil operation and one for the matrix multiplication.
>   - Initially make use of explicit memory copies from host to device and vice-versa and make use only of the default CUDA stream.
>   - Make sure to add utility functions for error checking and for verifying your results.
> - Profile your code using nsys and document/comment on the time spent in each CUDA API call. Also, make note on the time spent on host and device.
> - Try switching from explicit memory copies to managed memory. 
>    - Profile again using either nsys on ncu and comment on the performance of your application. 

Given that the two implementations made for the CUDA port of the algorithm were only meant to differ by how the memory was handled (i.e. explicit copies vs. managed memory), the `compute_stencil` and `matrix_mult` kernels were implemented in a separate header file, namely [compute_funcs_ex2.h](./hh/compute_funcs_ex2.h) which was included in both implementations. These implementations are relatively simple, as they don't use any shared memory, and only the default CUDA stream is used when they are called. 

Profiling for these implementations was performed using `nsys`, as can be seen in the `profiling` rule in [Makefile](./Makefile), and the results were stored as `.csv` files. Overall, the version using explicit memory copies took ~336.759 ms (14.6x faster than CPU version), while the one using managed memory took ~399.310 ms (12.3x faster than CPU version). When observing the results in the GPU Kernel Sum report for the two versions, we see that, while there is still a major difference in how much time the matrix multiplication computation took when compared to the stencil operation, ~90-99% and ~1-9% respectively, their compute time is vastly smaller than their CPU-based counterparts profiled before. In fact, for the CUDA version with explicit memory copies (which was the fastest out of the two of these CUDA implementations), the matrix multiplication has a speedup of ~67.2x, while the stencil operation has a speedup of ~39.4 when compared to the CPU version. 

An important thing to note is that version of the performance CUDA implementation which uses explicit memory copies (henceforth referred to as the "explicit" version) differs with respect to the implementation that uses managed memory (the "managed" version). This is particularly noticeable when comparing the time it took for the stencil operation to finish. For the explicit version, the time was ~380,423.0 ns per stencil kernel call, while fort he managed, it was significantly larger as ~3,774,567.5 ns. Given that the only differnce between both version is the memory handling, this nearly order-of-magnitude difference is a result of the used of Unified (managed) Memory, which introduced implicit page migration when the GPU first accesses memory that was initialized on the host. The associated overhead due to page migration and page faults dominates kernel execution time. On the other hand, in the explicit version, the use of preemptive host-to-memory copies ensures that all of the required data is already in the GPU before the kernel execution takes place.

The overhead of using managed memory is in the GPU memory time sum profiling reports. For the managed version, this report can be found in [profile_managed_cuda_gpu_mem_time_sum.csv](./profiling/profile_managed_cuda_gpu_mem_time_sum.csv), while for the explicit version it is [profile_explicit_cuda_gpu_mem_time_sum.csv](./profiling/explicit_managed_cuda_gpu_mem_time_sum.csv). In these reports, we can see that the amount of host-to-device memory copies is vastly larger for the managed version at 696 when compared to the explicit version, which only has 3 host-to-device copy operations. We can also see that the amount of copies in the opposite direction are also significatly more abundant in the managed version of the code. Overall, we can conclude that the use of Unifier Memory, while syntactically convenient, introduces overhead.

## Optimizing performance in CUDA

> - Optimize the performance of your code making use of non-default CUDA streams and shared memory. 
> - Once you have decided on the best approach, profile your application and compare the time spent in each API call and the overall timing of your application with your initial CUDA implementation.

This optimized version of the CUDA implementation incorporates three main changes aimed at improving performance: Firstly, because certain operations such as the stencil, or the memory copies are independent, they are allowed to run concurrently by introducing CUDA streams. In addition, we use shared memory in the kernels, which allows for the reduction of global to local memory transfers during their execution. Finally, having observed, the significant overhead introduced by using managed memory, we fall back to using explicit memory copies.

These changes made a significant impact in performance, as the overal time for the code to run was ~232.857 ms accoridng to the NVIDIA Nsight application. This represents a speed up of ~21x over the baseline CPU version of the algorithm. The CUDA API sum report illustrates how the use of asynchronous memory copies and non-default CUDA streams significantly helped improve performance by optimizing data transfers. In this version of the code, the time spent on memory copies (`cudaMemcpyAsync`) is ~6.85 ms, while for the previous explicit version of the code ~76.5 ms were spent on `cudaMemcpy` calls. This reduction is achieved by overlapping memory transfers with kernel execution using multiple streams.

The use of shared memory in the optimized version offered additional performance boosts by reducing the global memory traffic. Kernel profiling shows that the matrix multiplication runtime decreased from ~72.2 ms in the explicit version to ~5.13 ms in this optimized version. This improvement can be attributed to high data reused in matrix multiplication. The stencil kernel also showed significant improvement, with its average execution time reduced from ~380 $\mu$s to ~250 $\mu$s. This is due to the reduction of redundant global memory accesses.

## Making use of Alpaka

> - Re-write your application making use of the Alpaka portability library.
> - Describe the steps you had to follow to re-write your code.

Because the underlying algorithm in the Alpaka version of the code is the same as in all of the other implementations, we began by outlining the code based on these other cases. The main components in the outline consisted of buffer allocations, memory copies and initializations, application of the stencil and matrix multiplication kernels to the corresponding matrices, a final copy to obtain the results computed by the device, and a check of the results. The tools for some of these steps were already implemented for other versions in [`utils.h`](./hh/utils.h), and could thus be reused. For instance, for the host side initializations, we simply used the `init_matrix` function that was used previously. Another example was the results validation functions `check_mult` and `check_stencil`. They did not require adaptation to the code itself, as all that was needed to be used was the pointer to the inputs, which were obtained using `alpaka::getPtrNative()`.

Once the code was outlined, the next step consisted of referring back to the instructional material and Alpaka documentation to understand which functions and classes I would need in each step of the algorithm. I followed examples provided in the Alpaka lecture slides, making any appropriate modifications where neccessary. This process was iterative, and consisted of repetitive testing and debugging as the script was being developed. Moreover, for syntactic simplicity, we made extensive use of the tools provided in the [`config.h`](./hh/config.h) and [`WorkDiv.hpp](./hh/WorkDiv.hpp) header files.

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

Note that the `hh` directory contained a collection of header files with utilities used by the different version of the implemented algorithm. Of particular note are [`config.h`](./hh/config.h) and [`WorkDiv.hpp`](./hh/WorkDiv.hpp), which contain helpful utilities for the Alpaka version of the code, and which were fetched from [this](https://github.com/fwyzard/intro_to_alpaka/tree/master/alpaka) repository that was part of the instructional material for the Alpaka lecture. 

### Results Validation

The functions implemented to validate the results from the stencil and matrix multiplication operations can be found in [utils.h](./hh/utils.h). Because we did not want these to dominate computing time, they are purposely simple, making use of tha fact that the matrices `A` and `B` are always initialized with all values equal to 1. For the stencil check function, we compare what the separate sum of the inner, edge and corner set of elements in each stenciled plot should be given the `RAD`, `N` and that all elements are initially 1. For the multiplication check, we simply make sure that all elements in the inner part of the matrix `C = A_stn * B_stn` are the same and equal to the predicted value.


---
[^1]: `N` and `RAD` are specified in [utils.h](./hh/utils.h), which is included in all versions of the algorithm so that they all share these same values.