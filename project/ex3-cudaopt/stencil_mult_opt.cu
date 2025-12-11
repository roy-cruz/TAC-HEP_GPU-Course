#include <stdio.h>
#include <cstdlib>
#include "utils.h"
#include "utils_cuda.h"
#include "compute_funcs_ex3.h"

using namespace std;
/*
Main optimizations:
- Shared memory now being used in both stencil and matrix mult
- Explicit memory copies
- Remove unneccesary cuda synchronize since mem copies are blocking
- Using streams to parallelize data copies and stencil kernel executions
*/

int main(void) {
    
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int *A_stn, *B_stn;
    int *d_A_stn, *d_B_stn; // Stenciled versions
    
    // Mem alloc
    int Np = N + 2*RAD; // Padded size
    A = (int*) malloc(Np * Np * sizeof(int));
    B = (int*) malloc(Np * Np * sizeof(int));
    A_stn = (int*) malloc(N * N * sizeof(int));
    B_stn = (int*) malloc(N * N * sizeof(int));
    C = (int*) malloc(N * N * sizeof(int));
    cudaMalloc(&d_A, Np * Np * sizeof(int));
    cudaMalloc(&d_B, Np * Np * sizeof(int));
    cudaMalloc(&d_A_stn, N * N * sizeof(int));
    cudaMalloc(&d_B_stn, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    // Non default stream
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaCheckErrors("cudaStreamCreate failure");

    // Init
    init_matrix(A, 1, true);
    init_matrix(B, 1, true);
    init_matrix(C, 0, false);
    init_matrix(A_stn, 0, false);
    init_matrix(B_stn, 0, false);
    cudaMemcpyAsync(d_A, A, Np * Np * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, B, Np * Np * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_C, C, N * N * sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaDeviceSynchronize();
    cudaCheckErrors("cudaMemcpy (h->d) failure");

    dim3 block (BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid (GRID_SIZE, GRID_SIZE);
    
    // STENCIL & MULT
    compute_stencil<<<grid, block, 0u, stream1>>>(d_A + RAD * Np + RAD, d_A_stn);
    compute_stencil<<<grid, block, 0u, stream2>>>(d_B + RAD * Np + RAD, d_B_stn);
    cudaCheckErrors("compute_stencil kernel launch failure");
    cudaDeviceSynchronize();
    matrix_mult<<<grid, block, 0u, stream3>>>(d_A_stn, d_B_stn, d_C);
    cudaCheckErrors("matrix_mult kernel launch failure");
    
    // COPY BACK
    cudaMemcpyAsync(A_stn, d_A_stn, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(B_stn, d_B_stn, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream3);
    cudaCheckErrors("cudaMemcpy (d->h) failure");
    cudaDeviceSynchronize();
    
    // VAL RESULTS
    bool stencil_ok = check_stencil(A_stn) && check_stencil(B_stn);
    bool mult_ok = check_mult(C);
    if (stencil_ok && mult_ok) {
        printf("Results OK!\n");
    } else {
        printf("Results MISMATCH!\n");
    }

    free(A);
    free(B);
    free(C);
    free(A_stn);
    free(B_stn);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_stn);
    cudaFree(d_B_stn);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}
