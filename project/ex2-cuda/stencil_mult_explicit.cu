#include <stdio.h>
#include <cstdlib>
#include "utils.h"
#include "utils_cuda.h"
#include "compute_funcs_ex2.h"

using namespace std;

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

    // Init
    init_matrix(A, 1, true);
    init_matrix(B, 1, true);
    init_matrix(C, 0, false);
    init_matrix(A_stn, 0, false);
    init_matrix(B_stn, 0, false);
    cudaMemcpy(d_A, A, Np * Np * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Np * Np * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy (h->d) failure");

    dim3 block (BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid (GRID_SIZE, GRID_SIZE);
    
    // STENCIL & MULTIPLY
    compute_stencil<<<grid, block>>>(d_A + RAD * Np + RAD, d_A_stn);
    compute_stencil<<<grid, block>>>(d_B + RAD * Np + RAD, d_B_stn);
    cudaCheckErrors("compute_stencil kernel launch failure");
    matrix_mult<<<grid, block>>>(d_A_stn, d_B_stn, d_C);
    cudaCheckErrors("matrix_mult kernel launch failure");

    // COPY BACK
    cudaMemcpy(A_stn, d_A_stn, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_stn, d_B_stn, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy (d->h) failure");

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

    return 0;
}
