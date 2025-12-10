#include <stdio.h>
#include <cstdlib>
#include "utils.h"
#include "utils_cuda.h"
#include "compute_funcs_ex2.h"

using namespace std;

int main(void) {
    
    int *A, *B, *C;
    int *A_stn, *B_stn;
    int Np = N + 2*RAD; // Padded size
    cudaMallocManaged(&A, Np * Np * sizeof(int));
    cudaMallocManaged(&B, Np * Np * sizeof(int));
    cudaMallocManaged(&A_stn, N * N * sizeof(int));
    cudaMallocManaged(&B_stn, N * N * sizeof(int));
    cudaMallocManaged(&C, N * N * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    // Init
    init_matrix(A, 1, true);
    init_matrix(B, 1, true);
    init_matrix(A_stn, 0, false);
    init_matrix(B_stn, 0, false);
    init_matrix(C, 0, false);
    
    dim3 grid (GRID_SIZE, GRID_SIZE);
    dim3 block (BLOCK_SIZE, BLOCK_SIZE);
    
    // STENCIL & MULTIPLY
    compute_stencil<<<grid, block>>>(A + RAD * Np + RAD, A_stn);
    compute_stencil<<<grid, block>>>(B + RAD * Np + RAD, B_stn);
    cudaCheckErrors("compute_stencil kernel launch failure");
    matrix_mult<<<grid, block>>>(A_stn, B_stn, C);
    cudaCheckErrors("matrix_mult kernel launch failure");
    cudaDeviceSynchronize(); // Wait until finished
    
    // VAL RESULTS
    bool stencil_ok = check_stencil(A_stn) && check_stencil(B_stn);
    bool mult_ok = check_mult(C);
    if (stencil_ok && mult_ok) {
        printf("Results OK!\n");
    } else {
        printf("Results MISMATCH!\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(A_stn);
    cudaFree(B_stn);

    return 0;
}
