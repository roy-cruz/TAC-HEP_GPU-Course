#include <stdio.h>
#include <cstdlib>
#include "../hh/utils_cuda.h"
#include "../hh/utils.h"
#include "../hh/compute_funcs_ex3.h"

using namespace std;
/*
Main changes:
- Shared memory now being used in stencil computation (compute_funcs_ex3.h)

*/

int main(void) {
    
    // Instantiate matrices
    int *A, *B, *C;
    int *A_stn, *B_stn;
    int Np = N + 2*RAD; // Padded size
    cudaMallocManaged(&A, Np * Np * sizeof(int));
    cudaMallocManaged(&B, Np * Np * sizeof(int));
    cudaMallocManaged(&C, N * N * sizeof(int));
    cudaMallocManaged(&A_stn, N * N * sizeof(int));
    cudaMallocManaged(&B_stn, N * N * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    // Init
    init_matrix(A, 1, true);
    init_matrix(B, 1, true);
    init_matrix(C, 0, false);
    init_matrix(A_stn, 0, false);
    init_matrix(B_stn, 0, false);
    
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // STENCIL
    compute_stencil<<<grid, block>>>(
        A + RAD * Np + RAD,
        A_stn
    );
    compute_stencil<<<grid, block>>>(
        B + RAD * Np + RAD, 
        B_stn
    );
    cudaCheckErrors("compute_stencil kernel launch failure");
    cudaDeviceSynchronize();
    printf("Stencil A:\n");
    // print_mtrx(A_stn, N);
    
    // CHECK STENCIL
    if (!check_stencil(A_stn)) {
        printf("Stencil computation for A failed!\n");
        return -1;
    }
    if (!check_stencil(B_stn)) {
        printf("Stencil computation for B failed!\n");
        return -1;
    }
    printf("Stencil computation passed!\n");
    
    // MATRIX MULTIPLICATION
    matrix_mult<<<grid, block>>>(A_stn, B_stn, C);
    cudaCheckErrors("matrix_mult kernel launch failure");
    cudaDeviceSynchronize();
    
    // CHECK MULTIPLICATION
    if (!check_mult(A_stn, B_stn, C)) {
        printf("Matrix multiplication failed!\n");
        return -1;
    }
    printf("Matrix multiplication passed!\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(A_stn);
    cudaFree(B_stn);

    return 0;
}
