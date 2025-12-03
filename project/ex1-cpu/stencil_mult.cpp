#include <stdio.h>
#include <cstdlib>
#include "../hh/utils.h"

using namespace std;

void compute_stencil(
    int *input, 
    int *output, 
    int size, 
    int rad 
) {
    int size_p = size + 2 * rad; // Padded size
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int sum = input[i * size_p + j];
            for (int off_i = -rad; off_i <= rad; off_i++)
                sum += off_i == 0 ? 0 : input[(i + off_i) * size_p + j];
            for (int off_j = -rad; off_j <= rad; off_j++)
                sum += off_j == 0 ? 0 : input[i * size_p + (j + off_j)]; // skip center element
            output[i * size + j] = sum;
        }
    }
}

void matrix_mult(int *A, int *B, int *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

int main(void) {
    
    // Construct two int matrices A and B. Init w/ 1.
    int *A, *B, *C;
    int Np = N + 2*RAD;
    A = (int*) malloc(Np * Np * sizeof(int));
    B = (int*) malloc(Np * Np * sizeof(int));
    C = (int*) malloc(N * N * sizeof(int));

    int *A_stenciled, *B_stenciled;
    A_stenciled = (int*) malloc(N * N * sizeof(int));
    B_stenciled = (int*) malloc(N * N * sizeof(int));
    init_matrix(A, 1, true);
    init_matrix(B, 1, true);
    init_matrix(A_stenciled, 0, false);
    init_matrix(B_stenciled, 0, false);
    init_matrix(C, 0, false);

    // STENCIL
    compute_stencil(
        A + (RAD * Np + RAD), 
        A_stenciled, 
        N, RAD
    );
    // print_mtrx(A_stenciled, N);
    compute_stencil(
        B + (RAD * Np + RAD),
        B_stenciled,
        N, RAD
    );
    // CHECK STENCIL
    if (!check_stencil(A_stenciled)) {
        printf("Stencil computation for A failed!\n");
        return -1;
    }
    if (!check_stencil(B_stenciled)) {
        printf("Stencil computation for B failed!\n");
        return -1;
    }
    printf("Stencil computation passed!\n");
    
    // MATRIX MULTIPLICATION
    matrix_mult(A_stenciled, B_stenciled, C);
    
    // CHECK MULTIPLICATION
    if (!check_mult(A_stenciled, B_stenciled, C)) {
        printf("Matrix multiplication failed!\n");
        return -1;
    }
    printf("Matrix multiplication passed!\n");

    free(A);
    free(B);
    free(C);
    free(A_stenciled);
    free(B_stenciled);

    return 0;
}
