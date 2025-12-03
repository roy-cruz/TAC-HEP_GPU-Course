#include <cstdio>
#include <cmath>

#define RAD 3
#define N 64

#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)


void init_matrix(
    int *mat, 
    int value, 
    bool is_padded=false
) {
    int size = is_padded ? (N + 2 * RAD) : N;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            bool is_pad_elem = is_padded && 
                (i < RAD || i >= (size - RAD) || j < RAD || j >= (size - RAD));
            mat[i * size + j] = is_pad_elem ? 0 : value;
        }
    }
}

bool check_stencil(const int *stenciled) {
    // What they should be
    int M = N - 2*RAD; // Size of the sides of the matrix, without counting edge elements
    int center_expected = pow(M, 2) * (1 + 4*RAD);
    
    int edge_expected = (M * RAD);
    int edge_stencil_contrib = 0;
    for (int r = 1; r <= RAD; ++r) {
        edge_stencil_contrib += (4*RAD - r);
    }
    edge_expected = edge_expected + M * edge_stencil_contrib;
    edge_expected *= 4;

    int corner_expected = 3 * pow(RAD, 3); // HOW!?
    corner_expected *= 4;

    // What they are
    int center_computed = 0;
    int edge_computed = 0;
    int corner_computed = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; j++) {
            if (i >= RAD && i < N - RAD && j >= RAD && j < N - RAD) {
                center_computed += stenciled[i * N + j];
            } else if ((i < RAD && j < RAD) || (i < RAD && j >= N - RAD) ||
                       (i >= N - RAD && j < RAD) || (i >= N - RAD && j >= N - RAD)) {
                corner_computed += stenciled[i * N + j];
            } else {    
                edge_computed += stenciled[i * N + j];
            }
        }
    }

    if (center_computed != center_expected) {
        printf("Stencil center sum mismatch: got %d, expected %d\n", center_computed, center_expected);
        return false;
    }

    if (edge_computed != edge_expected) {
        printf("Stencil edge sum mismatch: got %d, expected %d\n",  edge_computed, edge_expected);
        return false;
    }

    if (corner_computed != corner_expected) {
        printf("Stencil corner sum mismatch: got %d, expected %d\n", corner_computed, corner_expected);
        return false;
    }

    return true;
}

bool check_mult(const int *A, const int *B, const int *C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int expected = 0;
            for (int k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            int computed = C[i * N + j];
            if (computed != expected) {
                printf("Matrix multiplication mismatch at [%d,%d]: got %d, expected %d\n", i, j, computed, expected);
                return false;
            }
        }
    }
    return true;
}

void print_mtrx(const int *arr, const int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d, ", arr[i * size + j]); // use size as the stride
        }
        printf("\n");
    }
    printf("\n");
}