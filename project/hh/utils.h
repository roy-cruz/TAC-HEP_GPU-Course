#include <cstdio>
#include <cmath>

#define RAD 3
#define N 1024
// for CUDA exercises
#define BLOCK_SIZE 32
#define GRID_SIZE ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)
//

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

    int corner_expected = 3 * pow(RAD, 3);
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

bool check_mult(const int *C) {
    int inner_stenciled = 1 + 4*RAD;
    int inner_mult_expected = (N - 2 * RAD) * inner_stenciled * inner_stenciled;
    for (int k = 1; k <= RAD; ++k) {
        inner_mult_expected += 2 * (inner_stenciled - k) * (inner_stenciled - k);
    }
    bool consistent = true;
    for (int i = RAD; i < N - RAD; ++i) {
        for (int j = RAD; j < N - RAD; ++j) {
                consistent &= (C[i * N + j] == inner_mult_expected);
                if (!consistent) {
                    printf("Matrix mult mismatch at (%d, %d): got %d, expected %d\n", 
                        i, j, C[i * N + j], inner_mult_expected);
                    return false;
            }
        }
    }
    return true;
}

void print_mtrx(const int *arr, const int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d, ", arr[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}