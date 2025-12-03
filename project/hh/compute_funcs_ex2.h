__global__ void compute_stencil(
    int *input, 
    int *output
) {
    int size_p = N + 2 * RAD; // Padded size
    int gidx_x = threadIdx.x + blockIdx.x * blockDim.x; // index of row that this thread is working on
    int gidx_y = threadIdx.y + blockIdx.y * blockDim.y; // index of column that this thread is working on
    
    if ((gidx_x < N) && (gidx_y < N)) {
        int linear_index = gidx_x * size_p + gidx_y;
        int rslt = input[linear_index];

        for (int x_offset = -RAD; x_offset <= RAD; x_offset++) {
            int x_idx = gidx_x + x_offset;
            rslt += (x_idx >= 0 && x_idx < N && x_idx != gidx_x) ? input[x_idx * size_p + gidx_y] : 0;
        }
        for (int y_offset = -RAD; y_offset <= RAD; y_offset++) {
            int y_idx = gidx_y + y_offset;
            rslt += (y_idx >= 0 && y_idx < N && y_idx != gidx_y) ? input[gidx_x * size_p + y_idx] : 0;
        }

        output[gidx_x * N + gidx_y] = rslt;
    }
}

__global__ void matrix_mult(int *A, int *B, int *C) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if ((idx < N) && (idy < N)) {
        float temp = 0;
        for (int k = 0; k < N; k++){
            temp += A[idx * N + k] * B[k * N + idy];
        }
        C[idx * N + idy] = temp;
    }
}