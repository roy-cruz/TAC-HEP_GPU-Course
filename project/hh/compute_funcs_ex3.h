__global__ void compute_stencil(
    int *input, 
    int *output
) {
    
    __shared__ int temp[BLOCK_SIZE + 2 * RAD][BLOCK_SIZE + 2 * RAD];
    // Global mem idx
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    // Local mem idx
	int lindex_x = threadIdx.x + RAD;
	int lindex_y = threadIdx.y + RAD;
    
    // Read into shared memory
    int size_p = N + 2 * RAD; // Padded size
    temp[lindex_x][lindex_y] = input[gindex_x * size_p + gindex_y];

    if (threadIdx.x < RAD) {
        temp[lindex_x - RAD][lindex_y] = input[(gindex_x - RAD) * size_p + gindex_y];
        temp[lindex_x + BLOCK_SIZE][lindex_y] = input[(gindex_x + BLOCK_SIZE) * size_p + gindex_y];
    }
    if (threadIdx.y < RAD) {
        temp[lindex_x][lindex_y - RAD] = input[gindex_x * size_p + (gindex_y - RAD)];
        temp[lindex_x][lindex_y + BLOCK_SIZE] = input[gindex_x * size_p + (gindex_y + BLOCK_SIZE)];
    }
    __syncthreads();

    // stencil
    int result = temp[lindex_x][lindex_y];
    for (int offset = -RAD; offset <= RAD; offset++){
        if (offset != 0){ 
            result += temp[lindex_x + offset][lindex_y];
            result += temp[lindex_x][lindex_y + offset];
        }
    }
    output[gindex_x * N + gindex_y] = result;
}

__global__ void matrix_mult(int *A, int *B, int *C) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; // C row
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // C col

    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    int sum = 0; 

    // Loop over tiles along K dimension
    for (int t = 0; t < N; t += BLOCK_SIZE) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)]; 
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}