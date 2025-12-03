__global__ void compute_stencil(
    int *input, 
    int *output
) {
    
    __shared__ int temp[BLOCK_SIZE + 2 * RAD][BLOCK_SIZE + 2 * RAD];
	// Global index: This is the index in the global memory
    int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
    // Local index: This is the index in the shared memory. It starts at RAD to leave space for the halo
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

    // Apply the stencil
    int result = temp[lindex_x][lindex_y];
    for (int offset = -RAD; offset <= RAD; offset++){
        if (offset != 0){ // To avoid adding the temp[lindex_x][launch_y] element twice
            result += temp[lindex_x + offset][lindex_y];
            result += temp[lindex_x][lindex_y + offset];
        }
    }
    
    // Store the result
    output[gindex_x * N + gindex_y] = result;
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