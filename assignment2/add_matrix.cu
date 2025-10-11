#include <stdio.h>

const int DSIZE_X = 256;
const int DSIZE_Y = 256;
const int NELEMS = DSIZE_X * DSIZE_Y;
const int PRINTSIZE = 3;

__global__ void add_matrix(float *A, float *B, float *C, int width, int height) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int mtrx_idx = idx + width * idy;

    if ((idx < width) && (idy < height))
        C[mtrx_idx] =  A[mtrx_idx] + B[mtrx_idx];
}

__host__ void print_mtrx(const float *arr, const int width, const int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%f, ", arr[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main()
{
   
    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    h_A = new float[NELEMS]; 
    h_B = new float[NELEMS];
    h_C = new float[NELEMS];

    for (int i = 0; i < NELEMS; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0.0f;
    }
    
    cudaMalloc(&d_A, NELEMS*sizeof(float));
    cudaMalloc(&d_B, NELEMS*sizeof(float));
    cudaMalloc(&d_C, NELEMS*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, NELEMS*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(16, 16); // 16x16 = 256 threads per block
    dim3 gridSize( // Number of blocks in each dimension
        (DSIZE_X + blockSize.x - 1)/blockSize.x,
        (DSIZE_Y + blockSize.y - 1)/blockSize.y 
    ); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy result back to host 
    cudaMemcpy(h_C, d_C, NELEMS*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure the addition was successful
    printf("A:\n");
    print_mtrx(h_A, PRINTSIZE, PRINTSIZE);
    printf("B:\n");
    print_mtrx(h_B, PRINTSIZE, PRINTSIZE);
    printf("==============\n\n");
    printf("C = A + B:\n");
    print_mtrx(h_C, PRINTSIZE, PRINTSIZE);

    // Free the memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}