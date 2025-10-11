#include <stdio.h>
#include <time.h>

const int DSIZE = 256;
const int NELEMS = DSIZE*DSIZE;
const float A_val = 3.0f;
const float B_val = 2.0f;
const int PRINTSIZE = 3;

// error checking macro
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


__host__ void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
    int summation = 0;
    for (int i=0; i<DSIZE; i++) {
        for (int j=0; j<DSIZE; j++) {
            summation = 0;
            for (int k=0; k<DSIZE; k++) {
                summation += A[i + k * DSIZE] * B[k + j * DSIZE];
            }
            C[i * DSIZE + j] = summation;
        }
    }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {
    // create thread x index
    // create thread y index
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // x = j
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // y = i
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float temp = 0;
        for (int k = 0; k < size; k++){
            // temp += A_ik * B_kj;
            temp += A[idy * size + k] * B[k * size + idx];
        }
        //C_ij = A_ik * B_kj
        C[idy * size + idx] = temp;
    }
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


int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Timing variables
    clock_t t_cuda_start, t_cuda_end, t_cpu_start, t_cpu_end;
    double t_cuda=0.0;
    double t_cpu=0.0;

    // N*N matrices defined in 1 dimension
    h_A = new float[NELEMS];
    h_B = new float[NELEMS];
    h_C = new float[NELEMS];
    for (int i = 0; i < NELEMS; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    printf("A (%d x %d):\n", DSIZE, DSIZE);
    print_mtrx(h_A, PRINTSIZE, PRINTSIZE);
    printf("B (%d x %d):\n", DSIZE, DSIZE);
    print_mtrx(h_B, PRINTSIZE, PRINTSIZE);
    printf("==============\n\n");

    cudaMalloc(&d_A, NELEMS*sizeof(float));
    cudaMalloc(&d_B, NELEMS*sizeof(float));
    cudaMalloc(&d_C, NELEMS*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy (h->d) failure");

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(
        (DSIZE + block.x - 1)/block.x,
        (DSIZE + block.y - 1)/block.y
    );

    // GPU timing start (excluding memory allocation and copy)
    t_cuda_start = clock();
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    t_cuda_end = clock();
    cudaCheckErrors("kernel launch failure");

    // Copy results back to host
    cudaMemcpy(h_C, d_C, NELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy (d->h) failure");

    // GPU timing
    t_cuda = ((double)(t_cuda_end - t_cuda_start))/CLOCKS_PER_SEC;

    printf("GPU:\n");
    printf("Time for compute: %f seconds\n", t_cuda);
    printf("GPU Result for C = A * B (%d x %d):\n", DSIZE, DSIZE);
    print_mtrx(h_C, PRINTSIZE, PRINTSIZE);
    printf("==============\n\n");

    // Re-initialize matrix C to 0
    for (int i = 0; i < NELEMS; i++){
        h_C[i] = 0;
    }

    // CPU timing start
    t_cpu_start = clock();
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);
    t_cpu_end = clock();

    t_cpu = ((double)(t_cpu_end-t_cpu_start))/CLOCKS_PER_SEC;
    printf("CPU:\n");
    printf("Time for compute: %f seconds\n", t_cpu);
    printf("CPU Result for C = A * B (%d x %d):\n", DSIZE, DSIZE);
    print_mtrx(h_C, PRINTSIZE, PRINTSIZE);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
