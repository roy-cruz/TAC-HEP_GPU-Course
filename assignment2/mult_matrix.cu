#include <stdio.h>
#include <time.h>

const int DSIZE = 256;
const int NELEMS = DSIZE*DSIZE;
const float A_val = 3.0f;
const float B_val = 2.0f;

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

   // Square matrix multiplication on CPU : C = A * B
__host__ void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
    //FIXME:
    int summation = 0;
    for (int i=0; i<DSIZE; i++) {
        for (int j=0; j<DSIZE; j++) {
            summation = 0;
            for (int k=0; k<DSIZE; k++) {
                summation += A[i + k * DSIZE] * B[k + j * DSIZE];
            }
            C[i + j * DSIZE] = summation;
        }
    }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {

    //FIXME:
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

    // These are used for timing
    clock_t t_init_start, t_init_end, t_cuda_start, t_cuda_end, t_cudaAll_start, t_cudaAll_end, t_cpu_start, t_cpu_end;
    double t_init=0.0;
    double t_cuda=0.0;
    double t_cudaAll = 0.0;
    double t_cpu=0.0;

    // Initialization timing
    t_init_start = clock();

    // N*N matrices defined in 1 dimention
    h_A = new float[NELEMS];
    h_B = new float[NELEMS];
    h_C = new float[NELEMS];
    for (int i = 0; i < NELEMS; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Initialization timing end
    t_init_end = clock();
    t_init = ((double)(t_init_end-t_init_start))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t_init);

    printf("Matrix A\n");
    print_mtrx(h_A, 10, 10);
    printf("Matrix B\n");
    print_mtrx(h_B, 10, 10);
    printf("Matrix C before multiplication\n");
    print_mtrx(h_C, 10, 10);

    t_cudaAll_start = clock();
    // Allocate device memory and copy input data from host to device
    cudaMalloc(&d_A, NELEMS*sizeof(float));
    cudaMalloc(&d_B, NELEMS*sizeof(float));
    cudaMalloc(&d_C, NELEMS*sizeof(float));
    //FIXME:Add all other allocations and copies from host to device
    cudaMemcpy(d_A, h_A, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, NELEMS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, NELEMS*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // Specify the block and grid dimentions 
    dim3 block(16, 16);  //FIXME
    dim3 grid((DSIZE + block.x - 1)/block.x,
              (DSIZE + block.y - 1)/block.y); //FIXME

    t_cuda_start = clock();
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    t_cuda_end = clock();

    // Copy results back to host
    cudaMemcpy(h_C, d_C, NELEMS*sizeof(float), cudaMemcpyDeviceToHost);

    // GPU timing
    t_cudaAll_end = clock();

    t_cuda = ((double)(t_cuda_end - t_cuda_start))/CLOCKS_PER_SEC;
    t_cudaAll = ((double)(t_cudaAll_end - t_cudaAll_start))/CLOCKS_PER_SEC;

    printf("GPU:\n");
    printf("Time for compute with memory allocation and copy: %f seconds\n", t_cudaAll);
    printf("Time for compute without memory allocation and copy: %f seconds\n", t_cuda);
    printf("Matrix C after multiplication w/ GPU\n");
    print_mtrx(h_C, 10, 10);

    // Re-initialize matrix C to 0
    for (int i = 0; i < NELEMS; i++){
        h_C[i] = 0;
    }

    // CPU timing start
    t_cpu_start = clock();
    // Excecute and time the cpu matrix multiplication function
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);
    // CPU timing
    t_cpu_end = clock();

    t_cpu = ((double)(t_cpu_end-t_cpu_start))/CLOCKS_PER_SEC;
    printf("CPU:\n");
    printf ("Done. Compute took %f seconds\n", t_cpu);
    printf("Matrix C after multiplication w/ CPU\n");
    print_mtrx(h_C, 10, 10);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
