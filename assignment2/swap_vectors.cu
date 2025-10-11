#include <stdio.h>

const int DSIZE = 40960; // Size of the vectors
const int block_size = 256;
const int grid_size = (DSIZE + block_size - 1)/block_size;
const int PRINTSIZE = 4;


__global__ void swap_vectors(float *A, float *B, const int n_elems) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_elems){
        float temp = A[idx];
        A[idx] = B[idx];
        B[idx] = temp;
    }
}


__host__ void print_array(const float *arr, const int length) {
    for (int i = 0; i < length; i++) {
        printf("%f, ", arr[i]);
    }
    printf("\n");
}


int main() {

    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE]; 
    h_B = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Print initial arrays
    printf("Some elements of the arrays before swapping\n");
    printf("A = ");
    print_array(h_A, PRINTSIZE); // Print first 4 elements
    printf("B = ");
    print_array(h_B, PRINTSIZE);
    printf("\n==============\n\n");

    cudaMalloc(&d_A, DSIZE*sizeof(float)); // d_A is pointer to a device mem address
    cudaMalloc(&d_B, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    swap_vectors<<<grid_size, block_size>>>(d_A, d_B, DSIZE);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successful
    printf("Some elements of the arrays after swapping\n");
    printf("A = ");
    print_array(h_A, PRINTSIZE); // Print first 4 elements
    printf("B = ");
    print_array(h_B, PRINTSIZE);

    // Free the memory 
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
