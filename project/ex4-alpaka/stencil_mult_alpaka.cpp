#include <cstdlib>
#include <iostream>

#include <alpaka/alpaka.hpp>
#include "config.h"
#include "WorkDiv.hpp"
#include "utils.h"
#include "utils_cuda.h"

using namespace std;

struct StencilKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
        int const* input, int* output, int rad, int size
    ) const {
        for (auto ndindex : alpaka::uniformElementsND(acc, {size, size})) {
            int i = ndindex[0];
            int j = ndindex[1];
            if (i < size && j < size) {
                int size_p = size + 2 * rad; // Padded size
                int linear_index = i * size_p + j;
                int rslt = input[linear_index];

                for (int x_offset = -rad; x_offset <= rad; x_offset++) {
                    int x_idx = i + x_offset;
                    rslt += (x_idx >= 0 && x_idx < size && x_idx != i) ? input[x_idx * size_p + j] : 0;
                }
                for (int y_offset = -rad; y_offset <= rad; y_offset++) {
                    int y_idx = j + y_offset;
                    rslt += (y_idx >= 0 && y_idx < size && y_idx != j) ? input[i * size_p + y_idx] : 0;
                }

                output[i * size + j] = rslt;
            }
        }
    }
};

struct MatrixMultKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        int const* input1, int const* input2, int* output, int size) const
        {
        for (auto ndindex : alpaka::uniformElementsND(acc, {size, size})) {
            int i = ndindex[0];
            int j = ndindex[1];
            if (i < size && j < size) {
                float temp = 0;
                for (int k = 0; k < size; k++){
                    temp += input1[i * size + k] * input2[k * size + j];
                }
                output[i * size + j] = temp;
            }
        }
    }
};

int main() {

    Host host = alpaka::getDevByIdx(HostPlatform{}, 0u);
    Device device = alpaka::getDevByIdx(Platform{}, 0u);
    Queue queue{device};

    // Mem alloc
    int Np = N + 2*RAD;
    Idx size = N * N;
    Idx size_p = Np * Np;
    auto A = alpaka::allocBuf<int, Idx>(host, size_p);
    auto B = alpaka::allocBuf<int, Idx>(host, size_p);
    auto A_stn = alpaka::allocBuf<int, Idx>(host, size);
    auto B_stn = alpaka::allocBuf<int, Idx>(host, size);
    auto C = alpaka::allocBuf<int, Idx>(host, size);
    auto d_A = alpaka::allocBuf<int, Idx>(device, size_p);
    auto d_B = alpaka::allocBuf<int, Idx>(device, size_p);
    auto d_A_stn = alpaka::allocBuf<int, Idx>(device, size);
    auto d_B_stn = alpaka::allocBuf<int, Idx>(device, size);
    auto d_C = alpaka::allocBuf<int, Idx>(device, size);
    
    // Initialize data
    init_matrix(alpaka::getPtrNative(A), 1, true);
    init_matrix(alpaka::getPtrNative(B), 1, true);
    init_matrix(alpaka::getPtrNative(A_stn), 0, false);
    init_matrix(alpaka::getPtrNative(B_stn), 0, false);
    init_matrix(alpaka::getPtrNative(C), 0, false);
    alpaka::memcpy(queue, d_A, A);
    alpaka::memcpy(queue, d_B, B);
    alpaka::memset(queue, d_C, 0x00);    

    // STENCIL & MULT
    auto threads = Vec2D{BLOCK_SIZE, BLOCK_SIZE};
    auto blocks = Vec2D{GRID_SIZE, GRID_SIZE};

    auto workdiv = makeWorkDiv<Acc2D>(threads, blocks);
    alpaka::exec<Acc2D>(
        queue, workdiv, StencilKernel{}, 
        alpaka::getPtrNative(d_A) + RAD * Np + RAD, 
        alpaka::getPtrNative(d_A_stn), 
        RAD, N
    );
    alpaka::exec<Acc2D>(
        queue, workdiv, StencilKernel{}, 
        alpaka::getPtrNative(d_B) + RAD * Np + RAD, 
        alpaka::getPtrNative(d_B_stn), 
        RAD, N
    );
    alpaka::exec<Acc2D>(
        queue, workdiv, MatrixMultKernel{}, 
        alpaka::getPtrNative(d_A_stn), alpaka::getPtrNative(d_B_stn), alpaka::getPtrNative(d_C), N
    );
    alpaka::wait(queue);

    // Copy results back to host
    alpaka::memcpy(queue, A_stn, d_A_stn);
    alpaka::memcpy(queue, B_stn, d_B_stn);
    alpaka::memcpy(queue, C, d_C);
    alpaka::wait(queue);
    
    // Validate results
    bool stencil_ok = check_stencil(alpaka::getPtrNative(A_stn)) && check_stencil(alpaka::getPtrNative(B_stn));
    bool mult_ok = check_mult(alpaka::getPtrNative(C));
    if (stencil_ok && mult_ok) {
        cout << "Results OK!" << endl;
    } else {
        cout << "Results MISMATCH!" << endl;
    }

    return 0;
}