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
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        int const* input, int* out, int RAD, int N) const
        {
            ...
        }
};

struct MatrixMultKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        int const* input1, int const* input2, int* output, int N) const
        {
            ...
        }
};

int main() {
    Platform platform; // Device platform
    HostPlatform host_platform; // Host platform
    Host host = alpaka::getDevByIdx(host_platform, 0u); // Get host device from host platform
    Device device = alpaka::getDevByIdx(platform, 0u); // Get device from device platform
    Queue queue{device}; // Create a queue for the device
    Queue host_queue{host}; // Create a queue for the host

    // Mem alloc
    int Np = N + 2*RAD; // Padded size
    auto A = alpaka::allocBuf<int, uint32_t>(host, Np * Np, host_queue);
    auto B = alpaka::allocBuf<int, uint32_t>(host, Np * Np, host_queue);
    auto A_stn = alpaka::allocBuf<int, uint32_t>(host, N * N, host_queue);
    auto B_stn = alpaka::allocBuf<int, uint32_t>(host, N * N, host_queue);
    auto C = alpaka::allocBuf<int, uint32_t>(host, N * N, host_queue);

    auto d_A = alpaka::allocBuf<int, uint32_t>(device, Np * Np, queue);
    auto d_B = alpaka::allocBuf<int, uint32_t>(device, Np * Np, queue);
    auto d_A_stn = alpaka::allocBuf<int, uint32_t>(device, N * N, queue);
    auto d_B_stn = alpaka::allocBuf<int, uint32_t>(device, N * N, queue);
    auto d_C = alpaka::allocBuf<int, uint32_t>(device, N * N, queue);
    
    // Initialize data
    init_matrix(alpaka::getPtrNative(A), 1, true);
    init_matrix(alpaka::getPtrNative(B), 1, true);
    alpaka::memset(host_queue, C, 0x00); // More eff than init_matrix
    alpaka::memset(host_queue, A_stn, 0x00);
    alpaka::memset(host_queue, B_stn, 0x00);
    alpaka::memcpy(queue, d_A, A);
    alpaka::memcpy(queue, d_B, B);
    alpaka::memset(queue, d_C, 0x00);    

    // STENCIL & MULT
    auto grid = makeWorkDiv<Acc2D>(BLOCK_SIZE, GRID_SIZE);
    alpaka::exec<Acc2D>(
        queue, grid, StencilKernel{}, 
        alpaka::getPtrNative(d_A) + RAD * Np + RAD, 
        alpaka::getPtrNative(d_A_stn), 
        RAD, N
    );
    alpaka::exec<Acc2D>(
        queue, grid, StencilKernel{}, 
        alpaka::getPtrNative(d_B) + RAD * Np + RAD, 
        alpaka::getPtrNative(d_B_stn), 
        RAD, N
    );
    alpaka::exec<Acc2D>(
        queue, grid, MatrixMultKernel{}, 
        alpaka::getPtrNative(d_A_stn), alpaka::getPtrNative(d_B_stn), alpaka::getPtrNative(d_C), N
    );
    alpaka::wait(queue); // Ensure all operations are done

    // Copy results back to host
    alpaka::memcpy(host_queue, A_stn, d_A_stn);
    alpaka::memcpy(host_queue, B_stn, d_B_stn);
    alpaka::memcpy(host_queue, C, d_C);
    alpaka::wait(host_queue); // Ensure all operations are done
    
    // Validate results
    bool stencil_ok = check_stencil(alpaka::getPtrNative(A_stn)) && check_stencil(alpaka::getPtrNative(B_stn));
    bool mult_ok = check_mult(alpaka::getPtrNative(A_stn), alpaka::getPtrNative(B_stn), alpaka::getPtrNative(C));
    if (stencil_ok && mult_ok) {
        cout << "Results OK!" << endl;
    } else {
        cout << "Results MISMATCH!" << endl;
    }

    return 0;
}