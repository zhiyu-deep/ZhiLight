#include "bmengine/core/core.h"
#include "bmengine/core/thread_pool.h"
#include "bmengine/functions/all.h"
#include <thread>
#include <iostream>
#include <cuda_fp16.h>

using bmengine::core::TaskThreadPool;

int main() {
    TaskThreadPool pool(2);
    for (int i = 0; i < 10; ++i) {
        pool.run([i] { std::cout << "task: " << i << " done" << std::endl; });
    }
    pool.wait();

    std::vector<bmengine::core::DeviceConfiguration> devices;
    int gpu_num;
    BM_CUDART_ASSERT(cudaGetDeviceCount(&gpu_num));
    for (int i = 0; i < gpu_num; ++i) {
        devices.emplace_back(i, 1ll * 1024 * 1024 * 1024);
    }
    std::cout << "all done" << std::endl;
    bmengine::core::Engine engine(devices);
    engine.device_foreach([](int i) { std::cout << "task: " << i << " done" << std::endl; });
}