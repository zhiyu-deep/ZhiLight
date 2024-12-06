#pragma once

#include <cuda.h>
void setL2AccessPolicyWindow(cudaStream_t stream, void* data, int window_size, float hitRatio=1.);