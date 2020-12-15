#include "MWKernelHeaders.hpp"
#include <math.h>
#include <stdio.h>

void __global__ __launch_bounds__(1024) scale_scalar_kernel(float *
inputBuffer, float *outputBuffer, float *pvpNsgGssdTxeVoFIkXI, long int
                                                            YNDVziqpDddiXQKYZZhX) {
    for (long int idx = blockDim.x * blockIdx.x +
                        threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx += blockDim.x * gridDim.x) {
        outputBuffer[idx] = pvpNsgGssdTxeVoFIkXI[0] * inputBuffer[idx];
    }
}

void __global__
__launch_bounds__(1024) scale_vector_kernel(float *inputBuffer, float *
outputBuffer, float *pvpNsgGssdTxeVoFIkXI, double YOWMnLKOMqAODXiVNoGy,
                                            double YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX) {
    for
            (long int idx = blockDim.x * blockIdx.x + threadIdx.x; idx <
                                                                   YNDVziqpDddiXQKYZZhX; idx += blockDim.x *
                                                                                                gridDim.x) {
        double batchIdx =
                floor(idx / YNmJhGSUszJKxsodxiuV);
        double i_batch = idx - (batchIdx *
                                YNmJhGSUszJKxsodxiuV);
        double channelIdx = floor(i_batch /
                                  YOWMnLKOMqAODXiVNoGy);
        outputBuffer[idx] =
                pvpNsgGssdTxeVoFIkXI[static_cast<long int>(channelIdx)] * inputBuffer[idx];
    }
}

void
__global__ __launch_bounds__(1024) scale_matrix2d_kernel(float *inputBuffer,
                                                         float *outputBuffer, float *pvpNsgGssdTxeVoFIkXI, double
                                                         YOWMnLKOMqAODXiVNoGy, long int YNDVziqpDddiXQKYZZhX) {
    for (long int
                 idx = blockDim.x * blockIdx.x + threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx +=
                                                                                                  blockDim.x *
                                                                                                  gridDim.x) {
        double totalChannelIdx = floor(idx /
                                       YOWMnLKOMqAODXiVNoGy);
        double i_channel = idx - (totalChannelIdx *
                                  YOWMnLKOMqAODXiVNoGy);
        outputBuffer[idx] =
                pvpNsgGssdTxeVoFIkXI[static_cast<long int>(i_channel)] * inputBuffer[idx];
    }
}

void
__global__ __launch_bounds__(1024) scale_tensor3d_kernel(float *inputBuffer,
                                                         float *outputBuffer, float *pvpNsgGssdTxeVoFIkXI, double
                                                         YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX) {
    for (long int
                 idx = blockDim.x * blockIdx.x + threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx +=
                                                                                                  blockDim.x *
                                                                                                  gridDim.x) {
        double batchIdx = floor(idx /
                                YNmJhGSUszJKxsodxiuV);
        double i_batch = idx - (batchIdx *
                                YNmJhGSUszJKxsodxiuV);
        outputBuffer[idx] =
                pvpNsgGssdTxeVoFIkXI[static_cast<long int>(i_batch)] * inputBuffer[idx];
    }
}

void
__global__ __launch_bounds__(1024) offset_scalar_kernel(float *inputBuffer,
                                                        float *outputBuffer, float *gNROjwaqhxDPvBWUCUcQ,
                                                        long int YNDVziqpDddiXQKYZZhX,
                                                        bool ZUTPCvgISoRdtnhGqXzM, int bQjijJlpNAVdwDDQgpaX, int
                                                        veFyKKHbdqBIvQLYBqfF) {
    for (long int idx = blockDim.x * blockIdx.x +
                        threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx += blockDim.x * gridDim.x) {
        float
                out = inputBuffer[idx] + gNROjwaqhxDPvBWUCUcQ[0];
        if (ZUTPCvgISoRdtnhGqXzM) {
            out =
                    out > veFyKKHbdqBIvQLYBqfF ? veFyKKHbdqBIvQLYBqfF : out;
            out = out <
                  bQjijJlpNAVdwDDQgpaX ? bQjijJlpNAVdwDDQgpaX : out;
        }
        outputBuffer[idx] = out;
    }
}

void __global__ __launch_bounds__(1024) offset_vector_kernel(float *
inputBuffer, float *outputBuffer, float *gNROjwaqhxDPvBWUCUcQ, double
                                                             YOWMnLKOMqAODXiVNoGy, double YNmJhGSUszJKxsodxiuV, long int
                                                             YNDVziqpDddiXQKYZZhX, bool ZUTPCvgISoRdtnhGqXzM,
                                                             int bQjijJlpNAVdwDDQgpaX, int
                                                             veFyKKHbdqBIvQLYBqfF) {
    for (long int idx = blockDim.x * blockIdx.x +
                        threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx += blockDim.x * gridDim.x) {
        double batchIdx = floor(idx / YNmJhGSUszJKxsodxiuV);
        double i_batch =
                idx - (batchIdx * YNmJhGSUszJKxsodxiuV);
        double channelIdx =
                floor(i_batch / YOWMnLKOMqAODXiVNoGy);
        float out = inputBuffer[idx] +
                    gNROjwaqhxDPvBWUCUcQ[static_cast<long int>(channelIdx)];
        if
                (ZUTPCvgISoRdtnhGqXzM) {
            out = out > veFyKKHbdqBIvQLYBqfF ?
                  veFyKKHbdqBIvQLYBqfF : out;
            out = out < bQjijJlpNAVdwDDQgpaX ?
                  bQjijJlpNAVdwDDQgpaX : out;
        }
        outputBuffer[idx] = out;
    }
}

void __global__
__launch_bounds__(1024) offset_matrix2d_kernel(float *inputBuffer, float *
outputBuffer, float *gNROjwaqhxDPvBWUCUcQ, double YOWMnLKOMqAODXiVNoGy,
                                               long int YNDVziqpDddiXQKYZZhX, bool ZUTPCvgISoRdtnhGqXzM, int
                                               bQjijJlpNAVdwDDQgpaX, int veFyKKHbdqBIvQLYBqfF) {
    for (long int idx =
            blockDim.x * blockIdx.x + threadIdx.x; idx < YNDVziqpDddiXQKYZZhX; idx +=
                                                                                       blockDim.x * gridDim.x) {
        double totalChannelIdx = floor(idx /
                                       YOWMnLKOMqAODXiVNoGy);
        double i_channel = idx - (totalChannelIdx *
                                  YOWMnLKOMqAODXiVNoGy);
        float out = inputBuffer[idx] +
                    gNROjwaqhxDPvBWUCUcQ[static_cast<long int>(i_channel)];
        if (ZUTPCvgISoRdtnhGqXzM) {
            out = out > veFyKKHbdqBIvQLYBqfF ? veFyKKHbdqBIvQLYBqfF : out;
            out = out <
                  bQjijJlpNAVdwDDQgpaX ? bQjijJlpNAVdwDDQgpaX : out;
        }
        outputBuffer[idx] = out;
    }
}

void __global__ __launch_bounds__(1024) offset_tensor3d_kernel(float *
inputBuffer, float *outputBuffer, float *gNROjwaqhxDPvBWUCUcQ, double
                                                               YNmJhGSUszJKxsodxiuV, long int YNDVziqpDddiXQKYZZhX, bool
                                                               ZUTPCvgISoRdtnhGqXzM, int bQjijJlpNAVdwDDQgpaX,
                                                               int veFyKKHbdqBIvQLYBqfF) {
    for (long int idx = blockDim.x * blockIdx.x + threadIdx.x; idx <
                                                               YNDVziqpDddiXQKYZZhX; idx += blockDim.x * gridDim.x) {
        double batchIdx =
                floor(idx / YNmJhGSUszJKxsodxiuV);
        double i_batch = idx - (batchIdx *
                                YNmJhGSUszJKxsodxiuV);
        float out = inputBuffer[idx] +
                    gNROjwaqhxDPvBWUCUcQ[static_cast<long int>(i_batch)];
        if (ZUTPCvgISoRdtnhGqXzM) {
            out = out > veFyKKHbdqBIvQLYBqfF ? veFyKKHbdqBIvQLYBqfF : out;
            out = out <
                  bQjijJlpNAVdwDDQgpaX ? bQjijJlpNAVdwDDQgpaX : out;
        }
        outputBuffer[idx] = out;
    }
}