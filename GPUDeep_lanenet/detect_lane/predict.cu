//
// File: predict.cu
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//

// Include Files
#include "predict.h"
#include "DeepLearningNetwork.h"
#include "detect_lane_internal_types.h"
#include "MWCudaDimUtility.hpp"
#include "cnn_api.hpp"

// Type Definitions
struct cell_wrap_6 {
    float f1[6];
};

struct cell_wrap_3 {
    float f1[154587];
};

// Function Declarations
static __global__ void DeepLearningNetwork_predict_kernel30(const float
                                                            varargin_1[154587], cell_wrap_3 miniBatchT[1]);

static __global__ void DeepLearningNetwork_predict_kernel31(const cell_wrap_6
                                                            outputsMiniBatch[1], float varargout_1[6]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float varargin_1[154587]
//                cell_wrap_3 miniBatchT[1]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void
DeepLearningNetwork_predict_kernel30(const float varargin_1[154587],
                                     cell_wrap_3 miniBatchT[1]) {
    unsigned long threadId;
    int i;
    int i1;
    int p;
    threadId = mwGetGlobalThreadIndex();
    i = static_cast<int>(threadId % 227UL);
    threadId = (threadId - static_cast<unsigned long>(i)) / 227UL;
    i1 = static_cast<int>(threadId % 227UL);
    threadId = (threadId - static_cast<unsigned long>(i1)) / 227UL;
    p = static_cast<int>(threadId);
    if ((static_cast<int>((static_cast<int>(p < 3)) && (static_cast<int>(i1 < 227))))
        && (static_cast<int>(i < 227))) {
        miniBatchT[0].f1[(i + 227 * i1) + 51529 * p] = varargin_1[(i1 + 227 * i) +
                                                                  51529 * p];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const cell_wrap_6 outputsMiniBatch[1]
//                float varargout_1[6]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void
DeepLearningNetwork_predict_kernel31(const cell_wrap_6 outputsMiniBatch[1],
                                     float varargout_1[6]) {
    int i;
    i = static_cast<int>(mwGetGlobalThreadIndex());
    if (i < 6) {
        varargout_1[i] = outputsMiniBatch[0].f1[i];
    }
}

//
// Arguments    : lanenet0_0 *obj
//                const float varargin_1[154587]
//                float varargout_1[6]
// Return Type  : void
//
namespace coder {
    void DeepLearningNetwork_predict(lanenet0_0 *obj, const float varargin_1
    [154587], float varargout_1[6]) {
        cell_wrap_3 (*gpu_miniBatchT)[1];
        cell_wrap_6 (*gpu_outputsMiniBatch)[1];
        float (*gpu_varargin_1)[154587];
        float (*gpu_varargout_1)[6];
        cudaMalloc(&gpu_varargout_1, 24UL);
        cudaMalloc(&gpu_outputsMiniBatch, 24UL);
        cudaMalloc(&gpu_miniBatchT, 618348UL);
        cudaMalloc(&gpu_varargin_1, 618348UL);
        cudaMemcpy(gpu_varargin_1, (void *) &varargin_1[0], 618348UL,
                   cudaMemcpyHostToDevice);
        DeepLearningNetwork_predict_kernel30<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>
                (*gpu_varargin_1, *gpu_miniBatchT);
        cudaMemcpy(obj->getInputDataPointer(0), (*gpu_miniBatchT)[0].f1, obj->
                           layers[0]->getOutputTensor(0)->getNumElements() * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        obj->predict();
        cudaMemcpy((*gpu_outputsMiniBatch)[0].f1, obj->getLayerOutput(17, 0),
                   obj->layers[17]->getOutputTensor(0)->getNumElements() * sizeof
                           (float), cudaMemcpyDeviceToDevice);
        DeepLearningNetwork_predict_kernel31<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
                (*gpu_outputsMiniBatch, *gpu_varargout_1);
        cudaMemcpy(&varargout_1[0], gpu_varargout_1, 24UL, cudaMemcpyDeviceToHost);
        cudaFree(*gpu_varargin_1);
        cudaFree(*gpu_miniBatchT);
        cudaFree(*gpu_outputsMiniBatch);
        cudaFree(*gpu_varargout_1);
    }
}

//
// File trailer for predict.cu
//
// [EOF]
//
