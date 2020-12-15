//
// File: DeepLearningNetwork.cu
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//

// Include Files
#include "DeepLearningNetwork.h"
#include "detect_lane_internal_types.h"
#include "MWElementwiseAffineLayer.hpp"
#include "MWFusedConvReLULayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include <cstdio>

const char *errorString =
        "Abnormal termination due to: %s.\nError in %s (line %d).";

// Function Declarations
static void checkCleanupCudaError(cudaError_t errCode, const char *file,
                                  unsigned int line);

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::allocate() {
    this->targetImpl->allocate(290400, 2);
    for (int idx = 0; idx < 18; idx++) {
        this->layers[idx]->allocate();
    }

    (static_cast<MWTensor<float> *>(this->inputTensors[0]))->setData(this->layers
                                                                     [0]->getLayerOutput(0));
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::cleanup() {
    this->deallocate();
    for (int idx = 0; idx < 18; idx++) {
        this->layers[idx]->cleanup();
    }

    if (this->targetImpl) {
        this->targetImpl->cleanup();
    }
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::deallocate() {
    this->targetImpl->deallocate();
    for (int idx = 0; idx < 18; idx++) {
        this->layers[idx]->deallocate();
    }
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::postsetup() {
    this->targetImpl->postSetup(this->layers, this->numLayers);
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::setSize() {
    for (int idx = 0; idx < 18; idx++) {
        this->layers[idx]->propagateSize();
    }

    this->allocate();
    this->postsetup();
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::setup() {
    this->targetImpl->preSetup();
    this->targetImpl->setAutoTune(true);
    (static_cast<MWInputLayer *>(this->layers[0]))->createInputLayer
            (this->targetImpl, this->inputTensors[0], 227, 227, 3, 0, "", 0);
    (static_cast<MWElementwiseAffineLayer *>(this->layers[1]))
            ->createElementwiseAffineLayer(this->targetImpl, this->layers[0]
                                                   ->getOutputTensor(0), 227, 227, 3, 227, 227, 3, false, 1, 1,
                                           "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_data_scale.bin",
                                           "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_data_offset.bin", 0);
    (static_cast<MWFusedConvReLULayer *>(this->layers[2]))
            ->createFusedConvReLULayer(this->targetImpl, 1, this->layers[1]
                                               ->getOutputTensor(0), 11, 11, 3, 96, 4, 4, 0, 0, 0, 0, 1, 1, 1,
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv1_w.bin",
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv1_b.bin", 1);
    (static_cast<MWNormLayer *>(this->layers[3]))->createNormLayer
            (this->targetImpl, this->layers[2]->getOutputTensor(0), 5, 0.0001, 0.75, 1.0,
             0);
    (static_cast<MWMaxPoolingLayer *>(this->layers[4]))->createMaxPoolingLayer
            (this->targetImpl, this->layers[3]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0,
             0, 0, 1, 1);
    (static_cast<MWFusedConvReLULayer *>(this->layers[5]))
            ->createFusedConvReLULayer(this->targetImpl, 1, this->layers[4]
                                               ->getOutputTensor(0), 5, 5, 48, 128, 1, 1, 2, 2, 2, 2, 1, 1, 2,
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv2_w.bin",
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv2_b.bin", 0);
    (static_cast<MWNormLayer *>(this->layers[6]))->createNormLayer
            (this->targetImpl, this->layers[5]->getOutputTensor(0), 5, 0.0001, 0.75, 1.0,
             1);
    (static_cast<MWMaxPoolingLayer *>(this->layers[7]))->createMaxPoolingLayer
            (this->targetImpl, this->layers[6]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0,
             0, 0, 1, 0);
    (static_cast<MWFusedConvReLULayer *>(this->layers[8]))
            ->createFusedConvReLULayer(this->targetImpl, 1, this->layers[7]
                                               ->getOutputTensor(0), 3, 3, 256, 384, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv3_w.bin",
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv3_b.bin", 1);
    (static_cast<MWFusedConvReLULayer *>(this->layers[9]))
            ->createFusedConvReLULayer(this->targetImpl, 1, this->layers[8]
                                               ->getOutputTensor(0), 3, 3, 192, 192, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv4_w.bin",
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv4_b.bin", 0);
    (static_cast<MWFusedConvReLULayer *>(this->layers[10]))
            ->createFusedConvReLULayer(this->targetImpl, 1, this->layers[9]
                                               ->getOutputTensor(0), 3, 3, 192, 128, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv5_w.bin",
                                       "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv5_b.bin", 1);
    (static_cast<MWMaxPoolingLayer *>(this->layers[11]))->createMaxPoolingLayer
            (this->targetImpl, this->layers[10]->getOutputTensor(0), 3, 3, 2, 2, 0, 0, 0,
             0, 0, 1, 0);
    (static_cast<MWFCLayer *>(this->layers[12]))->createFCLayer(this->targetImpl,
                                                                this->layers[11]->getOutputTensor(0), 9216, 4096,
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fc6_w.bin",
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fc6_b.bin",
                                                                1);
    (static_cast<MWReLULayer *>(this->layers[13]))->createReLULayer
            (this->targetImpl, this->layers[12]->getOutputTensor(0), 1);
    (static_cast<MWFCLayer *>(this->layers[14]))->createFCLayer(this->targetImpl,
                                                                this->layers[13]->getOutputTensor(0), 4096, 16,
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fcLane1_w.bin",
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fcLane1_b.bin",
                                                                0);
    (static_cast<MWReLULayer *>(this->layers[15]))->createReLULayer
            (this->targetImpl, this->layers[14]->getOutputTensor(0), 0);
    (static_cast<MWFCLayer *>(this->layers[16]))->createFCLayer(this->targetImpl,
                                                                this->layers[15]->getOutputTensor(0), 16, 6,
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fcLane2_w.bin",
                                                                "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_fcLane2_b.bin",
                                                                1);
    (static_cast<MWOutputLayer *>(this->layers[17]))->createOutputLayer
            (this->targetImpl, this->layers[16]->getOutputTensor(0), 1);
    this->outputTensors[0] = this->layers[17]->getOutputTensor(0);
    this->setSize();
}

//
// Arguments    : cudaError_t errCode
//                const char *file
//                unsigned int line
// Return Type  : void
//
static void checkCleanupCudaError(cudaError_t errCode, const char *file,
                                  unsigned int line) {
    if ((errCode != cudaSuccess) && (errCode != cudaErrorCudartUnloading)) {
        printf(errorString, cudaGetErrorString(errCode), file, line);
    }
}

//
// Arguments    : void
// Return Type  : int
//
int lanenet0_0::getBatchSize() {
    return this->inputTensors[0]->getBatchSize();
}

//
// Arguments    : int b_index
// Return Type  : float *
//
float *lanenet0_0::getInputDataPointer(int b_index) {
    return (static_cast<MWTensor<float> *>(this->inputTensors[b_index]))->getData();
}

//
// Arguments    : void
// Return Type  : float *
//
float *lanenet0_0::getInputDataPointer() {
    return (static_cast<MWTensor<float> *>(this->inputTensors[0]))->getData();
}

//
// Arguments    : int layerIndex
//                int portIndex
// Return Type  : float *
//
float *lanenet0_0::getLayerOutput(int layerIndex, int portIndex) {
    return this->layers[layerIndex]->getLayerOutput(portIndex);
}

//
// Arguments    : int b_index
// Return Type  : float *
//
float *lanenet0_0::getOutputDataPointer(int b_index) {
    return (static_cast<MWTensor<float> *>(this->outputTensors[b_index]))->getData
            ();
}

//
// Arguments    : void
// Return Type  : float *
//
float *lanenet0_0::getOutputDataPointer() {
    return (static_cast<MWTensor<float> *>(this->outputTensors[0]))->getData();
}

//
// Arguments    : void
// Return Type  : void
//
lanenet0_0::lanenet0_0() {
    this->numLayers = 18;
    this->targetImpl = 0;
    this->layers[0] = new MWInputLayer;
    this->layers[0]->setName("data");
    this->layers[1] = new MWElementwiseAffineLayer;
    this->layers[1]->setName("data_normalization");
    this->layers[1]->setInPlaceIndex(0, 0);
    this->layers[2] = new MWFusedConvReLULayer;
    this->layers[2]->setName("conv1_relu1");
    this->layers[3] = new MWNormLayer;
    this->layers[3]->setName("norm1");
    this->layers[4] = new MWMaxPoolingLayer;
    this->layers[4]->setName("pool1");
    this->layers[5] = new MWFusedConvReLULayer;
    this->layers[5]->setName("conv2_relu2");
    this->layers[6] = new MWNormLayer;
    this->layers[6]->setName("norm2");
    this->layers[7] = new MWMaxPoolingLayer;
    this->layers[7]->setName("pool2");
    this->layers[8] = new MWFusedConvReLULayer;
    this->layers[8]->setName("conv3_relu3");
    this->layers[9] = new MWFusedConvReLULayer;
    this->layers[9]->setName("conv4_relu4");
    this->layers[10] = new MWFusedConvReLULayer;
    this->layers[10]->setName("conv5_relu5");
    this->layers[11] = new MWMaxPoolingLayer;
    this->layers[11]->setName("pool5");
    this->layers[12] = new MWFCLayer;
    this->layers[12]->setName("fc6");
    this->layers[13] = new MWReLULayer;
    this->layers[13]->setName("relu6");
    this->layers[13]->setInPlaceIndex(0, 0);
    this->layers[14] = new MWFCLayer;
    this->layers[14]->setName("fcLane1");
    this->layers[15] = new MWReLULayer;
    this->layers[15]->setName("fcLane1Relu");
    this->layers[15]->setInPlaceIndex(0, 0);
    this->layers[16] = new MWFCLayer;
    this->layers[16]->setName("fcLane2");
    this->layers[17] = new MWOutputLayer;
    this->layers[17]->setName("output");
    this->layers[17]->setInPlaceIndex(0, 0);
    this->targetImpl = new MWTargetNetworkImpl;
    this->inputTensors[0] = new MWTensor<float>;
    this->inputTensors[0]->setHeight(227);
    this->inputTensors[0]->setWidth(227);
    this->inputTensors[0]->setChannels(3);
    this->inputTensors[0]->setBatchSize(1);
    this->inputTensors[0]->setSequenceLength(1);
}

//
// Arguments    : void
// Return Type  : void
//
lanenet0_0::~lanenet0_0() {
    this->cleanup();
    checkCleanupCudaError(cudaGetLastError(), __FILE__, __LINE__);
    for (int idx = 0; idx < 18; idx++) {
        delete this->layers[idx];
    }

    if (this->targetImpl) {
        delete this->targetImpl;
    }

    delete this->inputTensors[0];
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::predict() {
    for (int idx = 0; idx < 18; idx++) {
        this->layers[idx]->predict();
    }
}

//
// Arguments    : void
// Return Type  : void
//
void lanenet0_0::resetState() {
}

//
// Arguments    : lanenet0_0 *obj
// Return Type  : void
//
namespace coder {
    void DeepLearningNetwork_setup(lanenet0_0 *obj) {
        obj->setup();
    }
}

//
// File trailer for DeepLearningNetwork.cu
//
// [EOF]
//
