//
// File: detect_lane_internal_types.h
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//
#ifndef DETECT_LANE_INTERNAL_TYPES_H
#define DETECT_LANE_INTERNAL_TYPES_H

// Include Files
#include "detect_lane_types.h"
#include "rtwtypes.h"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"

// Type Definitions
class lanenet0_0 {
public:
    lanenet0_0();

    void setSize();

    void resetState();

    void setup();

    void predict();

    void cleanup();

    float *getLayerOutput(int layerIndex, int portIndex);

    float *getInputDataPointer(int b_index);

    float *getInputDataPointer();

    float *getOutputDataPointer(int b_index);

    float *getOutputDataPointer();

    int getBatchSize();

    ~lanenet0_0();

private:
    void allocate();

    void postsetup();

    void deallocate();

public:
    int numLayers;
    MWCNNLayer *layers[18];
private:
    MWTensorBase *inputTensors[1];
    MWTensorBase *outputTensors[1];
    MWTargetNetworkImpl *targetImpl;
};

#endif

//
// File trailer for detect_lane_internal_types.h
//
// [EOF]
//
