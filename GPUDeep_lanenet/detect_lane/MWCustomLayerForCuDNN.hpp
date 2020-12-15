/* Copyright 2019-2020 The MathWorks, Inc. */

// cuDNN specific header for Custom Layer Base Class
#ifndef CUSTOM_LAYER_CUDNN_HPP
#define CUSTOM_LAYER_CUDNN_HPP

#include "cnn_api.hpp"
#include "cudnn.h"
#include <map>

/* forward declare classes */
class MWTargetNetworkImpl;

class MWCustomLayerForCuDNN : public MWCNNLayer {

public:
    MWCustomLayerForCuDNN();

    ~MWCustomLayerForCuDNN();

    MWTargetNetworkImpl *m_ntwkImpl;

    void allocate();

    void deallocate();

    void cleanup();

protected:
    void setupLayer(MWTargetNetworkImpl *);

    // needs to be invoked from propagateSize() call
    // only for the tensors that need permutation between NCHW <--> NCWH or NCHW <--> NHWC
    void setupInputDescriptors(MWTensorBase *aTensor,
                               MWTensorBase::DIMSLABEL srcLayout[],
                               MWTensorBase::DIMSLABEL implLayout[]);

    void setupOutputDescriptors(MWTensorBase *aTensor,
                                MWTensorBase::DIMSLABEL implLayout[],
                                MWTensorBase::DIMSLABEL destLayout[]);

    // reorder data to between SNCHW <--> SNCWH or SNCHW <--> SNHWC
    void reorderInputData(MWTensorBase *aTensor, int bufIndex);

    void reorderOutputData(MWTensorBase *aTensor, int bufIndex);

    virtual void cleanupLayer() {};

private:
    // descriptors for input and output tensor data
    std::map<MWTensorBase *, cudnnTensorDescriptor_t *> m_inOutDescriptor;

    // descriptors for workspace data that holds the permuted/transformed data for the input and
    // output tensors
    std::map<MWTensorBase *, cudnnTensorDescriptor_t *> m_inOutTransformDescriptor;

    void createInOutDescriptors(MWTensorBase *aTensor);

    void createInOutTransformDescriptors(MWTensorBase *aTensor);
};

#endif
