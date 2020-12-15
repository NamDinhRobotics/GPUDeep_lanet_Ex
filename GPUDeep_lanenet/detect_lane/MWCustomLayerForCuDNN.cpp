/* Copyright 2019-2020 The MathWorks, Inc. */

// cuDNN specific implementation of the Custom Layer Base Class
#include "MWCustomLayerForCuDNN.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "MWCNNLayerImpl.hpp"

MWCustomLayerForCuDNN::MWCustomLayerForCuDNN()
        : m_ntwkImpl(NULL) {
    m_isCustomLayer = true;
}

void MWCustomLayerForCuDNN::setupLayer(MWTargetNetworkImpl *targetImpl) {
    m_ntwkImpl = targetImpl;
}

void MWCustomLayerForCuDNN::createInOutDescriptors(MWTensorBase *aTensor) {
    // create a new descriptor if one does not exist yet
    std::map<MWTensorBase *, cudnnTensorDescriptor_t *>::iterator it =
            m_inOutDescriptor.find(aTensor);
    if (it == m_inOutDescriptor.end()) {
        // create a descriptor if one does not exist already
        m_inOutDescriptor[aTensor] = new cudnnTensorDescriptor_t;
        CUDNN_CALL(cudnnCreateTensorDescriptor(m_inOutDescriptor[aTensor]));
    }
}

void MWCustomLayerForCuDNN::createInOutTransformDescriptors(MWTensorBase *aTensor) {
    // create a new descriptor if one does not exist yet
    std::map<MWTensorBase *, cudnnTensorDescriptor_t *>::iterator it =
            m_inOutTransformDescriptor.find(aTensor);
    if (it == m_inOutTransformDescriptor.end()) {
        m_inOutTransformDescriptor[aTensor] = new cudnnTensorDescriptor_t;
        CUDNN_CALL(cudnnCreateTensorDescriptor(m_inOutTransformDescriptor[aTensor]));
    }
}

void MWCustomLayerForCuDNN::setupInputDescriptors(MWTensorBase *aTensor,
                                                  MWTensorBase::DIMSLABEL srcLayout[],
                                                  MWTensorBase::DIMSLABEL customImplLayout[]) {

    // set the descriptor for the source : tensor buffer
    createInOutDescriptors(aTensor);

    const int size = 5;
    int inDims[size];
    aTensor->getDimsByLayout(srcLayout, size, inDims);

    int strides[size];
    MWTensorBase::getStrides(inDims, size, strides);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(*m_inOutDescriptor[aTensor], CUDNN_DATA_FLOAT, size,
                                          inDims, strides));


    // aTensor data will be transposed from SNCHW to SNCWH (row-major format to col-major format) or
    // vice-versa setting the descriptor for transpose operation : targetImpl buffer
    createInOutTransformDescriptors(aTensor);

    int outDims[size];
    aTensor->getDimsByLayout(customImplLayout, size, outDims);

    MWTensorBase::getTransformStrides(srcLayout, customImplLayout, outDims, size,
                                      strides); // compute strides based on dims for customImplLayout
    CUDNN_CALL(cudnnSetTensorNdDescriptor(*m_inOutTransformDescriptor[aTensor], CUDNN_DATA_FLOAT,
                                          size, inDims, strides)); // dims here still is input dims
}


void MWCustomLayerForCuDNN::setupOutputDescriptors(MWTensorBase *aTensor,
                                                   MWTensorBase::DIMSLABEL customImplLayout[],
                                                   MWTensorBase::DIMSLABEL destLayout[]) {

    // set the descriptor for the source : targetImpl buffer
    createInOutDescriptors(aTensor);

    const int size = 5;
    int inDims[size];
    aTensor->getDimsByLayout(customImplLayout, size, inDims);

    int strides[size];
    MWTensorBase::getStrides(inDims, size, strides);
    CUDNN_CALL(cudnnSetTensorNdDescriptor(*m_inOutDescriptor[aTensor], CUDNN_DATA_FLOAT, size,
                                          inDims, strides));

    // aTensor data will be transposed from SNCHW to SNCWH (row-major format to col-major format) or
    // vice-versa setting the descriptor for transpose operation : tensor buffer
    createInOutTransformDescriptors(aTensor);

    int outDims[size];
    aTensor->getDimsByLayout(destLayout, size, outDims);

    MWTensorBase::getTransformStrides(customImplLayout, destLayout, outDims, size,
                                      strides); // compute strides based on dims for destLayout
    CUDNN_CALL(cudnnSetTensorNdDescriptor(*m_inOutTransformDescriptor[aTensor], CUDNN_DATA_FLOAT,
                                          size, inDims, strides));
}

void MWCustomLayerForCuDNN::reorderInputData(MWTensorBase *aTensor, int bufIndex) {

    float oneV = 1.0;
    float zeroV = 0.0;

    // perform reorder operation
    // aTensor->getData() is in NCHW format
    // want to populate buffer with data in NCWH/NHWC format
    CUDNN_CALL(cudnnTransformTensor(
            *m_ntwkImpl->getCudnnHandle(), &oneV, *m_inOutDescriptor[aTensor],
            static_cast<MWTensor<float> *>(aTensor)->getData(), &zeroV,
            *m_inOutTransformDescriptor[aTensor], m_ntwkImpl->getPermuteBuffer(bufIndex)));
}

void MWCustomLayerForCuDNN::reorderOutputData(MWTensorBase *aTensor, int bufIndex) {

    float oneV = 1.0;
    float zeroV = 0.0;

    // perform reorder operation
    // source data here can be either in NCWH/NHWC format
    // want to populate aTensor->getData() with data in NCHW format
    CUDNN_CALL(cudnnTransformTensor(
            *m_ntwkImpl->getCudnnHandle(), &oneV, *m_inOutDescriptor[aTensor],
            m_ntwkImpl->getPermuteBuffer(bufIndex), &zeroV, *m_inOutTransformDescriptor[aTensor],
            static_cast<MWTensor<float> *>(aTensor)->getData()));
}

void MWCustomLayerForCuDNN::allocate() {
    for (size_t iOut = 0; iOut < getNumOutputs(); iOut++) {
        MWTensorBase *opTensorBase = getOutputTensor(iOut);
        MWTensor<float> *opTensor = static_cast<MWTensor<float> *>(opTensorBase);
        opTensor->setData(m_ntwkImpl->getBufferPtr(opTensor->getopBufIndex()));
    }
}

void MWCustomLayerForCuDNN::deallocate() {
    for (size_t iOut = 0; iOut < getNumOutputs(); iOut++) {
        static_cast<MWTensor<float> *>(getOutputTensor(iOut))->setData((float *) NULL);
    }
}

void MWCustomLayerForCuDNN::cleanup() {

    // call cleanup on the descriptors
    for (std::map<MWTensorBase *, cudnnTensorDescriptor_t *>::iterator it = m_inOutDescriptor.begin();
         it != m_inOutDescriptor.end(); ++it) {
        CUDNN_CALL(cudnnDestroyTensorDescriptor(*it->second));
        delete it->second;
        it->second = 0;
    }

    for (std::map<MWTensorBase *, cudnnTensorDescriptor_t *>::iterator it =
            m_inOutTransformDescriptor.begin();
         it != m_inOutTransformDescriptor.end(); ++it) {
        CUDNN_CALL(cudnnDestroyTensorDescriptor(*it->second));
        delete it->second;
        it->second = 0;
    }

    // call layer specific cleanup (this code will be auto-generated)
    this->cleanupLayer();

    // call base layer cleanup
    this->MWCNNLayer::cleanup();
}

MWCustomLayerForCuDNN::~MWCustomLayerForCuDNN() {
}
