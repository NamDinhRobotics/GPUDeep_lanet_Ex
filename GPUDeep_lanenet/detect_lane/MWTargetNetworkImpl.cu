#include "MWTargetNetworkImpl.hpp"
#include "MWCNNLayerImpl.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>

void MWTargetNetworkImpl::allocate(int BufSize, int numBufsToAlloc) {
    numBufs
            = numBufsToAlloc;
    for (int i = 0; i < numBufs; i++) {
        float *memPtr = 0;
        CUDA_CALL(cudaMalloc((void **) &memPtr, sizeof(float) * BufSize));
        memBuffer.push_back(memPtr);
    }
}

void
MWTargetNetworkImpl::allocatePermuteBuffers(int bufSize, int numBufsToAlloc) {
    for (int i = 0; i < numBufsToAlloc; i++) {
        float *memPtr = 0;
        CUDA_CALL(cudaMalloc((void **) &memPtr, sizeof(float) * bufSize));
        mtolGPkUMBYDlSSqrRzc.push_back(memPtr);
    }
}

void
MWTargetNetworkImpl::preSetup() {
    NldNILHvuQqQPSAHXxdT = new
            cublasHandle_t;
    if (!NldNILHvuQqQPSAHXxdT) {
        MWCNNLayerImpl::throwAllocationError(__LINE__, __FILE__);
    }
    cublasCreate(NldNILHvuQqQPSAHXxdT);
    NmExSIssnXpisMKKatUq = new
            cudnnHandle_t;
    if (!NmExSIssnXpisMKKatUq) {
        MWCNNLayerImpl::throwAllocationError(__LINE__, __FILE__);
    }
    cudnnCreate(NmExSIssnXpisMKKatUq);
}

void
MWTargetNetworkImpl::postSetup(MWCNNLayer *layers[], int numLayers) {
    if
            (*getProposedWorkSpaceSize() > *getAllocatedWorkSpaceSize()) {
        if
                (xcusoQxPPodcHwVviCWI) { destroyWorkSpace(xcusoQxPPodcHwVviCWI); }
        createWorkSpace(xcusoQxPPodcHwVviCWI);
        while ((!xcusoQxPPodcHwVviCWI) &&
               (*getProposedWorkSpaceSize() > 0)) {
            setProposedWorkSpaceSize(MWTargetNetworkImpl::getNextProposedWorkSpaceSize(*getProposedWorkSpaceSize()));
            createWorkSpace(xcusoQxPPodcHwVviCWI);
        }
    }
    for (int i = 0; i < numLayers; i++) { layers[i]->postSetup(); }
}

size_t
MWTargetNetworkImpl::getNextProposedWorkSpaceSize(size_t failedWorkSpaceSize) {
    assert(failedWorkSpaceSize > 0);
    return failedWorkSpaceSize / 2;
}

void
MWTargetNetworkImpl::createWorkSpace(float *&xkUNToJIgvoLoUQuzKRF) {
    cudaError_t rlQsibXJSWJVnUVpdNeL = cudaMalloc((void **) &xkUNToJIgvoLoUQuzKRF,
                                                  *getProposedWorkSpaceSize());
    if (rlQsibXJSWJVnUVpdNeL != cudaSuccess) {
        xkUNToJIgvoLoUQuzKRF = NULL;
        setAllocatedWorkSpaceSize(0);
        rlQsibXJSWJVnUVpdNeL = cudaGetLastError();
    } else {
        setAllocatedWorkSpaceSize(*getProposedWorkSpaceSize());
    }
}

void
MWTargetNetworkImpl::destroyWorkSpace(float *&xkUNToJIgvoLoUQuzKRF) {
    CUDA_FREE_CALL(xkUNToJIgvoLoUQuzKRF);
    xkUNToJIgvoLoUQuzKRF = NULL;
    setAllocatedWorkSpaceSize(0);
}

void
MWTargetNetworkImpl::setProposedWorkSpaceSize(size_t wss) {
    ncMionCCOTOYjWcmaIVD = wss;
}

size_t *
MWTargetNetworkImpl::getProposedWorkSpaceSize() {
    return
            &ncMionCCOTOYjWcmaIVD;
}

void
MWTargetNetworkImpl::setAllocatedWorkSpaceSize(size_t wss) {
    GvDXGhRLfipwBoRPoGfI = wss;
}

size_t *
MWTargetNetworkImpl::getAllocatedWorkSpaceSize() {
    return
            &GvDXGhRLfipwBoRPoGfI;
}

float *
MWTargetNetworkImpl::getWorkSpace() { return xcusoQxPPodcHwVviCWI; }

float *
MWTargetNetworkImpl::getPermuteBuffer(int bufIndex) {
    return
            mtolGPkUMBYDlSSqrRzc[bufIndex];
}

cublasHandle_t *
MWTargetNetworkImpl::getCublasHandle() { return NldNILHvuQqQPSAHXxdT; }

cudnnHandle_t *MWTargetNetworkImpl::getCudnnHandle() {
    return
            NmExSIssnXpisMKKatUq;
}

void MWTargetNetworkImpl::setAutoTune(bool
                                      autotune) { MW_autoTune = autotune; }

bool MWTargetNetworkImpl::getAutoTune()
const { return MW_autoTune; }

void MWTargetNetworkImpl::deallocate() {
    for (int
                 i = 0; i < memBuffer.size(); i++) {
        float *memPtr = memBuffer[i];
        if (memPtr) {
            CUDA_FREE_CALL(memPtr);
        }
    }
    memBuffer.clear();
    for (int i = 0; i <
                    mtolGPkUMBYDlSSqrRzc.size(); i++) {
        float *memPtr =
                mtolGPkUMBYDlSSqrRzc[i];
        if (memPtr) { CUDA_FREE_CALL(memPtr); }
    }
    mtolGPkUMBYDlSSqrRzc.clear();
}

void MWTargetNetworkImpl::cleanup() {
    if
            (xcusoQxPPodcHwVviCWI) { destroyWorkSpace(xcusoQxPPodcHwVviCWI); }
    if
            (NldNILHvuQqQPSAHXxdT) {
        cublasDestroy(*NldNILHvuQqQPSAHXxdT);
        delete
                NldNILHvuQqQPSAHXxdT;
    }
    if (NmExSIssnXpisMKKatUq) {
        cudnnDestroy(*NmExSIssnXpisMKKatUq);
        delete NmExSIssnXpisMKKatUq;
    }
}

float *MWTargetNetworkImpl::getBufferPtr(int bufferIndex) {
    return
            memBuffer[bufferIndex];
}

MWTargetNetworkImpl::~MWTargetNetworkImpl() {}