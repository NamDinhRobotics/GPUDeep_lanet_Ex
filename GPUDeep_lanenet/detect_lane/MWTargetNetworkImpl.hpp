/* Copyright 2017-2020 The MathWorks, Inc. */

#ifndef CNN_TARGET_NTWK_IMPL
#define CNN_TARGET_NTWK_IMPL

#include <cudnn.h>
#include <cublas_v2.h>
#include "cnn_api.hpp"

#include <vector>

#define MW_TARGET_TYPE_CUDNN 1

class MWTargetNetworkImpl {
public:

    MWTargetNetworkImpl()
            : xcusoQxPPodcHwVviCWI(0), NldNILHvuQqQPSAHXxdT(0), NmExSIssnXpisMKKatUq(0), MW_autoTune(true),
              ncMionCCOTOYjWcmaIVD(0), GvDXGhRLfipwBoRPoGfI(0) {}

    ~MWTargetNetworkImpl();

    void allocate(int, int);

    void deallocate();

    void preSetup();

    void postSetup(MWCNNLayer *layers[], int numLayers);

    void cleanup();

    void setProposedWorkSpaceSize(size_t);  // Set the proposed workspace size of the target network impl
    size_t *getProposedWorkSpaceSize();     // Get the proposed workspace size of the target network impl

    void setAllocatedWorkSpaceSize(size_t);  // Set the allocated workspace size of the target network impl 
    size_t *getAllocatedWorkSpaceSize();     // Get the allocated workspace size of the target network impl

    float *getWorkSpace();          // Get the workspace buffer in GPU memory
    cublasHandle_t *getCublasHandle();      // Get the cuBLAS handle to use for GPU computation
    cudnnHandle_t *getCudnnHandle();        // Get the cuDNN handle to use for GPU computation
    std::vector<float *> memBuffer;
    int numBufs;

    void setAutoTune(bool);

    bool getAutoTune() const;

    float *getBufferPtr(int bufferIndex);

    float *getPermuteBuffer(int index);    // Get the buffer in GPU memory for custom layers' data layout permutation
    void allocatePermuteBuffers(int, int); // allocate buffer for custom layers' data layout permutation

private:
    size_t ncMionCCOTOYjWcmaIVD;
    size_t GvDXGhRLfipwBoRPoGfI;
    float *xcusoQxPPodcHwVviCWI;
    cublasHandle_t *NldNILHvuQqQPSAHXxdT;
    cudnnHandle_t *NmExSIssnXpisMKKatUq;
    bool MW_autoTune;
    long int FLuSVNoPhAFKtLUchSvv;

    std::vector<float *> mtolGPkUMBYDlSSqrRzc;

private:
    void createWorkSpace(float *&);  // Create the workspace needed for this layer
    void destroyWorkSpace(float *&);

    static size_t getNextProposedWorkSpaceSize(size_t failedWorkSpaceSize);

public:
    static void getStrides(const int *dims, int size, int *stride);

    static void getTransformStrides(const int src[],
                                    const int dest[],
                                    const int dims[],
                                    int size,
                                    int strides[]);
};

#endif
