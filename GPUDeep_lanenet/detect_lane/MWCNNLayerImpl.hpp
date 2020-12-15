/* Copyright 2017-2020 The MathWorks, Inc. */

#ifndef CNN_API_IMPL
#define CNN_API_IMPL

#include <cudnn.h>
#include <cublas_v2.h>
#include <map>
#include <vector>
#include <assert.h>
#include "cnn_api.hpp"


class MWTargetNetworkImpl;

#define CUDA_CALL(status) cuda_call_line_file(status, __LINE__, __FILE__)
#define MALLOC_CALL(msize) malloc_call_line_file(msize, __LINE__, __FILE__)
#define CUDNN_CALL(status) cudnn_call_line_file(status, __LINE__, __FILE__)
#define CUBLAS_CALL(status) cublas_call_line_file(status, __LINE__, __FILE__)
#define CUDA_FREE_CALL(buf) call_cuda_free(buf, __LINE__, __FILE__)

//#define RANDOM
#ifdef RANDOM
#include <curand.h>
#define CURAND_CALL(status) curand_call_line_file(status, __LINE__, __FILE__)
#endif

void cuda_call_line_file(cudaError_t, const int, const char *);

void cudnn_call_line_file(cudnnStatus_t, const int, const char *);

float *malloc_call_line_file(size_t, const int, const char *);

const char *cublasGetErrorString(cublasStatus_t);

void cublas_call_line_file(cublasStatus_t, const int, const char *);

void
throw_cuda_error(cudaError_t rlQsibXJSWJVnUVpdNeL, const int bDTIjtxZiSHtjwzgEluE, const char *QMgBqCuvjnbWHWiVPEwn);

template<class T>
void call_cuda_free(T *mem, const int bDTIjtxZiSHtjwzgEluE, const char *QMgBqCuvjnbWHWiVPEwn) {
    if (!mem) {
        return;
    }

    cudaError_t rlQsibXJSWJVnUVpdNeL = cudaFree(mem);

    if ((rlQsibXJSWJVnUVpdNeL != cudaSuccess) && (rlQsibXJSWJVnUVpdNeL != cudaErrorCudartUnloading)) {
        throw_cuda_error(rlQsibXJSWJVnUVpdNeL, bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
    }
}

#ifdef RANDOM
void curand_call_line_file(curandStatus_t, const int, const char*);
#endif


class MWCNNLayerImpl {
public:
    MWCNNLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *ntwk_impl);

    virtual ~MWCNNLayerImpl();

    virtual void predict() = 0;

    virtual void cleanup() = 0;

    virtual void propagateSize() = 0;

    virtual void allocate() {
    }

    template<class T>
    void allocateOutputData(int);

    virtual void deallocate() {
    }

    template<class T>
    void deallocateOutputData(int);

    virtual void postSetup() {
    }

    template<typename T>
    T *getData(int index = 0);

    MWCNNLayer *getLayer() {
        return atVCyzqXZAZxwlkRLBRA;
    }

    MWTargetNetworkImpl *getTargetNetworkImpl() {
        return dMxIKDGTITyhdLqIHBLA;
    }

protected:
    MWCNNLayer *atVCyzqXZAZxwlkRLBRA;
    std::map<int, cudnnTensorDescriptor_t *> lHtftnmGBvlSSoGOXVui; // output descriptors
    MWTargetNetworkImpl *dMxIKDGTITyhdLqIHBLA;

    float PfisSEEWDaQFynnzlcin;
    float OzygUJRIZYnGLzSjgahB;
    float OwenhowBxTAXHXmJpIKd;

    float *getZeroPtr();   // Get the pointer to a zero value parameter
    float *getOnePtr();    // Get the pointer to a one value parameter
    float *getNegOnePtr(); // Get the pointer to a negative one value parameter

protected:
    template<class T>
    static void setDescriptor(cudnnTensorDescriptor_t &, MWTensor<T> *);

    template<class T>
    static cudnnDataType_t getCuDNNDataType();

    cudnnTensorDescriptor_t *getDescriptor(int);

    cudnnTensorDescriptor_t *createAndAddDescriptor(int);

public:
    static void padInput(float *, int, int, int, int, int, int, int, float *, int);

    static void
    fillOutputBuffer(signed char *, int, int, int, int, int, int, int, signed char *, int, int);

    static cudnnTensorDescriptor_t getCuDNNDescriptor(MWTensorBase *);

    static void throwAllocationError(const int, const char *);
};

class MWInputLayerImpl : public MWCNNLayerImpl {

public:
    MWInputLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *ntwk_impl)
            : MWCNNLayerImpl(layer, ntwk_impl) {
    }

    ~MWInputLayerImpl() {
    }

    void predict() {
    }

    void cleanup() {
    }

    void propagateSize() {
    }
};

// ReLULayer
class MWReLULayerImpl : public MWCNNLayerImpl {
public:
    MWReLULayerImpl(MWCNNLayer *, MWTargetNetworkImpl *);

    ~MWReLULayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

private:
    cudnnActivationDescriptor_t oKIvzXXMucEDsTGGpdpm;
};

class MWNormLayerImpl : public MWCNNLayerImpl {
public:
    MWNormLayerImpl(MWCNNLayer *, MWTargetNetworkImpl *, unsigned, double, double, double);

    ~MWNormLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

private:
    cudnnLRNDescriptor_t dJcdBfQQLhIAYHPxwQeg;
};

// MaxPooling2DLayer
class MWMaxPoolingLayerImpl : public MWCNNLayerImpl {
public:
    // Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
    MWMaxPoolingLayerImpl(MWCNNLayer *,
                          MWTargetNetworkImpl *,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool,
                          int);

    ~MWMaxPoolingLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

    void allocate();

    void deallocate();

    float *getIndexData();

public:
    int DSsxcjIrUgZCKZovyNQf;
    int EfvWctmlsWAPsxXgdKWf;
    int DRzwhbNPpftRRIXXfHzd;
    int ECTnqgWHyHCHCLBZlffd;

    int CZNYmBcNFSZWvaCklqeM;
    int CTCbzQMDaLxINPbODdng;
    int CqtPRJvHlGJFssiPzsOm;
    int CufLFODQDXTAPyRqYodN;

    int FrpxvsDMwwgbpqHXWxmN;
    int FwLnexHgxHRquTKmNpoa;

    bool BUOdotSvmFyUWQKMUdra;

private:
    float *UdmcwaUkepxfZrpdpcAN;
    float *PVBPDNaynqYkBlDZgXgj;
    int fXhhiexIRPLyKXApPmmy;
    cudnnPoolingDescriptor_t muwRQxtWMMXAPxSuMYBw;
};

// FullyConnectedLayer
class MWFCLayerImpl : public MWCNNLayerImpl {
public:
    int CGbFsczkgkhjcHoCKzBx;

private:
    int BLjrjqvCcCommiXWQLjs;
    int BRSPqxNffoBYKqpSVHne;
    int CDJtexcMbXMWAmnNZsNf;
    float *vpXxoeEhdEosLSsYXkNG;
    float *wJyXsrUCMgxdIKVIJSyx;
    float *IwKnaBoXVubIRYcxEJLH;

    int xHiBGayUfxIpXKkCTDNU;

public:
    MWFCLayerImpl(MWCNNLayer *, MWTargetNetworkImpl *, int, int, const char *, const char *);

    ~MWFCLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

    void postSetup();

private:
    void loadWeights(const char *);

    void loadBias(const char *);

    void prepareWeights();

private:
    cudnnTensorDescriptor_t JgLfgHrHMEMmMYTettJF;
};

// SoftmaxLayer
class MWSoftmaxLayerImpl : public MWCNNLayerImpl {
public:
    MWSoftmaxLayerImpl(MWCNNLayer *, MWTargetNetworkImpl *);

    ~MWSoftmaxLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

private:
    cudnnLRNDescriptor_t dJcdBfQQLhIAYHPxwQeg;
    cudnnTensorDescriptor_t shEncNmxJsMuJKwbrwok;
    cudnnTensorDescriptor_t sjLjZacPSDNBEjAccrGU;
};

// AvgPooling2DLayer
class MWAvgPoolingLayerImpl : public MWCNNLayerImpl {
public:
    // Create AvgPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
    MWAvgPoolingLayerImpl(MWCNNLayer *,
                          MWTargetNetworkImpl *,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int);

    ~MWAvgPoolingLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

    void allocate();

    void deallocate();

public:
    int DSsxcjIrUgZCKZovyNQf;
    int EfvWctmlsWAPsxXgdKWf;
    int DRzwhbNPpftRRIXXfHzd;
    int ECTnqgWHyHCHCLBZlffd;
    int FrpxvsDMwwgbpqHXWxmN;
    int FwLnexHgxHRquTKmNpoa;
    int CZNYmBcNFSZWvaCklqeM;
    int CTCbzQMDaLxINPbODdng;
    int CqtPRJvHlGJFssiPzsOm;
    int CufLFODQDXTAPyRqYodN;

private:
    cudnnPoolingDescriptor_t muwRQxtWMMXAPxSuMYBw;
    int nDsbARncmIrIaLubvLVZ;
    int nNULvWnBXnnWdpEkHPAH;

    cudnnTensorDescriptor_t XhAYHFyEVtlwoxGBuTpu;
    // for temporary pre-padded input for asymmetric padding
    MWTensorBase *TbrNrGxaFFHrzKUcfHNZ;
    int cCXqPFPPcoHzYMDpnUxQ;
    int cAUupmktEnGPfLHyWfFm;
    bool IIiwAtyrOtLzLWAUlTey;
};

class MWOutputLayerImpl : public MWCNNLayerImpl {
public:
    MWOutputLayerImpl(MWCNNLayer *, MWTargetNetworkImpl *);

    ~MWOutputLayerImpl();

    void deallocateOutputData(int);

    void propagateSize();

    void predict();

    void cleanup();
};

#endif
