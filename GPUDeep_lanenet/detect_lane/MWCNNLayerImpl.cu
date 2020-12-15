#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <stdexcept>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"

#ifdef RANDOM
#include <curand.h>
curandGenerator_t REXdEoRjxuQJkqgIDihy; void
curand_call_line_file(curandStatus_t rlQsibXJSWJVnUVpdNeL, const int 
bDTIjtxZiSHtjwzgEluE, const char *QMgBqCuvjnbWHWiVPEwn) { if (rlQsibXJSWJVnUVpdNeL != 
CURAND_STATUS_SUCCESS) { char buffer[100]; int numElem = sprintf(buffer, 
"%d at line: %d, file: %s\n", rlQsibXJSWJVnUVpdNeL, bDTIjtxZiSHtjwzgEluE, 
QMgBqCuvjnbWHWiVPEwn); throw std::runtime_error(buffer); } }
#endif

float *malloc_call_line_file(size_t msize, const int bDTIjtxZiSHtjwzgEluE, const
char *QMgBqCuvjnbWHWiVPEwn) {
    float *mem = (float *) malloc(msize);
    if (!mem) {
        char
                buffer[100];
        int numElem = sprintf(buffer, "%s at line: %d, file: %s\n",
                              "Memory allocation failed. ", bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
        throw
                std::runtime_error(buffer);
    }
    return mem;
}

void
cuda_call_line_file(cudaError_t rlQsibXJSWJVnUVpdNeL, const int bDTIjtxZiSHtjwzgEluE,
                    const char *QMgBqCuvjnbWHWiVPEwn) {
    if (rlQsibXJSWJVnUVpdNeL != cudaSuccess) {
        throw_cuda_error(rlQsibXJSWJVnUVpdNeL, bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
    }
}

void throw_cuda_error(cudaError_t rlQsibXJSWJVnUVpdNeL, const int bDTIjtxZiSHtjwzgEluE,
                      const char *QMgBqCuvjnbWHWiVPEwn) {
    char buffer[100];
    int numElem = sprintf(buffer,
                          "Cuda Error %d(%s) at line: %d, file: %s\n", rlQsibXJSWJVnUVpdNeL,
                          cudaGetErrorString(rlQsibXJSWJVnUVpdNeL), bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
    rlQsibXJSWJVnUVpdNeL = cudaGetLastError();
    throw std::runtime_error(buffer);
}

void cudnn_call_line_file(cudnnStatus_t rlQsibXJSWJVnUVpdNeL, const int
bDTIjtxZiSHtjwzgEluE, const char *QMgBqCuvjnbWHWiVPEwn) {
    if (rlQsibXJSWJVnUVpdNeL !=
        CUDNN_STATUS_SUCCESS) {
        char buffer[100];
        int numElem = sprintf(buffer,
                              "CuDNN Error %d(%s) at line: %d, file: %s\n", rlQsibXJSWJVnUVpdNeL,
                              cudnnGetErrorString(rlQsibXJSWJVnUVpdNeL), bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
        throw std::runtime_error(buffer);
    }
}

const char *
cublasGetErrorString(cublasStatus_t rlQsibXJSWJVnUVpdNeL) {
    switch (rlQsibXJSWJVnUVpdNeL) {
        case CUBLAS_STATUS_SUCCESS:
            return
                    "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return
                    "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return
                    "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return
                    "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return
                    "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return
                    "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return
                    "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return
                    "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return
                    "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return
                    "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

void
cublas_call_line_file(cublasStatus_t rlQsibXJSWJVnUVpdNeL, const int
bDTIjtxZiSHtjwzgEluE, const char *QMgBqCuvjnbWHWiVPEwn) {
    if (rlQsibXJSWJVnUVpdNeL !=
        CUBLAS_STATUS_SUCCESS) {
        char buffer[100];
        int numElem = sprintf(buffer,
                              "CuBlas Error %d(%s) at line: %d, file: %s\n", rlQsibXJSWJVnUVpdNeL,
                              cublasGetErrorString(rlQsibXJSWJVnUVpdNeL), bDTIjtxZiSHtjwzgEluE, QMgBqCuvjnbWHWiVPEwn);
        throw std::runtime_error(buffer);
    }
}

MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *
ntwk_impl) : PfisSEEWDaQFynnzlcin(0.0), OzygUJRIZYnGLzSjgahB(1.0),
             OwenhowBxTAXHXmJpIKd(-1.0), atVCyzqXZAZxwlkRLBRA(layer),
             dMxIKDGTITyhdLqIHBLA(ntwk_impl) {}

MWCNNLayerImpl::~MWCNNLayerImpl() {
    for (std::map<int, cudnnTensorDescriptor_t *>::iterator it =
            lHtftnmGBvlSSoGOXVui.begin(); it != lHtftnmGBvlSSoGOXVui.end(); ++it) {
        CUDNN_CALL(cudnnDestroyTensorDescriptor(*it->second));
        delete it->second;
        it->second = 0;
    }
}

template<class T>
void
MWCNNLayerImpl::allocateOutputData(int outIdx) {
    MWTensorBase *opTensorBase =
            getLayer()->getOutputTensor(outIdx);
    bool bufferReuse =
            opTensorBase->getopBufIndex() >= 0;
    if (bufferReuse) {
        assert(opTensorBase->isFloat());
        MWTensor<float> *opTensor =
                static_cast<MWTensor<float> *>(opTensorBase);
        opTensor->setData(dMxIKDGTITyhdLqIHBLA->memBuffer[opTensor->getopBufIndex()]);
    } else {
        int inIdx = getLayer()->getInPlaceIndex(outIdx);
        if (inIdx != -1) {
            MWTensor<T> *ipTensor =
                    static_cast<MWTensor<T> *>(getLayer()->getInputTensor(inIdx));
            MWTensor<T> *
                    opTensor = static_cast<MWTensor<T> *>(opTensorBase);
            T *ipData =
                    ipTensor->getData();
            assert(ipData);
            opTensor->setData(ipData);
        } else {
            MWTensor<T> *opTensor = static_cast<MWTensor<T> *>(opTensorBase);
            T *
                    OKaRVOctKLlnIyGmjRNW;
            CUDA_CALL(cudaMalloc((void **) &OKaRVOctKLlnIyGmjRNW,
                                 sizeof(T) * opTensor->getNumElements()));
            opTensor->setData(OKaRVOctKLlnIyGmjRNW);
        }
    }
}

template void MWCNNLayerImpl::allocateOutputData<float>(int);

template void
MWCNNLayerImpl::allocateOutputData<signed char>(int);

template<class T>
void
MWCNNLayerImpl::deallocateOutputData(int outIdx) {
    if (getLayer()->getInPlaceIndex(outIdx) == -1) {
        MWTensor<T> *opTensor =
                static_cast<MWTensor<T> *>(getLayer()->getOutputTensor(outIdx));
        T *data =
                opTensor->getData();
        CUDA_FREE_CALL(data);
    }
}

template void
MWCNNLayerImpl::deallocateOutputData<float>(int);

template void
MWCNNLayerImpl::deallocateOutputData<signed char>(int);

float *
MWCNNLayerImpl::getZeroPtr() { return &PfisSEEWDaQFynnzlcin; }

float *
MWCNNLayerImpl::getOnePtr() { return &OzygUJRIZYnGLzSjgahB; }

float *
MWCNNLayerImpl::getNegOnePtr() { return &OwenhowBxTAXHXmJpIKd; }

cudnnTensorDescriptor_t *MWCNNLayerImpl::createAndAddDescriptor(int index) {
    std::map<int, cudnnTensorDescriptor_t *>::iterator it =
            lHtftnmGBvlSSoGOXVui.find(index);
    assert(it == lHtftnmGBvlSSoGOXVui.end());
    cudnnTensorDescriptor_t *newDescriptor = new cudnnTensorDescriptor_t;
    if
            (!newDescriptor) { MWCNNLayerImpl::throwAllocationError(__LINE__, __FILE__); }
    lHtftnmGBvlSSoGOXVui[index] = newDescriptor;
    CUDNN_CALL(cudnnCreateTensorDescriptor(newDescriptor));
    return newDescriptor;
}

cudnnTensorDescriptor_t *MWCNNLayerImpl::getDescriptor(int index) {
    std::map<int, cudnnTensorDescriptor_t *>::iterator it =
            lHtftnmGBvlSSoGOXVui.find(index);
    if (it != lHtftnmGBvlSSoGOXVui.end()) {
        return it->second;
    } else { return NULL; }
}

template<class T>
void
MWCNNLayerImpl::setDescriptor(cudnnTensorDescriptor_t &desc, MWTensor<T> *
tensor) {
    if (tensor->getSequenceLength() == 1) {
        CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                              MWCNNLayerImpl::getCuDNNDataType<T>(), tensor->getBatchSize(),
                                              tensor->getChannels(), tensor->getHeight(), tensor->getWidth()));
    } else {
        int dims[5] = {tensor->getSequenceLength(), tensor->getBatchSize(),
                       tensor->getChannels(), tensor->getHeight(), tensor->getWidth()};
        int
                strides[5];
        MWTensorBase::getStrides(dims, 5, strides);
        CUDNN_CALL(cudnnSetTensorNdDescriptor(desc,
                                              MWCNNLayerImpl::getCuDNNDataType<T>(), 5, dims, strides));
    }
}

template void
MWCNNLayerImpl::setDescriptor<float>(cudnnTensorDescriptor_t &,
                                     MWTensor<float> *);

template void MWCNNLayerImpl::setDescriptor<signed
char>(cudnnTensorDescriptor_t &, MWTensor<signed char> *);

template<>
cudnnDataType_t MWCNNLayerImpl::getCuDNNDataType<float>() {
    return
            CUDNN_DATA_FLOAT;
}

template<>
cudnnDataType_t
MWCNNLayerImpl::getCuDNNDataType<signed char>() { return CUDNN_DATA_INT8; }

cudnnTensorDescriptor_t MWCNNLayerImpl::getCuDNNDescriptor(MWTensorBase *
tensor) {
    MWCNNLayer *layer = tensor->getOwner();
    MWCNNLayerImpl *impl =
            layer->getImpl();
    if (impl) {
        cudnnTensorDescriptor_t *desc =
                impl->getDescriptor(tensor->getSourcePortIndex());
        if (desc == NULL) {
            impl->createAndAddDescriptor(tensor->getSourcePortIndex());
            desc =
                    impl->getDescriptor(tensor->getSourcePortIndex());
            assert(desc);
        }
        if
                (tensor->isFloat()) {
            MWCNNLayerImpl::setDescriptor<float>(*desc,
                                                 static_cast<MWTensor<float> *>(tensor));
        } else {
            assert(tensor->isInt8());
            MWCNNLayerImpl::setDescriptor<signed char>(*desc, static_cast<MWTensor<signed
            char> *>(tensor));
        }
        return *desc;
    } else {
        cudnnTensorDescriptor_t
                tmpDescriptor;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&tmpDescriptor));
        if
                (tensor->isFloat()) {
            MWCNNLayerImpl::setDescriptor<float>(tmpDescriptor,
                                                 static_cast<MWTensor<float> *>(tensor));
        } else {
            assert(tensor->isInt8());
            MWCNNLayerImpl::setDescriptor<signed char>(tmpDescriptor,
                                                       static_cast<MWTensor<signed char> *>(tensor));
        }
        return tmpDescriptor;
    }
}

void
__global__ __launch_bounds__(1024) padInputImpl(float *in, int inputH, int
inputW, int inputCh, int outputH, int outputW, int offsetH, int offsetW, float *
out, int inputElems) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i <
                                                        inputElems; i += blockDim.x * gridDim.x) {
        int idxB = i / (inputH * inputW * inputCh);
        int rem = (i - idxB * (inputH * inputW * inputCh));
        int idxCh = rem / (inputH * inputW);
        int rem1 = rem - idxCh * (inputH * inputW);
        int idxH = rem1 / inputW;
        int idxCol =
                rem1 - idxH * inputW;
        if ((idxH < inputH) && (idxCol < inputW)) {
            int outputR =
                    idxH + offsetH;
            int outputCol = idxCol + offsetW;
            int outputCh = inputCh;
            out[idxB * (outputH * outputW * outputCh) + idxCh * (outputH * outputW) +
                outputR * (outputW) + outputCol] = in[i];
        }
    }
}

void
MWCNNLayerImpl::padInput(float *TbrNrGxaFFHrzKUcfHNZ, int VenwEUlYwOBrwLVUhgUH, int
WOJynDmqVUPWjAGVIuMQ, int VFKMunbyHoAmpHUSkuUn, int lWJYwWaFPmWNQDPrlqER, int
                         lsqeARVLtpJTWezgnTkg, int gWETwFdWHfKuelmlKNCC, int hDaNSVZAofAENeIAiWEw, float *
jscBrjkVJyVfMMDjFpgl, int eqOmMKQRpqBqRQCnJmxt) {
    int tGsvtyAVkrDznETdweDC =
            (eqOmMKQRpqBqRQCnJmxt + 31) / 32 * 32;
    tGsvtyAVkrDznETdweDC =
            (tGsvtyAVkrDznETdweDC < 1024) ? tGsvtyAVkrDznETdweDC : 1024;
    int
            KHClOltUSuqFVVErSxVb = (eqOmMKQRpqBqRQCnJmxt + tGsvtyAVkrDznETdweDC -
                                    1) / tGsvtyAVkrDznETdweDC;
    padInputImpl<<<KHClOltUSuqFVVErSxVb,
    tGsvtyAVkrDznETdweDC>>>(TbrNrGxaFFHrzKUcfHNZ, VenwEUlYwOBrwLVUhgUH,
                            WOJynDmqVUPWjAGVIuMQ, VFKMunbyHoAmpHUSkuUn, lWJYwWaFPmWNQDPrlqER, lsqeARVLtpJTWezgnTkg,
                            gWETwFdWHfKuelmlKNCC, hDaNSVZAofAENeIAiWEw, jscBrjkVJyVfMMDjFpgl, eqOmMKQRpqBqRQCnJmxt);
}

void __global__ __launch_bounds__(1024) fillOutputBufferImpl(signed char *in,
                                                             int inputH, int inputW, int inputCh, int outputH,
                                                             int outputW, int offsetH, int
                                                             offsetW, signed char *out, int inputElems, int outputCh) {
    for (int i =
            blockDim.x * blockIdx.x + threadIdx.x; i < inputElems; i +=
                                                                           blockDim.x * gridDim.x) {
        int idxB = i / (inputH * inputW * inputCh);
        int rem = (i -
                   idxB * (inputH * inputW * inputCh));
        int idxCh = rem / (inputH * inputW);
        int rem1 = rem
                   - idxCh * (inputH * inputW);
        int idxH = rem1 / inputW;
        int idxCol = rem1 -
                     idxH * inputW;
        if ((idxH < inputH) && (idxCol < inputW)) {
            int outputR = idxH +
                          offsetH;
            int outputCol = idxCol + offsetW;
            *(out +
              idxB * (outputH * outputW * outputCh) + idxCh * (outputH * outputW) + outputR * (outputW) +
              outputCol) = *(in + i);
        }
    }
}

void MWCNNLayerImpl::fillOutputBuffer(signed
                                      char *TbrNrGxaFFHrzKUcfHNZ, int VenwEUlYwOBrwLVUhgUH, int WOJynDmqVUPWjAGVIuMQ,
                                      int
                                      VFKMunbyHoAmpHUSkuUn, int lWJYwWaFPmWNQDPrlqER, int lsqeARVLtpJTWezgnTkg, int
                                      gWETwFdWHfKuelmlKNCC, int hDaNSVZAofAENeIAiWEw, signed char *jscBrjkVJyVfMMDjFpgl,
                                      int
                                      eqOmMKQRpqBqRQCnJmxt, int kqftrrQBBOgGsrDSkIUk) {
    int tGsvtyAVkrDznETdweDC
            = (eqOmMKQRpqBqRQCnJmxt < 1024) ? eqOmMKQRpqBqRQCnJmxt : 1024;
    int
            KHClOltUSuqFVVErSxVb = (eqOmMKQRpqBqRQCnJmxt + tGsvtyAVkrDznETdweDC -
                                    1) / tGsvtyAVkrDznETdweDC;
    fillOutputBufferImpl<<<KHClOltUSuqFVVErSxVb,
    tGsvtyAVkrDznETdweDC>>>(TbrNrGxaFFHrzKUcfHNZ, VenwEUlYwOBrwLVUhgUH,
                            WOJynDmqVUPWjAGVIuMQ, VFKMunbyHoAmpHUSkuUn, lWJYwWaFPmWNQDPrlqER, lsqeARVLtpJTWezgnTkg,
                            gWETwFdWHfKuelmlKNCC, hDaNSVZAofAENeIAiWEw, jscBrjkVJyVfMMDjFpgl, eqOmMKQRpqBqRQCnJmxt,
                            kqftrrQBBOgGsrDSkIUk);
}

void MWCNNLayerImpl::throwAllocationError(const int
                                          line, const char *file) {
    char buffer[200];
    int numElem = sprintf(buffer,
                          "Failed to allocate memory at %d, file %s\n", line, file);
    throw
            std::runtime_error(buffer);
}

MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer *
layer, MWTargetNetworkImpl *ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) {
    CUDNN_CALL(cudnnCreateActivationDescriptor(&oKIvzXXMucEDsTGGpdpm));
    createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
}

MWReLULayerImpl::~MWReLULayerImpl() {}

void MWReLULayerImpl::propagateSize() {
    MWTensorBase *opTensor = getLayer()->getOutputTensor(0);
    cudnnTensorDescriptor_t *desc = getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc,
                         static_cast<MWTensor<float> *>(opTensor));
    CUDNN_CALL(cudnnSetActivationDescriptor(oKIvzXXMucEDsTGGpdpm,
                                            CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
}

void
MWReLULayerImpl::predict() {
    MWReLULayer *reluLayer =
            static_cast<MWReLULayer *>(getLayer());
    MWTensorBase *ipTensorBase =
            reluLayer->getInputTensor(0);
    MWTensorBase *opTensorBase =
            reluLayer->getOutputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    cudnnTensorDescriptor_t ipDesc =
            MWCNNLayerImpl::getCuDNNDescriptor(ipTensorBase);
    CUDNN_CALL(cudnnActivationForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                      oKIvzXXMucEDsTGGpdpm, getOnePtr(), ipDesc, ipTensor->getData(), getZeroPtr(),
                                      *desc, opTensor->getData()));
}

void MWReLULayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyActivationDescriptor(oKIvzXXMucEDsTGGpdpm));
}

MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *
ntwk_impl, unsigned GFggoMvRWucDMqzlWzCl, double AHqhysOOIgbDpWZoPUFT,
                                 double AIXLuRgdeiqpaCehGSYD, double BdqURaHPmdnfzvtUvocl) : MWCNNLayerImpl(layer,
                                                                                                            ntwk_impl) {
    CUDNN_CALL(cudnnCreateLRNDescriptor(&dJcdBfQQLhIAYHPxwQeg));
    createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
    CUDNN_CALL(cudnnSetLRNDescriptor(dJcdBfQQLhIAYHPxwQeg,
                                     GFggoMvRWucDMqzlWzCl, AHqhysOOIgbDpWZoPUFT, AIXLuRgdeiqpaCehGSYD,
                                     BdqURaHPmdnfzvtUvocl));
}

MWNormLayerImpl::~MWNormLayerImpl() {}

void
MWNormLayerImpl::propagateSize() {
    MWTensorBase *opTensor =
            getLayer()->getOutputTensor(0);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc, static_cast<MWTensor<float> *>(opTensor));
}

void
MWNormLayerImpl::predict() {
    MWNormLayer *normLayer =
            static_cast<MWNormLayer *>(getLayer());
    MWTensorBase *ipTensorBase =
            normLayer->getInputTensor();
    MWTensorBase *opTensorBase =
            normLayer->getOutputTensor();
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    cudnnTensorDescriptor_t ipDesc =
            MWCNNLayerImpl::getCuDNNDescriptor(ipTensorBase);
    CUDNN_CALL(cudnnLRNCrossChannelForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                           dJcdBfQQLhIAYHPxwQeg, CUDNN_LRN_CROSS_CHANNEL_DIM1, getOnePtr(), ipDesc,
                                           ipTensor->getData(), getZeroPtr(), *desc, opTensor->getData()));
}

void
MWNormLayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyLRNDescriptor(dJcdBfQQLhIAYHPxwQeg));
}

void __global__
MWSetDyForBackPropImpl(float *PVBPDNaynqYkBlDZgXgj, const int fOpFYwKNwIfWjnPzNuob);

void __global__ doMWMaxPoolingLayerImpl(float *UdmcwaUkepxfZrpdpcAN,
                                        float *UWAGLbDcvybdWBtshhsr, const int BkwhtPQUCQKchmmimoXs);

MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer *layer,
                                             MWTargetNetworkImpl *ntwk_impl, int DqxLTLaJwwgQqmrtCDuu, int
                                             EvebzoroiuKkIxwjkGnD, int FshVHIJMRAhtQirYPlZd, int
                                             GDRXdUDklKFEYEfifhIH, int CpMjJjtGOeWOzwxpAAQP, int
                                             ClEhcJFlvGCgiavziIag, int DCdZnqpcBnvXVgEsLBnz, int
                                             DGzdAcREJHGXjyRzNjJV, bool GIbahSoBBDrvvZduPEqU, int fSKMHAqIghbYYgyIpNDw)
        : MWCNNLayerImpl(layer, ntwk_impl),
          BUOdotSvmFyUWQKMUdra(GIbahSoBBDrvvZduPEqU), UdmcwaUkepxfZrpdpcAN(0), PVBPDNaynqYkBlDZgXgj(0),
          DSsxcjIrUgZCKZovyNQf(DqxLTLaJwwgQqmrtCDuu),
          EfvWctmlsWAPsxXgdKWf(EvebzoroiuKkIxwjkGnD),
          DRzwhbNPpftRRIXXfHzd(DqxLTLaJwwgQqmrtCDuu),
          ECTnqgWHyHCHCLBZlffd(EvebzoroiuKkIxwjkGnD),
          CZNYmBcNFSZWvaCklqeM(CpMjJjtGOeWOzwxpAAQP),
          CTCbzQMDaLxINPbODdng(ClEhcJFlvGCgiavziIag),
          CqtPRJvHlGJFssiPzsOm(DCdZnqpcBnvXVgEsLBnz),
          CufLFODQDXTAPyRqYodN(DGzdAcREJHGXjyRzNjJV),
          FrpxvsDMwwgbpqHXWxmN(FshVHIJMRAhtQirYPlZd),
          FwLnexHgxHRquTKmNpoa(GDRXdUDklKFEYEfifhIH),
          fXhhiexIRPLyKXApPmmy(fSKMHAqIghbYYgyIpNDw) {
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&muwRQxtWMMXAPxSuMYBw));
    createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
}

MWMaxPoolingLayerImpl::~MWMaxPoolingLayerImpl() {}

void
MWMaxPoolingLayerImpl::propagateSize() {
    MWTensorBase *ipTensor =
            getLayer()->getInputTensor(0);
    MWTensorBase *opTensor =
            getLayer()->getOutputTensor(0);
    if ((DSsxcjIrUgZCKZovyNQf == -1) &&
        (EfvWctmlsWAPsxXgdKWf == -1)) {
        DRzwhbNPpftRRIXXfHzd = ipTensor->getHeight();
        ECTnqgWHyHCHCLBZlffd = ipTensor->getWidth();
    }
    int nDsbARncmIrIaLubvLVZ =
            CZNYmBcNFSZWvaCklqeM;
    int nNULvWnBXnnWdpEkHPAH =
            CqtPRJvHlGJFssiPzsOm;
    CUDNN_CALL(cudnnSetPooling2dDescriptor(muwRQxtWMMXAPxSuMYBw, CUDNN_POOLING_MAX,
                                           CUDNN_NOT_PROPAGATE_NAN, DRzwhbNPpftRRIXXfHzd, ECTnqgWHyHCHCLBZlffd,
                                           nDsbARncmIrIaLubvLVZ, nNULvWnBXnnWdpEkHPAH, FrpxvsDMwwgbpqHXWxmN,
                                           FwLnexHgxHRquTKmNpoa));
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc, static_cast<MWTensor<float> *>(opTensor));
}

void
MWMaxPoolingLayerImpl::allocate() {
    MWMaxPoolingLayer *maxpoolLayer =
            static_cast<MWMaxPoolingLayer *>(getLayer());
    MWTensorBase *ipTensor =
            maxpoolLayer->getInputTensor(0);
    MWTensorBase *opTensor =
            maxpoolLayer->getOutputTensor(0);
    if (BUOdotSvmFyUWQKMUdra) {
        const int
                edQOkUJIZbwzEeIcCLzG = ipTensor->getNumElements();
        CUDA_CALL(cudaMalloc((void **) &UdmcwaUkepxfZrpdpcAN,
                             sizeof(float) * edQOkUJIZbwzEeIcCLzG));
        const int fOpFYwKNwIfWjnPzNuob =
                opTensor->getNumElements();
        CUDA_CALL(cudaMalloc((void **) &PVBPDNaynqYkBlDZgXgj,
                             sizeof(float) * fOpFYwKNwIfWjnPzNuob));
        int tGsvtyAVkrDznETdweDC =
                (fOpFYwKNwIfWjnPzNuob < 1024) ? fOpFYwKNwIfWjnPzNuob : 1024;
        int
                KHClOltUSuqFVVErSxVb = (fOpFYwKNwIfWjnPzNuob + tGsvtyAVkrDznETdweDC -
                                        1) / tGsvtyAVkrDznETdweDC;
        MWSetDyForBackPropImpl<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(PVBPDNaynqYkBlDZgXgj, fOpFYwKNwIfWjnPzNuob);
    }
}

void
MWMaxPoolingLayerImpl::deallocate() {
    if (UdmcwaUkepxfZrpdpcAN) {
        CUDA_FREE_CALL(UdmcwaUkepxfZrpdpcAN);
        UdmcwaUkepxfZrpdpcAN =
                NULL;
    }
    if (PVBPDNaynqYkBlDZgXgj) {
        CUDA_FREE_CALL(PVBPDNaynqYkBlDZgXgj);
        PVBPDNaynqYkBlDZgXgj =
                NULL;
    }
}

void MWMaxPoolingLayerImpl::predict() {
    MWMaxPoolingLayer *
            maxpoolLayer = static_cast<MWMaxPoolingLayer *>(getLayer());
    MWTensorBase *
            ipTensorBase = maxpoolLayer->getInputTensor(0);
    MWTensorBase *opTensorBase =
            maxpoolLayer->getOutputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    cudnnTensorDescriptor_t XhAYHFyEVtlwoxGBuTpu =
            MWCNNLayerImpl::getCuDNNDescriptor(ipTensorBase);
    CUDNN_CALL(cudnnPoolingForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                   muwRQxtWMMXAPxSuMYBw, getOnePtr(), XhAYHFyEVtlwoxGBuTpu, ipTensor->getData(),
                                   getZeroPtr(), *desc, opTensor->getData()));
    if (BUOdotSvmFyUWQKMUdra) {
        CUDNN_CALL(cudnnPoolingBackward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                        muwRQxtWMMXAPxSuMYBw, getOnePtr(), *desc, opTensor->getData(), *desc,
                                        PVBPDNaynqYkBlDZgXgj, XhAYHFyEVtlwoxGBuTpu, ipTensor->getData(), getZeroPtr(),
                                        XhAYHFyEVtlwoxGBuTpu, UdmcwaUkepxfZrpdpcAN));
        int edQOkUJIZbwzEeIcCLzG =
                ipTensor->getNumElements();
        int tGsvtyAVkrDznETdweDC =
                (edQOkUJIZbwzEeIcCLzG < 1024) ? edQOkUJIZbwzEeIcCLzG : 1024;
        int
                KHClOltUSuqFVVErSxVb = (edQOkUJIZbwzEeIcCLzG + tGsvtyAVkrDznETdweDC -
                                        1) / tGsvtyAVkrDznETdweDC;
        doMWMaxPoolingLayerImpl<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(UdmcwaUkepxfZrpdpcAN,
                                static_cast<MWTensor<float> *>(maxpoolLayer->getOutputTensor(1))->getData(),
                                edQOkUJIZbwzEeIcCLzG);
    }
    return;
}

void MWMaxPoolingLayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(muwRQxtWMMXAPxSuMYBw));
}

float *
MWMaxPoolingLayerImpl::getIndexData() {
    return
            static_cast<MWTensor<float> *>(getLayer()->getOutputTensor(1))->getData();
}

void __global__ __launch_bounds__(1024) MWSetDyForBackPropImpl(float *
PVBPDNaynqYkBlDZgXgj, const int fOpFYwKNwIfWjnPzNuob) {
    for (int i = blockDim.x *
                 blockIdx.x + threadIdx.x; i < fOpFYwKNwIfWjnPzNuob; i += blockDim.x * gridDim.x) {
        PVBPDNaynqYkBlDZgXgj[i] = i + 1;
    }
}

void __global__ __launch_bounds__(1024)
doMWMaxPoolingLayerImpl(float *UdmcwaUkepxfZrpdpcAN, float *
UWAGLbDcvybdWBtshhsr, const int BkwhtPQUCQKchmmimoXs) {
    for (int i = blockDim.x *
                 blockIdx.x + threadIdx.x; i < BkwhtPQUCQKchmmimoXs; i += blockDim.x * gridDim.x) {
        if
                (static_cast<int>(UdmcwaUkepxfZrpdpcAN[i]) != 0) {
            UWAGLbDcvybdWBtshhsr[static_cast<int>(UdmcwaUkepxfZrpdpcAN[i]) - 1] =
                    i;
        }
    }
}

MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *
ntwk_impl, int XNZmftADYzuZnIYIpBaT, int lteHjcLsItGbVPMQtGDB, const char *
xHViLEwTujGGrPZZgmbF, const char *JwxFdqOKggeawILBfGgg) :
        MWCNNLayerImpl(layer, ntwk_impl),
        CDJtexcMbXMWAmnNZsNf(XNZmftADYzuZnIYIpBaT),
        CGbFsczkgkhjcHoCKzBx(lteHjcLsItGbVPMQtGDB), vpXxoeEhdEosLSsYXkNG(NULL),
        wJyXsrUCMgxdIKVIJSyx(NULL), IwKnaBoXVubIRYcxEJLH(NULL),
        xHiBGayUfxIpXKkCTDNU(false) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&JgLfgHrHMEMmMYTettJF));
    createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
    CUDA_CALL(cudaMalloc((void **) &vpXxoeEhdEosLSsYXkNG,
                         sizeof(float) * CDJtexcMbXMWAmnNZsNf * CGbFsczkgkhjcHoCKzBx));
    CUDA_CALL(cudaMalloc((void **) &IwKnaBoXVubIRYcxEJLH,
                         sizeof(float) * CGbFsczkgkhjcHoCKzBx));
    wJyXsrUCMgxdIKVIJSyx =
            MALLOC_CALL(sizeof(float) * CDJtexcMbXMWAmnNZsNf * CGbFsczkgkhjcHoCKzBx);
    loadWeights(xHViLEwTujGGrPZZgmbF);
    loadBias(JwxFdqOKggeawILBfGgg);
}

MWFCLayerImpl::~MWFCLayerImpl() {}

void MWFCLayerImpl::propagateSize() {
    MWFCLayer *fcLayer = static_cast<MWFCLayer *>(getLayer());
    MWTensorBase *
            opTensor = fcLayer->getOutputTensor(0);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc, static_cast<MWTensor<float> *>(opTensor));
    if
            (opTensor->getSequenceLength() == 1) {
        CUDNN_CALL(cudnnSetTensor4dDescriptor(JgLfgHrHMEMmMYTettJF, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, 1, CGbFsczkgkhjcHoCKzBx, 1, 1));
    } else {
        int dims[5]
                = {1, 1, CGbFsczkgkhjcHoCKzBx, 1, 1};
        int strides[5];
        MWTensorBase::getStrides(dims, 5, strides);
        CUDNN_CALL(cudnnSetTensorNdDescriptor(JgLfgHrHMEMmMYTettJF, CUDNN_DATA_FLOAT, 5,
                                              dims, strides));
    }
}

void MWFCLayerImpl::loadWeights(const char *
QTXuPiGKeBUnmRzhlIDp) {
    FILE *QjgQHaUACFNSteMrRtRj =
            MWCNNLayer::openBinaryFile(QTXuPiGKeBUnmRzhlIDp);
    assert(QjgQHaUACFNSteMrRtRj);
    int
            dkLDkRwCBjeybwDHbKiE = CDJtexcMbXMWAmnNZsNf * CGbFsczkgkhjcHoCKzBx;
    call_fread(wJyXsrUCMgxdIKVIJSyx, sizeof(float), dkLDkRwCBjeybwDHbKiE,
               QjgQHaUACFNSteMrRtRj, QTXuPiGKeBUnmRzhlIDp);
    fclose(QjgQHaUACFNSteMrRtRj);
}

void
MWFCLayerImpl::prepareWeights() {
    if (!xHiBGayUfxIpXKkCTDNU) {
        int
                dkLDkRwCBjeybwDHbKiE = CDJtexcMbXMWAmnNZsNf * CGbFsczkgkhjcHoCKzBx;
        MWFCLayer *fcLayer = static_cast<MWFCLayer *>(getLayer());
        MWTensorBase *
                ipTensor = fcLayer->getInputTensor(0);
        if (ipTensor->getHeight() != 1 &&
            ipTensor->getWidth() != 1) {
            float *KZWeXiYFmdpQdsgidKeG =
                    MALLOC_CALL(sizeof(float) * ipTensor->getHeight() * ipTensor->getWidth());
            for (int
                         k = 0; k < dkLDkRwCBjeybwDHbKiE / ipTensor->getHeight() / ipTensor->getWidth(); k++) {
                for (int i = 0; i < ipTensor->getHeight() * ipTensor->getWidth(); i++)
                    KZWeXiYFmdpQdsgidKeG[i] = wJyXsrUCMgxdIKVIJSyx[k * ipTensor->getHeight() * ipTensor->getWidth() +
                                                                   i];
                for (int j = 0; j < ipTensor->getHeight(); j++)
                    for (int i = 0; i < ipTensor->getWidth();
                         i++)
                        wJyXsrUCMgxdIKVIJSyx[k * ipTensor->getHeight() * ipTensor->getWidth() +
                                             j * ipTensor->getWidth() + i] = KZWeXiYFmdpQdsgidKeG[j + i *
                                                                                                      ipTensor->getHeight()];
            }
            free(KZWeXiYFmdpQdsgidKeG);
        }
        CUDA_CALL(cudaMemcpy(vpXxoeEhdEosLSsYXkNG,
                             wJyXsrUCMgxdIKVIJSyx, sizeof(float) * dkLDkRwCBjeybwDHbKiE,
                             cudaMemcpyHostToDevice));
        free(wJyXsrUCMgxdIKVIJSyx);
        wJyXsrUCMgxdIKVIJSyx = NULL;
        xHiBGayUfxIpXKkCTDNU = true;
    }
}

void
MWFCLayerImpl::loadBias(const char *QTXuPiGKeBUnmRzhlIDp) {
    MWFCLayer *fcLayer =
            static_cast<MWFCLayer *>(getLayer());
    MWTensorBase *opTensor =
            fcLayer->getOutputTensor(0);
    FILE *QjgQHaUACFNSteMrRtRj =
            MWCNNLayer::openBinaryFile(QTXuPiGKeBUnmRzhlIDp);
    assert(QjgQHaUACFNSteMrRtRj);
    int
            dkLDkRwCBjeybwDHbKiE = CGbFsczkgkhjcHoCKzBx;
    float *KZWeXiYFmdpQdsgidKeG =
            MALLOC_CALL(sizeof(float) * dkLDkRwCBjeybwDHbKiE);
    call_fread(KZWeXiYFmdpQdsgidKeG,
               sizeof(float), dkLDkRwCBjeybwDHbKiE, QjgQHaUACFNSteMrRtRj, QTXuPiGKeBUnmRzhlIDp);
    CUDA_CALL(cudaMemcpy(IwKnaBoXVubIRYcxEJLH, KZWeXiYFmdpQdsgidKeG,
                         sizeof(float) * dkLDkRwCBjeybwDHbKiE, cudaMemcpyHostToDevice));
    free(KZWeXiYFmdpQdsgidKeG);
    fclose(QjgQHaUACFNSteMrRtRj);
}

void
MWFCLayerImpl::postSetup() { prepareWeights(); }

void MWFCLayerImpl::predict() {
    MWFCLayer *fcLayer = static_cast<MWFCLayer *>(getLayer());
    MWTensorBase *
            ipTensorBase = fcLayer->getInputTensor(0);
    MWTensorBase *opTensorBase =
            fcLayer->getOutputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    int numOutputRows =
            opTensor->getChannels();
    int numOutputCols =
            ipTensor->getBatchSize() * ipTensor->getSequenceLength();
    int innerDimension =
            ipTensor->getHeight() * ipTensor->getWidth() * ipTensor->getChannels();
    int
            URgvgDXnZskIYGdtimcU = 1;
    int UVzBVEOIylFjkSgHwFMp = 1;
    if (opTensor->getBatchSize() == 1 &&
        opTensor->getSequenceLength() == 1) {
        CUDA_CALL(cudaMemcpy(opTensor->getData(),
                             IwKnaBoXVubIRYcxEJLH, sizeof(float) * numOutputRows, cudaMemcpyDeviceToDevice));
        CUBLAS_CALL(cublasSgemv(*dMxIKDGTITyhdLqIHBLA->getCublasHandle(), CUBLAS_OP_T,
                                innerDimension, numOutputRows, getOnePtr(), vpXxoeEhdEosLSsYXkNG, innerDimension,
                                ipTensor->getData(), URgvgDXnZskIYGdtimcU, getOnePtr(), opTensor->getData(),
                                UVzBVEOIylFjkSgHwFMp));
    } else {
        CUBLAS_CALL(cublasSgemm(*dMxIKDGTITyhdLqIHBLA->getCublasHandle(), CUBLAS_OP_T,
                                CUBLAS_OP_N, numOutputRows, numOutputCols, innerDimension, getOnePtr(),
                                vpXxoeEhdEosLSsYXkNG, innerDimension, ipTensor->getData(), innerDimension,
                                getZeroPtr(), opTensor->getData(), numOutputRows));
        cudnnTensorDescriptor_t *
                desc = getDescriptor(opTensor->getSourcePortIndex());
        assert(desc);
        CUDNN_CALL(cudnnAddTensor(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(), getOnePtr(),
                                  JgLfgHrHMEMmMYTettJF, IwKnaBoXVubIRYcxEJLH, getOnePtr(), *desc, opTensor->getData()));
    }
    return;
}

void MWFCLayerImpl::cleanup() {
    if (vpXxoeEhdEosLSsYXkNG) {
        CUDA_FREE_CALL(vpXxoeEhdEosLSsYXkNG);
        vpXxoeEhdEosLSsYXkNG = NULL;
    }
    CUDNN_CALL(cudnnDestroyTensorDescriptor(JgLfgHrHMEMmMYTettJF));
    if
            (IwKnaBoXVubIRYcxEJLH) {
        CUDA_FREE_CALL(IwKnaBoXVubIRYcxEJLH);
        IwKnaBoXVubIRYcxEJLH = NULL;
    }
}

MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *
ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&shEncNmxJsMuJKwbrwok));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&sjLjZacPSDNBEjAccrGU));
}

MWSoftmaxLayerImpl::~MWSoftmaxLayerImpl() {}

void
MWSoftmaxLayerImpl::propagateSize() {
    MWSoftmaxLayer *sfmxLayer =
            static_cast<MWSoftmaxLayer *>(getLayer());
    MWTensorBase *ipTensor =
            sfmxLayer->getInputTensor(0);
    MWTensorBase *opTensor =
            sfmxLayer->getOutputTensor(0);
    CUDNN_CALL(cudnnSetTensor4dDescriptor(shEncNmxJsMuJKwbrwok, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, ipTensor->getSequenceLength() * ipTensor->getBatchSize(),
                                          ipTensor->getChannels(), ipTensor->getHeight(), ipTensor->getWidth()));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(sjLjZacPSDNBEjAccrGU, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, opTensor->getSequenceLength() * opTensor->getBatchSize(),
                                          opTensor->getChannels(), opTensor->getHeight(), opTensor->getWidth()));
}

void
MWSoftmaxLayerImpl::predict() {
    MWSoftmaxLayer *sfmxLayer =
            static_cast<MWSoftmaxLayer *>(getLayer());
    MWTensorBase *ipTensorBase =
            sfmxLayer->getInputTensor(0);
    MWTensorBase *opTensorBase =
            sfmxLayer->getOutputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    CUDNN_CALL(cudnnSoftmaxForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                   CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, getOnePtr(),
                                   shEncNmxJsMuJKwbrwok, ipTensor->getData(), getZeroPtr(),
                                   sjLjZacPSDNBEjAccrGU, opTensor->getData()));
}

void
MWSoftmaxLayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(shEncNmxJsMuJKwbrwok));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(sjLjZacPSDNBEjAccrGU));
}

MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer *layer,
                                             MWTargetNetworkImpl *ntwk_impl, int DqxLTLaJwwgQqmrtCDuu, int
                                             EvebzoroiuKkIxwjkGnD, int FshVHIJMRAhtQirYPlZd, int
                                             GDRXdUDklKFEYEfifhIH, int CpMjJjtGOeWOzwxpAAQP, int
                                             ClEhcJFlvGCgiavziIag, int DCdZnqpcBnvXVgEsLBnz, int
                                             DGzdAcREJHGXjyRzNjJV) : MWCNNLayerImpl(layer, ntwk_impl),
                                                                     TbrNrGxaFFHrzKUcfHNZ(NULL),
                                                                     DSsxcjIrUgZCKZovyNQf(DqxLTLaJwwgQqmrtCDuu),
                                                                     EfvWctmlsWAPsxXgdKWf(EvebzoroiuKkIxwjkGnD),
                                                                     DRzwhbNPpftRRIXXfHzd(DqxLTLaJwwgQqmrtCDuu),
                                                                     ECTnqgWHyHCHCLBZlffd(EvebzoroiuKkIxwjkGnD),
                                                                     FrpxvsDMwwgbpqHXWxmN(FshVHIJMRAhtQirYPlZd),
                                                                     FwLnexHgxHRquTKmNpoa(GDRXdUDklKFEYEfifhIH),
                                                                     CZNYmBcNFSZWvaCklqeM(CpMjJjtGOeWOzwxpAAQP),
                                                                     CTCbzQMDaLxINPbODdng(ClEhcJFlvGCgiavziIag),
                                                                     CqtPRJvHlGJFssiPzsOm(DCdZnqpcBnvXVgEsLBnz),
                                                                     CufLFODQDXTAPyRqYodN(DGzdAcREJHGXjyRzNjJV),
                                                                     IIiwAtyrOtLzLWAUlTey((CZNYmBcNFSZWvaCklqeM !=
                                                                                           CTCbzQMDaLxINPbODdng)
                                                                                          || (CqtPRJvHlGJFssiPzsOm !=
                                                                                              CufLFODQDXTAPyRqYodN)),
                                                                     nDsbARncmIrIaLubvLVZ(CpMjJjtGOeWOzwxpAAQP),
                                                                     nNULvWnBXnnWdpEkHPAH(DCdZnqpcBnvXVgEsLBnz) {
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&muwRQxtWMMXAPxSuMYBw));
    MWTensorBase *
            ipTensor = getLayer()->getInputTensor(0);
    if (IIiwAtyrOtLzLWAUlTey) {
        nDsbARncmIrIaLubvLVZ = 0;
        nNULvWnBXnnWdpEkHPAH = 0;
        TbrNrGxaFFHrzKUcfHNZ = new MWTensor<float>(-1, -1, -1, -1, -1, NULL, getLayer(), 0);
        if (!TbrNrGxaFFHrzKUcfHNZ) {
            MWCNNLayerImpl::throwAllocationError(__LINE__,
                                                 __FILE__);
        }
        CUDNN_CALL(cudnnCreateTensorDescriptor(&XhAYHFyEVtlwoxGBuTpu));
    } else { TbrNrGxaFFHrzKUcfHNZ = ipTensor; }
    assert(TbrNrGxaFFHrzKUcfHNZ != NULL);
    MWAvgPoolingLayer *avgpoolLayer = static_cast<MWAvgPoolingLayer *>(getLayer());
    MWTensorBase *opTensor = avgpoolLayer->getOutputTensor(0);
    createAndAddDescriptor(opTensor->getSourcePortIndex());
}

MWAvgPoolingLayerImpl::~MWAvgPoolingLayerImpl() {}

void
MWAvgPoolingLayerImpl::propagateSize() {
    MWTensorBase *ipTensor =
            getLayer()->getInputTensor(0);
    if ((DSsxcjIrUgZCKZovyNQf == -1) &&
        (EfvWctmlsWAPsxXgdKWf == -1)) {
        DRzwhbNPpftRRIXXfHzd = ipTensor->getHeight();
        ECTnqgWHyHCHCLBZlffd = ipTensor->getWidth();
    }
    int inputH;
    int inputW;
    if
            (IIiwAtyrOtLzLWAUlTey) {
        inputH = ipTensor->getHeight() +
                 CZNYmBcNFSZWvaCklqeM + CTCbzQMDaLxINPbODdng;
        inputW = ipTensor->getWidth() +
                 CqtPRJvHlGJFssiPzsOm + CufLFODQDXTAPyRqYodN;
    } else {
        inputH =
                ipTensor->getHeight();
        inputW = ipTensor->getWidth();
    }
    TbrNrGxaFFHrzKUcfHNZ->setHeight(inputH);
    TbrNrGxaFFHrzKUcfHNZ->setWidth(inputW);
    TbrNrGxaFFHrzKUcfHNZ->setChannels(ipTensor->getChannels());
    TbrNrGxaFFHrzKUcfHNZ->setBatchSize(ipTensor->getBatchSize());
    TbrNrGxaFFHrzKUcfHNZ->setSequenceLength(ipTensor->getSequenceLength());
    assert(TbrNrGxaFFHrzKUcfHNZ->getSequenceLength() == 1);
    if
            (IIiwAtyrOtLzLWAUlTey) {
        CUDNN_CALL(cudnnSetTensor4dDescriptor(XhAYHFyEVtlwoxGBuTpu, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, TbrNrGxaFFHrzKUcfHNZ->getBatchSize(),
                                              TbrNrGxaFFHrzKUcfHNZ->getChannels(),
                                              TbrNrGxaFFHrzKUcfHNZ->getHeight(), TbrNrGxaFFHrzKUcfHNZ->getWidth()));
    } else {
        XhAYHFyEVtlwoxGBuTpu = MWCNNLayerImpl::getCuDNNDescriptor(TbrNrGxaFFHrzKUcfHNZ);
    }
    MWTensorBase *opTensor = getLayer()->getOutputTensor(0);
    CUDNN_CALL(cudnnSetPooling2dDescriptor(muwRQxtWMMXAPxSuMYBw,
                                           CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
                                           DRzwhbNPpftRRIXXfHzd, ECTnqgWHyHCHCLBZlffd, nDsbARncmIrIaLubvLVZ,
                                           nNULvWnBXnnWdpEkHPAH, FrpxvsDMwwgbpqHXWxmN, FwLnexHgxHRquTKmNpoa));
    cudnnTensorDescriptor_t *desc = getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc,
                         static_cast<MWTensor<float> *>(opTensor));
}

void
MWAvgPoolingLayerImpl::allocate() {
    MWTensorBase *ipTensor =
            getLayer()->getInputTensor(0);
    if (IIiwAtyrOtLzLWAUlTey) {
        float *
                newInput;
        int inputH = ipTensor->getHeight() + CZNYmBcNFSZWvaCklqeM +
                     CTCbzQMDaLxINPbODdng;
        int inputW = ipTensor->getWidth() +
                     CqtPRJvHlGJFssiPzsOm + CufLFODQDXTAPyRqYodN;
        int paddedSize =
                ipTensor->getBatchSize() * ipTensor->getChannels() * inputH * inputW;
        CUDA_CALL(cudaMalloc((void **) &newInput, sizeof(float) * paddedSize));
        CUDA_CALL(cudaMemset(newInput, 0, sizeof(float) * paddedSize));
        static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->setData(newInput);
    }
}

void
MWAvgPoolingLayerImpl::deallocate() {
    if (TbrNrGxaFFHrzKUcfHNZ !=
        getLayer()->getInputTensor(0)) {
        assert(IIiwAtyrOtLzLWAUlTey);
        CUDA_FREE_CALL(static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->getData());
        static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->setData((float *) NULL);
    }
}

void
MWAvgPoolingLayerImpl::predict() {
    MWAvgPoolingLayer *avgpoolLayer =
            static_cast<MWAvgPoolingLayer *>(getLayer());
    MWTensorBase *opTensorBase =
            avgpoolLayer->getOutputTensor(0);
    MWTensorBase *ipTensorBase =
            avgpoolLayer->getInputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    if (TbrNrGxaFFHrzKUcfHNZ !=
        avgpoolLayer->getInputTensor()) {
        CUDA_CALL(cudaMemset(static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->getData(),
                             0, sizeof(float) * TbrNrGxaFFHrzKUcfHNZ->getNumElements()));
        MWCNNLayerImpl::padInput(ipTensor->getData(), ipTensor->getHeight(),
                                 ipTensor->getWidth(), ipTensor->getChannels(), TbrNrGxaFFHrzKUcfHNZ->getHeight(),
                                 TbrNrGxaFFHrzKUcfHNZ->getWidth(), CZNYmBcNFSZWvaCklqeM, CqtPRJvHlGJFssiPzsOm,
                                 static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->getData(),
                                 ipTensor->getNumElements());
    }
    assert(opTensor->getData() !=
           static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->getData());
    cudnnTensorDescriptor_t *desc = getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    CUDNN_CALL(cudnnPoolingForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                   muwRQxtWMMXAPxSuMYBw, getOnePtr(), XhAYHFyEVtlwoxGBuTpu,
                                   static_cast<MWTensor<float> *>(TbrNrGxaFFHrzKUcfHNZ)->getData(), getZeroPtr(), *desc,
                                   opTensor->getData()));
}

void MWAvgPoolingLayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(muwRQxtWMMXAPxSuMYBw));
    if
            (TbrNrGxaFFHrzKUcfHNZ != getLayer()->getInputTensor(0)) {
        assert(IIiwAtyrOtLzLWAUlTey);
        CUDNN_CALL(cudnnDestroyTensorDescriptor(XhAYHFyEVtlwoxGBuTpu));
    }
}

MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer *layer, MWTargetNetworkImpl *
ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) {}

MWOutputLayerImpl::~MWOutputLayerImpl() {}

void
MWOutputLayerImpl::propagateSize() {}

void
MWOutputLayerImpl::deallocateOutputData(int) {}

void
MWOutputLayerImpl::predict() {}

void MWOutputLayerImpl::cleanup() {}