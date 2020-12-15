#include "MWElementwiseAffineLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "MWKernelHeaders.hpp"
#include "cnn_api.hpp"
#include <math.h>
#include <cassert>
#include <stdio.h>

MWElementwiseAffineLayerImpl::MWElementwiseAffineLayerImpl(MWCNNLayer *layer,
                                                           MWTargetNetworkImpl *ntwk_impl, int scale_H, int scale_W,
                                                           int scale_C, int
                                                           offset_H, int offset_W, int offset_C, bool isClipped,
                                                           int lowerbound, int
                                                           upperbound, const char *rISNTTiSXOTdHqHTtNiB, const char *
iADjqLChtuDbEWfMYFLp) : MWCNNLayerImpl(layer, ntwk_impl),
                        pvpNsgGssdTxeVoFIkXI(NULL), gNROjwaqhxDPvBWUCUcQ(NULL), qBTcAwVGZERyCjGYByPe(scale_H),
                        qWwjVYwfnvEnFKlgpqwA(scale_W), pzUAoBDvaKAtdsmkQuct(scale_C),
                        hljcfGWsvZXJZNrImpJB(offset_H), hvqKUzPqCuUJRfoNlbwW(offset_W),
                        hKyfKjPACkOBDvLdESxH(offset_C), ZUTPCvgISoRdtnhGqXzM(isClipped),
                        bQjijJlpNAVdwDDQgpaX(lowerbound), veFyKKHbdqBIvQLYBqfF(upperbound) {
    CUDA_CALL(cudaMalloc((void **) &pvpNsgGssdTxeVoFIkXI,
                         sizeof(float) * qBTcAwVGZERyCjGYByPe * qWwjVYwfnvEnFKlgpqwA * pzUAoBDvaKAtdsmkQuct));
    CUDA_CALL(cudaMalloc((void **) &gNROjwaqhxDPvBWUCUcQ,
                         sizeof(float) * hljcfGWsvZXJZNrImpJB * hvqKUzPqCuUJRfoNlbwW * hKyfKjPACkOBDvLdESxH));
    loadScale(rISNTTiSXOTdHqHTtNiB);
    loadOffset(iADjqLChtuDbEWfMYFLp);
}

MWElementwiseAffineLayerImpl::~MWElementwiseAffineLayerImpl() {}

void
MWElementwiseAffineLayerImpl::propagateSize() {}

void
MWElementwiseAffineLayerImpl::predict() {
    MWTensorBase *ipTensorBase =
            getLayer()->getInputTensor(0);
    MWTensorBase *opTensorBase =
            getLayer()->getOutputTensor(0);
    MWTensor<float> *ipTensor =
            static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    int WmXADZOqdcQvtBUvFerh =
            ipTensor->getHeight();
    int WprSrhAStKGxyXeoxETy = ipTensor->getWidth();
    int
            WerBmCOBWhvoFbdqfitc = ipTensor->getChannels();
    long int
            YOWMnLKOMqAODXiVNoGy = WmXADZOqdcQvtBUvFerh * WprSrhAStKGxyXeoxETy;
    long
    int YNmJhGSUszJKxsodxiuV =
            YOWMnLKOMqAODXiVNoGy * WerBmCOBWhvoFbdqfitc;
    long int
            YNDVziqpDddiXQKYZZhX = ipTensor->getNumElements();
    long int sFIUeCwGDlfadqOrGZHC =
            ((YNDVziqpDddiXQKYZZhX + 31) / 32) * 32;
    int tGsvtyAVkrDznETdweDC =
            (sFIUeCwGDlfadqOrGZHC < 1024) ? sFIUeCwGDlfadqOrGZHC : 1024;
    long int
            KHClOltUSuqFVVErSxVb = (YNDVziqpDddiXQKYZZhX + tGsvtyAVkrDznETdweDC -
                                    1) / tGsvtyAVkrDznETdweDC;
    long int qEXwbWWsnOADJeTXfRVa =
            qBTcAwVGZERyCjGYByPe * qWwjVYwfnvEnFKlgpqwA * pzUAoBDvaKAtdsmkQuct;
    long int
            hnewnpwgzKmOdualajhn = hljcfGWsvZXJZNrImpJB * hvqKUzPqCuUJRfoNlbwW *
                                   hKyfKjPACkOBDvLdESxH;
    assert(qEXwbWWsnOADJeTXfRVa <= YNDVziqpDddiXQKYZZhX);
    assert(hnewnpwgzKmOdualajhn <= YNDVziqpDddiXQKYZZhX);
    if (qEXwbWWsnOADJeTXfRVa ==
        1) {
        scale_scalar_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(ipTensor->getData(), opTensor->getData(),
                                pvpNsgGssdTxeVoFIkXI, YNDVziqpDddiXQKYZZhX);
    } else if (qBTcAwVGZERyCjGYByPe == 1 &&
               qWwjVYwfnvEnFKlgpqwA == 1 && qEXwbWWsnOADJeTXfRVa > 1) {
        scale_vector_kernel<<<KHClOltUSuqFVVErSxVb, tGsvtyAVkrDznETdweDC>>>(
                ipTensor->getData(), opTensor->getData(), pvpNsgGssdTxeVoFIkXI,
                YOWMnLKOMqAODXiVNoGy, YNmJhGSUszJKxsodxiuV,
                YNDVziqpDddiXQKYZZhX);
    } else if (YNmJhGSUszJKxsodxiuV ==
               qEXwbWWsnOADJeTXfRVa) {
        scale_tensor3d_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(ipTensor->getData(), opTensor->getData(),
                                pvpNsgGssdTxeVoFIkXI, YNmJhGSUszJKxsodxiuV, YNDVziqpDddiXQKYZZhX);
    } else {
        scale_matrix2d_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(ipTensor->getData(), opTensor->getData(),
                                pvpNsgGssdTxeVoFIkXI, YOWMnLKOMqAODXiVNoGy, YNDVziqpDddiXQKYZZhX);
    }
    if
            (hnewnpwgzKmOdualajhn == 1) {
        offset_scalar_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(opTensor->getData(), opTensor->getData(),
                                gNROjwaqhxDPvBWUCUcQ, YNDVziqpDddiXQKYZZhX, ZUTPCvgISoRdtnhGqXzM,
                                bQjijJlpNAVdwDDQgpaX, veFyKKHbdqBIvQLYBqfF);
    } else if (hljcfGWsvZXJZNrImpJB
               == 1 && hvqKUzPqCuUJRfoNlbwW == 1 && hnewnpwgzKmOdualajhn > 1) {
        offset_vector_kernel<<<KHClOltUSuqFVVErSxVb, tGsvtyAVkrDznETdweDC>>>(
                opTensor->getData(), opTensor->getData(), gNROjwaqhxDPvBWUCUcQ,
                YOWMnLKOMqAODXiVNoGy, YNmJhGSUszJKxsodxiuV,
                YNDVziqpDddiXQKYZZhX, ZUTPCvgISoRdtnhGqXzM, bQjijJlpNAVdwDDQgpaX,
                veFyKKHbdqBIvQLYBqfF);
    } else if (YNmJhGSUszJKxsodxiuV ==
               hnewnpwgzKmOdualajhn) {
        offset_tensor3d_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(opTensor->getData(), opTensor->getData(),
                                gNROjwaqhxDPvBWUCUcQ, YNmJhGSUszJKxsodxiuV, YNDVziqpDddiXQKYZZhX,
                                ZUTPCvgISoRdtnhGqXzM, bQjijJlpNAVdwDDQgpaX, veFyKKHbdqBIvQLYBqfF);
    } else {
        offset_matrix2d_kernel<<<KHClOltUSuqFVVErSxVb,
        tGsvtyAVkrDznETdweDC>>>(opTensor->getData(), opTensor->getData(),
                                gNROjwaqhxDPvBWUCUcQ, YOWMnLKOMqAODXiVNoGy, YNDVziqpDddiXQKYZZhX,
                                ZUTPCvgISoRdtnhGqXzM, bQjijJlpNAVdwDDQgpaX, veFyKKHbdqBIvQLYBqfF);
    }
    return;
}

void MWElementwiseAffineLayerImpl::cleanup() {
    if (pvpNsgGssdTxeVoFIkXI) {
        CUDA_FREE_CALL(pvpNsgGssdTxeVoFIkXI);
        pvpNsgGssdTxeVoFIkXI = NULL;
    }
    if
            (gNROjwaqhxDPvBWUCUcQ) {
        CUDA_FREE_CALL(gNROjwaqhxDPvBWUCUcQ);
        gNROjwaqhxDPvBWUCUcQ =
                NULL;
    }
}

void MWElementwiseAffineLayerImpl::loadScale(const char *
rISNTTiSXOTdHqHTtNiB) {
    FILE *QjgQHaUACFNSteMrRtRj =
            MWCNNLayer::openBinaryFile(rISNTTiSXOTdHqHTtNiB);
    assert(QjgQHaUACFNSteMrRtRj);
    long
    int dkLDkRwCBjeybwDHbKiE = qBTcAwVGZERyCjGYByPe * qWwjVYwfnvEnFKlgpqwA * pzUAoBDvaKAtdsmkQuct;
    float *KZWeXiYFmdpQdsgidKeG = MALLOC_CALL(sizeof(float) * dkLDkRwCBjeybwDHbKiE);
    call_fread(KZWeXiYFmdpQdsgidKeG, sizeof(float), dkLDkRwCBjeybwDHbKiE, QjgQHaUACFNSteMrRtRj,
               rISNTTiSXOTdHqHTtNiB);
    CUDA_CALL(cudaMemcpy(pvpNsgGssdTxeVoFIkXI,
                         KZWeXiYFmdpQdsgidKeG, sizeof(float) * dkLDkRwCBjeybwDHbKiE, cudaMemcpyHostToDevice));
    free(KZWeXiYFmdpQdsgidKeG);
    fclose(QjgQHaUACFNSteMrRtRj);
}

void
MWElementwiseAffineLayerImpl::loadOffset(const char *iADjqLChtuDbEWfMYFLp) {
    FILE *QjgQHaUACFNSteMrRtRj = MWCNNLayer::openBinaryFile(iADjqLChtuDbEWfMYFLp);
    assert(QjgQHaUACFNSteMrRtRj);
    long int dkLDkRwCBjeybwDHbKiE =
            hljcfGWsvZXJZNrImpJB * hvqKUzPqCuUJRfoNlbwW * hKyfKjPACkOBDvLdESxH;
    float *
            KZWeXiYFmdpQdsgidKeG = MALLOC_CALL(sizeof(float) * dkLDkRwCBjeybwDHbKiE);
    call_fread(KZWeXiYFmdpQdsgidKeG, sizeof(float), dkLDkRwCBjeybwDHbKiE, QjgQHaUACFNSteMrRtRj,
               iADjqLChtuDbEWfMYFLp);
    CUDA_CALL(cudaMemcpy(gNROjwaqhxDPvBWUCUcQ,
                         KZWeXiYFmdpQdsgidKeG, sizeof(float) * dkLDkRwCBjeybwDHbKiE, cudaMemcpyHostToDevice));
    free(KZWeXiYFmdpQdsgidKeG);
    fclose(QjgQHaUACFNSteMrRtRj);
}