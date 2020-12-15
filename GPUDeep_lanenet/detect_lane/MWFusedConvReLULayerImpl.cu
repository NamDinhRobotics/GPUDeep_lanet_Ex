#include "MWFusedConvReLULayer.hpp"
#include "MWFusedConvReLULayerImpl.hpp"
#include <cassert>
#include <stdio.h>

MWFusedConvReLULayerImpl::MWFusedConvReLULayerImpl(MWCNNLayer *layer,
                                                   MWTargetNetworkImpl *ntwk_impl, int filt_H, int filt_W, int numGrps,
                                                   int
                                                   numChnls, int numFilts, int FshVHIJMRAhtQirYPlZd, int
                                                   GDRXdUDklKFEYEfifhIH, int CpMjJjtGOeWOzwxpAAQP, int
                                                   ClEhcJFlvGCgiavziIag, int DCdZnqpcBnvXVgEsLBnz, int
                                                   DGzdAcREJHGXjyRzNjJV, int AwZQzUhuWVLGrWgLHRuM, int
                                                   BHuHNDGoRwGRouCxeMbw, int eybNKlJCSDUvsznWynwK, const char *
xHViLEwTujGGrPZZgmbF, const char *JwxFdqOKggeawILBfGgg) :
        MWCNNLayerImpl(layer, ntwk_impl), vpXxoeEhdEosLSsYXkNG(NULL), IwKnaBoXVubIRYcxEJLH(NULL),
        UKtMXCCqdjeyaVHabkxg(NULL), XCLDbxHBtWRStETWIkId(NULL), HhKGcPZwrclEFnIdWerH(NULL),
        BLjrjqvCcCommiXWQLjs(filt_H), BRSPqxNffoBYKqpSVHne(filt_W),
        CCKWXUFWgrbBMjwfpOBN(numGrps), BlRIQPyqJZORKENzSdYf(numChnls),
        BuyZFXzwOMxcePIbCLfl(numFilts), FrpxvsDMwwgbpqHXWxmN(FshVHIJMRAhtQirYPlZd),
        FwLnexHgxHRquTKmNpoa(GDRXdUDklKFEYEfifhIH),
        CZNYmBcNFSZWvaCklqeM(CpMjJjtGOeWOzwxpAAQP),
        CTCbzQMDaLxINPbODdng(ClEhcJFlvGCgiavziIag),
        CqtPRJvHlGJFssiPzsOm(DCdZnqpcBnvXVgEsLBnz),
        CufLFODQDXTAPyRqYodN(DGzdAcREJHGXjyRzNjJV),
        AuqaQHxmPQSyYRemQvyX(AwZQzUhuWVLGrWgLHRuM),
        AzTsxYcYjIEJsGQbeYHm(BHuHNDGoRwGRouCxeMbw),
        fxxCPKTclxXPxrdMAkwi(eybNKlJCSDUvsznWynwK),
        IIiwAtyrOtLzLWAUlTey((CZNYmBcNFSZWvaCklqeM != CTCbzQMDaLxINPbODdng)
                             || (CqtPRJvHlGJFssiPzsOm != CufLFODQDXTAPyRqYodN)) {
#if (CUDNN_MAJOR < 6)
    throw std::runtime_error("Fused ConvReLU Layer only supported for cuDNN 6 or greater");
#else
    dMxIKDGTITyhdLqIHBLA = ntwk_impl;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&NXruhrCCiguRjAgSNDuz));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&QhTWatiCfcWYsHdkcyhZ));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&JgLfgHrHMEMmMYTettJF));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&oKIvzXXMucEDsTGGpdpm));
    MWTensorBase *ipTensor_conv = getLayer()->getInputTensor(0);
    int
            NZjOkZPwLzQsdEVkwMcX = CZNYmBcNFSZWvaCklqeM;
    int
            NbunkIVaMPVYgAQHXXYd = CqtPRJvHlGJFssiPzsOm;
    if
            (IIiwAtyrOtLzLWAUlTey) {
        NZjOkZPwLzQsdEVkwMcX = 0;
        NbunkIVaMPVYgAQHXXYd = 0;
        UKtMXCCqdjeyaVHabkxg = new MWTensor<float>(-1,
                                                   -1, -1, -1, -1, NULL, getLayer(), 0);
        if (!UKtMXCCqdjeyaVHabkxg) {
            MWCNNLayerImpl::throwAllocationError(__LINE__, __FILE__);
        }
        CUDNN_CALL(cudnnCreateTensorDescriptor(&XhAYHFyEVtlwoxGBuTpu));
    } else {
        UKtMXCCqdjeyaVHabkxg = ipTensor_conv;
    }
    assert(UKtMXCCqdjeyaVHabkxg != NULL);
    bYBVtTnVUuGDUlaTmmHp = CZNYmBcNFSZWvaCklqeM;
    cQBKlCKXxecGPJrXBXdk =
            CqtPRJvHlGJFssiPzsOm;
    MWFusedConvReLULayer *fusedConvReluLayer =
            static_cast<MWFusedConvReLULayer *>(getLayer());
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(NXruhrCCiguRjAgSNDuz,
                                               NZjOkZPwLzQsdEVkwMcX, NbunkIVaMPVYgAQHXXYd, FrpxvsDMwwgbpqHXWxmN,
                                               FwLnexHgxHRquTKmNpoa, AuqaQHxmPQSyYRemQvyX, AzTsxYcYjIEJsGQbeYHm,
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#if (FP16_ENABLED == 1 && (CUDNN_MAJOR > 7 || (CUDNN_MAJOR == 7 && CUDNN_MINOR >= 2)))
    CUDNN_CALL(cudnnSetConvolutionMathType(NXruhrCCiguRjAgSNDuz, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif
    if (CCKWXUFWgrbBMjwfpOBN > 1) {
        CUDNN_CALL(cudnnSetConvolutionGroupCount(NXruhrCCiguRjAgSNDuz,
                                                 CCKWXUFWgrbBMjwfpOBN));
    }
    CUDNN_CALL(cudnnSetActivationDescriptor(oKIvzXXMucEDsTGGpdpm,
                                            CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
    int
            euppfEoiaoCTcVgRPVhA = BlRIQPyqJZORKENzSdYf * CCKWXUFWgrbBMjwfpOBN;
    int
            fSbUUBgjKRbNXrHrlOLo = BuyZFXzwOMxcePIbCLfl * CCKWXUFWgrbBMjwfpOBN;
    CUDNN_CALL(cudnnSetFilter4dDescriptor(QhTWatiCfcWYsHdkcyhZ, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, fSbUUBgjKRbNXrHrlOLo,
                                          euppfEoiaoCTcVgRPVhA / CCKWXUFWgrbBMjwfpOBN, BLjrjqvCcCommiXWQLjs,
                                          BRSPqxNffoBYKqpSVHne));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(JgLfgHrHMEMmMYTettJF, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, 1, fSbUUBgjKRbNXrHrlOLo, 1, 1));
    int weightSize =
            BlRIQPyqJZORKENzSdYf * fSbUUBgjKRbNXrHrlOLo * BLjrjqvCcCommiXWQLjs * BRSPqxNffoBYKqpSVHne;
    CUDA_CALL(cudaMalloc((void **) &vpXxoeEhdEosLSsYXkNG, sizeof(float) * weightSize));
    CUDA_CALL(cudaMalloc((void **) &IwKnaBoXVubIRYcxEJLH,
                         sizeof(float) * fSbUUBgjKRbNXrHrlOLo));
    loadWeights(xHViLEwTujGGrPZZgmbF);
    loadBias(JwxFdqOKggeawILBfGgg);
    createAndAddDescriptor(getLayer()->getOutputTensor(0)->getSourcePortIndex());
#endif
}

MWFusedConvReLULayerImpl::~MWFusedConvReLULayerImpl() {}

void
MWFusedConvReLULayerImpl::propagateSize() {
#if (CUDNN_MAJOR >= 6)
    MWTensorBase *ipTensor_conv = getLayer()->getInputTensor(0);
    int inputH;
    int
            inputW;
    if (IIiwAtyrOtLzLWAUlTey) {
        inputH =
                ipTensor_conv->getHeight() + CZNYmBcNFSZWvaCklqeM + CTCbzQMDaLxINPbODdng;
        inputW = ipTensor_conv->getWidth() + CqtPRJvHlGJFssiPzsOm +
                 CufLFODQDXTAPyRqYodN;
    } else {
        inputH = ipTensor_conv->getHeight();
        inputW =
                ipTensor_conv->getWidth();
    }
    UKtMXCCqdjeyaVHabkxg->setHeight(inputH);
    UKtMXCCqdjeyaVHabkxg->setWidth(inputW);
    UKtMXCCqdjeyaVHabkxg->setChannels(ipTensor_conv->getChannels());
    UKtMXCCqdjeyaVHabkxg->setBatchSize(ipTensor_conv->getBatchSize());
    UKtMXCCqdjeyaVHabkxg->setSequenceLength(ipTensor_conv->getSequenceLength());
    assert(UKtMXCCqdjeyaVHabkxg->getSequenceLength() == 1);
    if
            (IIiwAtyrOtLzLWAUlTey) {
        CUDNN_CALL(cudnnSetTensor4dDescriptor(XhAYHFyEVtlwoxGBuTpu, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, UKtMXCCqdjeyaVHabkxg->getBatchSize(),
                                              UKtMXCCqdjeyaVHabkxg->getChannels(), UKtMXCCqdjeyaVHabkxg->getHeight(),
                                              UKtMXCCqdjeyaVHabkxg->getWidth()));
    } else {
        XhAYHFyEVtlwoxGBuTpu =
                MWCNNLayerImpl::getCuDNNDescriptor(UKtMXCCqdjeyaVHabkxg);
    }
    assert(BlRIQPyqJZORKENzSdYf ==
           UKtMXCCqdjeyaVHabkxg->getChannels() / CCKWXUFWgrbBMjwfpOBN);
    MWTensorBase *opTensor
            = getLayer()->getOutputTensor(0);
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
    setDescriptor<float>(*desc, static_cast<MWTensor<float> *>(opTensor));
#if (CUDNN_MAJOR < 7)
    {
   CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
   XhAYHFyEVtlwoxGBuTpu, QhTWatiCfcWYsHdkcyhZ, NXruhrCCiguRjAgSNDuz, *desc,
   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &NMMfJylfQjiIUAKhXCJb)); }
#else
    {
        cudnnConvolutionFwdAlgoPerf_t perf_results[3];
        int returnedAlgoCount;
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                                          XhAYHFyEVtlwoxGBuTpu, QhTWatiCfcWYsHdkcyhZ,
                                                          NXruhrCCiguRjAgSNDuz, *desc, 3,
                                                          &returnedAlgoCount, perf_results));
        NMMfJylfQjiIUAKhXCJb = perf_results[0].algo;
    }
#endif
    if (CUDNN_VERSION < 7402) fixConvAlgo();
    size_t tnTPxeDjBsqLAPkJcPJX = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                                       XhAYHFyEVtlwoxGBuTpu, QhTWatiCfcWYsHdkcyhZ, NXruhrCCiguRjAgSNDuz,
                                                       *desc,
                                                       NMMfJylfQjiIUAKhXCJb, &tnTPxeDjBsqLAPkJcPJX));
    if (tnTPxeDjBsqLAPkJcPJX >
        *dMxIKDGTITyhdLqIHBLA->getProposedWorkSpaceSize()) {
        dMxIKDGTITyhdLqIHBLA->setProposedWorkSpaceSize(tnTPxeDjBsqLAPkJcPJX);
    }
#endif
}

void MWFusedConvReLULayerImpl::allocate() {
    MWTensorBase *ipTensor_conv =
            getLayer()->getInputTensor(0);
    if (IIiwAtyrOtLzLWAUlTey) {
        float *
                newInput;
        int inputH = ipTensor_conv->getHeight() + CZNYmBcNFSZWvaCklqeM +
                     CTCbzQMDaLxINPbODdng;
        int inputW = ipTensor_conv->getWidth() +
                     CqtPRJvHlGJFssiPzsOm + CufLFODQDXTAPyRqYodN;
        int paddedSize =
                ipTensor_conv->getBatchSize() * ipTensor_conv->getChannels() * inputH * inputW;
        CUDA_CALL(cudaMalloc((void **) &newInput, sizeof(float) * paddedSize));
        CUDA_CALL(cudaMemset(newInput, 0, sizeof(float) * paddedSize));
        static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->setData(newInput);
    }
    XCLDbxHBtWRStETWIkId =
            static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData();
    setalpha2Ptr(getZeroPtr());
    int numInputs = getLayer()->getNumInputs();
    if
            (numInputs == 2) {
        setalpha2Ptr(getOnePtr());
        XCLDbxHBtWRStETWIkId =
                static_cast<MWTensor<float> *>(getLayer()->getInputTensor(1))->getData();
    }
}

void MWFusedConvReLULayerImpl::deallocate() {
    if (UKtMXCCqdjeyaVHabkxg !=
        getLayer()->getInputTensor(0)) {
        assert(IIiwAtyrOtLzLWAUlTey);
        CUDA_FREE_CALL(static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData());
        static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->setData((float *) NULL);
    }
}

void MWFusedConvReLULayerImpl::predict() {
    MWFusedConvReLULayer *
            fusedConvReluLayer = static_cast<MWFusedConvReLULayer *>(getLayer());
    MWTensorBase *ipTensorBase = fusedConvReluLayer->getInputTensor();
    MWTensorBase *opTensorBase = fusedConvReluLayer->getOutputTensor();
    MWTensor<float> *ipTensor = static_cast<MWTensor<float> *>(ipTensorBase);
    MWTensor<float> *opTensor = static_cast<MWTensor<float> *>(opTensorBase);
    if
            (UKtMXCCqdjeyaVHabkxg != fusedConvReluLayer->getInputTensor()) {
        CUDA_CALL(cudaMemset(static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData(),
                             0, sizeof(float) * UKtMXCCqdjeyaVHabkxg->getNumElements()));
        MWCNNLayerImpl::padInput(ipTensor->getData(), ipTensor->getHeight(),
                                 ipTensor->getWidth(), ipTensor->getChannels(), UKtMXCCqdjeyaVHabkxg->getHeight(),
                                 UKtMXCCqdjeyaVHabkxg->getWidth(), bYBVtTnVUuGDUlaTmmHp, cQBKlCKXxecGPJrXBXdk,
                                 static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData(),
                                 ipTensor->getNumElements());
    }
    cudnnTensorDescriptor_t *desc =
            getDescriptor(opTensor->getSourcePortIndex());
    assert(desc);
#if (CUDNN_MAJOR >= 6)
    assert(opTensor->getData() !=
           static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData() ||
           (getLayer()->getNumInputs() == 2));
    CUDNN_CALL(cudnnConvolutionBiasActivationForward(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                                     getOnePtr(), XhAYHFyEVtlwoxGBuTpu,
                                                     static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData(),
                                                     QhTWatiCfcWYsHdkcyhZ, vpXxoeEhdEosLSsYXkNG, NXruhrCCiguRjAgSNDuz,
                                                     NMMfJylfQjiIUAKhXCJb,
                                                     dMxIKDGTITyhdLqIHBLA->getWorkSpace(),
                                                     *dMxIKDGTITyhdLqIHBLA->getAllocatedWorkSpaceSize(), getalpha2Ptr(),
                                                     *desc,
                                                     XCLDbxHBtWRStETWIkId, JgLfgHrHMEMmMYTettJF, IwKnaBoXVubIRYcxEJLH,
                                                     oKIvzXXMucEDsTGGpdpm,
                                                     *desc, opTensor->getData()));
#endif
}

void MWFusedConvReLULayerImpl::cleanup() {
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(NXruhrCCiguRjAgSNDuz));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(QhTWatiCfcWYsHdkcyhZ));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(oKIvzXXMucEDsTGGpdpm));
    if
            (vpXxoeEhdEosLSsYXkNG) {
        CUDA_FREE_CALL(vpXxoeEhdEosLSsYXkNG);
        vpXxoeEhdEosLSsYXkNG = NULL;
    }
    CUDNN_CALL(cudnnDestroyTensorDescriptor(JgLfgHrHMEMmMYTettJF));
    if
            (IwKnaBoXVubIRYcxEJLH) {
        CUDA_FREE_CALL(IwKnaBoXVubIRYcxEJLH);
        IwKnaBoXVubIRYcxEJLH = NULL;
    }
    if
            (UKtMXCCqdjeyaVHabkxg != getLayer()->getInputTensor(0)) {
        assert(IIiwAtyrOtLzLWAUlTey);
        CUDNN_CALL(cudnnDestroyTensorDescriptor(XhAYHFyEVtlwoxGBuTpu));
    }
}

void
MWFusedConvReLULayerImpl::loadWeights(const char *QTXuPiGKeBUnmRzhlIDp) {
    FILE *
            QjgQHaUACFNSteMrRtRj = MWCNNLayer::openBinaryFile(QTXuPiGKeBUnmRzhlIDp);
    assert(QjgQHaUACFNSteMrRtRj);
    int dkLDkRwCBjeybwDHbKiE =
            BlRIQPyqJZORKENzSdYf * CCKWXUFWgrbBMjwfpOBN * BuyZFXzwOMxcePIbCLfl * BLjrjqvCcCommiXWQLjs *
            BRSPqxNffoBYKqpSVHne;
    float *KZWeXiYFmdpQdsgidKeG = MALLOC_CALL(sizeof(float) * dkLDkRwCBjeybwDHbKiE);
    call_fread(KZWeXiYFmdpQdsgidKeG, sizeof(float), dkLDkRwCBjeybwDHbKiE, QjgQHaUACFNSteMrRtRj,
               QTXuPiGKeBUnmRzhlIDp);
    CUDA_CALL(cudaMemcpy(vpXxoeEhdEosLSsYXkNG, KZWeXiYFmdpQdsgidKeG,
                         sizeof(float) * dkLDkRwCBjeybwDHbKiE, cudaMemcpyHostToDevice));
#if 0
    printf("%s loaded. Size = %d. %f\n", QTXuPiGKeBUnmRzhlIDp, dkLDkRwCBjeybwDHbKiE, KZWeXiYFmdpQdsgidKeG[0]);
#endif
    free(KZWeXiYFmdpQdsgidKeG);
    fclose(QjgQHaUACFNSteMrRtRj);
    return;
}

void
MWFusedConvReLULayerImpl::loadBias(const char *QTXuPiGKeBUnmRzhlIDp) {
    FILE *
            QjgQHaUACFNSteMrRtRj = MWCNNLayer::openBinaryFile(QTXuPiGKeBUnmRzhlIDp);
    assert(QjgQHaUACFNSteMrRtRj);
    int dkLDkRwCBjeybwDHbKiE =
            CCKWXUFWgrbBMjwfpOBN * BuyZFXzwOMxcePIbCLfl;
    float *KZWeXiYFmdpQdsgidKeG =
            MALLOC_CALL(sizeof(float) * dkLDkRwCBjeybwDHbKiE);
    call_fread(KZWeXiYFmdpQdsgidKeG,
               sizeof(float), dkLDkRwCBjeybwDHbKiE, QjgQHaUACFNSteMrRtRj, QTXuPiGKeBUnmRzhlIDp);
    CUDA_CALL(cudaMemcpy(IwKnaBoXVubIRYcxEJLH, KZWeXiYFmdpQdsgidKeG,
                         sizeof(float) * dkLDkRwCBjeybwDHbKiE, cudaMemcpyHostToDevice));
    free(KZWeXiYFmdpQdsgidKeG);
    fclose(QjgQHaUACFNSteMrRtRj);
    return;
}

void
MWFusedConvReLULayerImpl::postSetup() {
    if (dMxIKDGTITyhdLqIHBLA->getAutoTune()) { getConvAlgoTuned(); } else { getConvAlgoWorkSpaceLimit(); }
}

void
MWFusedConvReLULayerImpl::getConvAlgoTuned() {
    MWTensorBase *opTensorBase =
            getLayer()->getOutputTensor(0);
    MWTensor<float> *opTensor =
            static_cast<MWTensor<float> *>(opTensorBase);
    cudnnConvolutionFwdAlgoPerf_t
            perf_results[3];
    cudnnTensorDescriptor_t *desc =
            getDescriptor(getLayer()->getOutputTensor()->getSourcePortIndex());
    assert(desc);
    int returnedAlgoCount;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithmEx(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                                      XhAYHFyEVtlwoxGBuTpu,
                                                      static_cast<MWTensor<float> *>(UKtMXCCqdjeyaVHabkxg)->getData(),
                                                      QhTWatiCfcWYsHdkcyhZ, vpXxoeEhdEosLSsYXkNG, NXruhrCCiguRjAgSNDuz,
                                                      *desc,
                                                      opTensor->getData(), 3, &returnedAlgoCount, &perf_results[0],
                                                      dMxIKDGTITyhdLqIHBLA->getWorkSpace(),
                                                      *dMxIKDGTITyhdLqIHBLA->getAllocatedWorkSpaceSize()));
    NMMfJylfQjiIUAKhXCJb =
            perf_results[0].algo;
    if (CUDNN_VERSION < 7402) fixConvAlgo();
}

void
MWFusedConvReLULayerImpl::getConvAlgoWorkSpaceLimit() {
    cudnnTensorDescriptor_t *desc =
            getDescriptor(getLayer()->getOutputTensor()->getSourcePortIndex());
    assert(desc);
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(*dMxIKDGTITyhdLqIHBLA->getCudnnHandle(),
                                                   XhAYHFyEVtlwoxGBuTpu, QhTWatiCfcWYsHdkcyhZ, NXruhrCCiguRjAgSNDuz,
                                                   *desc,
                                                   CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                   *dMxIKDGTITyhdLqIHBLA->getAllocatedWorkSpaceSize(),
                                                   &NMMfJylfQjiIUAKhXCJb));
    if
            (CUDNN_VERSION < 7402)
        fixConvAlgo();
}

void
MWFusedConvReLULayerImpl::fixConvAlgo() {
    int inputH =
            UKtMXCCqdjeyaVHabkxg->getHeight();
    int inputW = UKtMXCCqdjeyaVHabkxg->getWidth();
    if (NMMfJylfQjiIUAKhXCJb == CUDNN_CONVOLUTION_FWD_ALGO_FFT && (inputH > 64 ||
                                                                   inputW > 64)) {
        NMMfJylfQjiIUAKhXCJb = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
}