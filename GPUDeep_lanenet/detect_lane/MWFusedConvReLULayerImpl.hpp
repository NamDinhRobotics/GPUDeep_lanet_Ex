/* Copyright 2018 The MathWorks, Inc. */

#ifndef __FUSED_CONV_RELU_LAYER_IMPL_HPP
#define __FUSED_CONV_RELU_LAYER_IMPL_HPP

#include "MWFusedConvReLULayer.hpp"
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"


class MWFusedConvReLULayerImpl : public MWCNNLayerImpl {

private:
    int BLjrjqvCcCommiXWQLjs;           //Filter height for CONV and FC
    int BRSPqxNffoBYKqpSVHne;            //Filter width for CONV and FC

    int CCKWXUFWgrbBMjwfpOBN;
    int BlRIQPyqJZORKENzSdYf;
    int BuyZFXzwOMxcePIbCLfl;

    int FrpxvsDMwwgbpqHXWxmN;
    int FwLnexHgxHRquTKmNpoa;
    int CZNYmBcNFSZWvaCklqeM;
    int CTCbzQMDaLxINPbODdng;
    int CqtPRJvHlGJFssiPzsOm;
    int CufLFODQDXTAPyRqYodN;
    int AuqaQHxmPQSyYRemQvyX;
    int AzTsxYcYjIEJsGQbeYHm;

    int fxxCPKTclxXPxrdMAkwi;

    float *vpXxoeEhdEosLSsYXkNG;
    float *IwKnaBoXVubIRYcxEJLH;
    MWTensorBase *UKtMXCCqdjeyaVHabkxg; // for pre-padded input
    int bYBVtTnVUuGDUlaTmmHp;
    int cQBKlCKXxecGPJrXBXdk;

    float *XCLDbxHBtWRStETWIkId;

    float *HhKGcPZwrclEFnIdWerH; // Scaling factor for addition

    bool IIiwAtyrOtLzLWAUlTey;

public:
    MWFusedConvReLULayerImpl(MWCNNLayer *, MWTargetNetworkImpl *,
                             int, int,
                             int, int, int,
                             int, int,
                             int, int, int, int,
                             int, int,
                             int,
                             const char *, const char *);

    ~MWFusedConvReLULayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

    void allocate();

    void deallocate();

    void postSetup();

private:
    void loadWeights(const char *);

    void loadBias(const char *);

    void getConvAlgoTuned();

    void getConvAlgoWorkSpaceLimit();

    void fixConvAlgo(); // g1916490

    void setalpha2Ptr(float *alpha2) { HhKGcPZwrclEFnIdWerH = alpha2; }

    float *getalpha2Ptr() { return HhKGcPZwrclEFnIdWerH; }


private:
    cudnnConvolutionDescriptor_t NXruhrCCiguRjAgSNDuz;
    cudnnConvolutionFwdAlgo_t NMMfJylfQjiIUAKhXCJb;

    cudnnFilterDescriptor_t QhTWatiCfcWYsHdkcyhZ;
    cudnnTensorDescriptor_t JgLfgHrHMEMmMYTettJF;

    cudnnTensorDescriptor_t XhAYHFyEVtlwoxGBuTpu;

    cudnnActivationDescriptor_t oKIvzXXMucEDsTGGpdpm;

};

#endif
