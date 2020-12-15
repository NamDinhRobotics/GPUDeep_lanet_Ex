/* Copyright 2019 The MathWorks, Inc. */
#ifndef __ELEMENTWISE_AFFINE_LAYER_IMPL_HPP
#define __ELEMENTWISE_AFFINE_LAYER_IMPL_HPP

#include "MWCNNLayerImpl.hpp"

/**
  *  Codegen class for Scaling Factor layer
**/
class MWCNNLayer;

class MWTargetNetworkImpl;

class MWElementwiseAffineLayerImpl : public MWCNNLayerImpl {
public:

    MWElementwiseAffineLayerImpl(MWCNNLayer *layer,
                                 MWTargetNetworkImpl *ntwk_impl,
                                 int scale_H,
                                 int scale_W,
                                 int scale_C,
                                 int offset_H,
                                 int offset_W,
                                 int offset_C,
                                 bool isClipped,
                                 int lowerbound,
                                 int upperbound,
                                 const char *scale_file,
                                 const char *offsetfile);

    ~MWElementwiseAffineLayerImpl();

    void predict();

    void cleanup();

    void propagateSize();

private:

    void loadScale(const char *);

    void loadOffset(const char *);

    float *pvpNsgGssdTxeVoFIkXI;
    float *gNROjwaqhxDPvBWUCUcQ;
    double qBTcAwVGZERyCjGYByPe;
    double qWwjVYwfnvEnFKlgpqwA;
    double pzUAoBDvaKAtdsmkQuct;
    double hljcfGWsvZXJZNrImpJB;
    double hvqKUzPqCuUJRfoNlbwW;
    double hKyfKjPACkOBDvLdESxH;
    bool ZUTPCvgISoRdtnhGqXzM;
    int bQjijJlpNAVdwDDQgpaX;
    int veFyKKHbdqBIvQLYBqfF;


};


#endif
