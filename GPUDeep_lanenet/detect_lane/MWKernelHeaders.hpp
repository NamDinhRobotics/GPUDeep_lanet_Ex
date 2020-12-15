/* Copyright 2018-2019 The MathWorks, Inc. */

#ifndef __KERNEL_HEADER_HPP
#define __KERNEL_HEADER_HPP

/* SSD Merge layer kernel declaration */
void __global__ ssdMergeLayerImpl(float *,
                                  float *,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  long int,
                                  long int,
                                  long int,
                                  long int,
                                  const long int);

/* YOLO reorg kernel impl */
void __global__ YoloReorg2dImpl(float *,
                                float *,
                                int,
                                int,
                                long int,
                                long int,
                                long int,
                                long int,
                                int,
                                int,
                                int,
                                int,
                                const long int BkwhtPQUCQKchmmimoXs);


/* YOLO extraction kernel impl */
void __global__ YoloExtractionImpl(float *,
                                   float *,
                                   float *,
                                   float *,
                                   int,
                                   int,
                                   long int,
                                   long int,
                                   long int,
                                   long int,
                                   long int,
                                   const long int);

/* Exponential kernel impl */
void __global__ exp_kernel(float *, float *, const long int);


/* Max unpooling kernel impl */
void __global__ MaxUnpoolingImpl(float *, float *, float *, const int);


/* Clipped ReLU kernel impl */
void __global__ ClippedReLUImpl_kernel(float *, float *, const double, const int);

void __global__ scale_scalar_kernel(float *, float *, float *, long int);

void __global__ scale_vector_kernel(float *, float *, float *, double, double, long int);

void __global__
scale_matrix2d_kernel(float *, float *, float *, double, long int);

void __global__
scale_tensor3d_kernel(float *, float *, float *, double, long int);

void __global__ offset_scalar_kernel(float *, float *, float *, long int, bool, int, int);

void __global__
offset_vector_kernel(float *, float *, float *, double, double, long int, bool, int, int);

void __global__ offset_matrix2d_kernel(float *,
                                       float *,
                                       float *,
                                       double,
                                       long int,
                                       bool,
                                       int,
                                       int);

void __global__ offset_tensor3d_kernel(float *,
                                       float *,
                                       float *,
                                       double,
                                       long int,
                                       bool,
                                       int,
                                       int);

/* LSTM kernel Impl */
void __global__ expand_batch_kernel(float *, float *, long int, long int);

void __global__
interleave_bilstm_passes(float *, float *, const long int, const long int, const long int);


/* Leaky ReLU kernel Impl */
void __global__ leakyReLUImpl(float *, float *, const double, const int);

/* WordEmbedding kernel Impl */
void __global__ wordEmbedding_kernel_vocab(float *, float *, const int, const int);

void __global__ wordEmbedding_kernel_lookup(float *, float *, float *, const int, const int);

/* crop 2d kernel Impl */
void __global__ Crop2dImpl(float *,
                           float *,
                           int,
                           long int,
                           long int,
                           long int,
                           long int,
                           int,
                           int,
                           int,
                           const long int);

#endif
