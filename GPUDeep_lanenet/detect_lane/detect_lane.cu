//
// File: detect_lane.cu
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//

// Include Files
#include "detect_lane.h"
#include "DeepLearningNetwork.h"
#include "detect_lane_data.h"
#include "detect_lane_initialize.h"
#include "detect_lane_internal_types.h"
#include "predict.h"
#include "MWCudaDimUtility.hpp"
#include <cmath>

// Variable Definitions
static lanenet0_0 lanenet;
static boolean_T lanenet_not_empty;

// Function Declarations
static __global__ void detect_lane_kernel1(const float frame[154587], float b
[154587]);

static __global__ void detect_lane_kernel10(double b_x[9]);

static __global__ void detect_lane_kernel11(const double t1, const double t2,
                                            double Tinv[9]);

static __global__ void detect_lane_kernel12(const double t1, const double t2,
                                            const int p2, double Tinv[9]);

static __global__ void detect_lane_kernel13(const double t1, const double t2,
                                            const int p3, double Tinv[9]);

static __global__ void detect_lane_kernel14(const float lt_y[28], float fv[84]);

static __global__ void detect_lane_kernel15(const double Tinv[9], const float
fv[84], float U[84]);

static __global__ void detect_lane_kernel16(const float U[84], float b[56]);

static __global__ void detect_lane_kernel17(const float b[56], float U[84]);

static __global__ void detect_lane_kernel18(const float U[84], float ltPts[56]);

static __global__ void detect_lane_kernel19(const double Tinv[9], double b_x[9]);

static __global__ void detect_lane_kernel2(const double laneCoeffMeans[6], const
double laneCoeffStds[6], float lanecoeffsNetworkOutput[6]);

static __global__ void detect_lane_kernel20(double b_x[9]);

static __global__ void detect_lane_kernel21(double b_x[9]);

static __global__ void detect_lane_kernel22(const double t1, const double t2,
                                            double Tinv[9]);

static __global__ void detect_lane_kernel23(const double t1, const double t2,
                                            const int p2, double Tinv[9]);

static __global__ void detect_lane_kernel24(const double t1, const double t2,
                                            const int p3, double Tinv[9]);

static __global__ void detect_lane_kernel25(const float rt_y[28], float fv[84]);

static __global__ void detect_lane_kernel26(const double Tinv[9], const float
fv[84], float U[84]);

static __global__ void detect_lane_kernel27(const float U[84], float b[56]);

static __global__ void detect_lane_kernel28(const float b[56], float U[84]);

static __global__ void detect_lane_kernel29(const float U[84], float rtPts[56]);

static __global__ void detect_lane_kernel3(const float lanecoeffsNetworkOutput[6],
                                           float rt_y[28]);

static __global__ void detect_lane_kernel4(const float lanecoeffsNetworkOutput,
                                           float rt_y[28]);

static __global__ void detect_lane_kernel5(const float lanecoeffsNetworkOutput[6],
                                           float lt_y[28]);

static __global__ void detect_lane_kernel6(const float lanecoeffsNetworkOutput,
                                           float lt_y[28]);

static __global__ void detect_lane_kernel7(double Tinv[9]);

static __global__ void detect_lane_kernel8(const double Tinv[9], double b_x[9]);

static __global__ void detect_lane_kernel9(double b_x[9]);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float frame[154587]
//                float b[154587]
// Return Type  : void
//
static __global__ __launch_bounds__(512, 1) void detect_lane_kernel1(const float
                                                                     frame[154587], float b[154587]) {
    unsigned long threadId;
    int ibcol;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    jtilecol = static_cast<int>(threadId % 227UL);
    threadId = (threadId - static_cast<unsigned long>(jtilecol)) / 227UL;
    ibcol = static_cast<int>(threadId % 227UL);
    threadId = (threadId - static_cast<unsigned long>(ibcol)) / 227UL;
    k = static_cast<int>(threadId);
    if ((static_cast<int>((static_cast<int>(k < 3)) && (static_cast<int>(ibcol <
                                                                         227)))) &&
        (static_cast<int>(jtilecol < 227))) {
        b[(ibcol + 227 * jtilecol) + 51529 * k] = frame[(jtilecol + 227 * ibcol) +
                                                        51529 * k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel10(double b_x
[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        b_x[4] -= b_x[1] * 1.0305949982581226;
        b_x[5] = -0.0032316839464807288 - b_x[2] * 1.0305949982581226;
        b_x[7] -= b_x[1] * -0.22205377950113064;
        b_x[8] = 1.9916790026632809E-35 - b_x[2] * -0.22205377950113064;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel11(const
                                                                     double t1, const double t2, double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[6] = ((1.0 - 1.0305949982581226 * t2) - -0.22205377950113064 * t1) /
                  1.1512965678044422;
        Tinv[7] = t2;
        Tinv[8] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                const int p2
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel12(const
                                                                     double t1, const double t2, const int p2,
                                                                     double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[p2] = -(1.0305949982581226 * t2 + -0.22205377950113064 * t1) /
                   1.1512965678044422;
        Tinv[p2 + 1] = t2;
        Tinv[p2 + 2] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                const int p3
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel13(const
                                                                     double t1, const double t2, const int p3,
                                                                     double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[p3] = -(1.0305949982581226 * t2 + -0.22205377950113064 * t1) /
                   1.1512965678044422;
        Tinv[p3 + 1] = t2;
        Tinv[p3 + 2] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float lt_y[28]
//                float fv[84]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel14(const float
                                                                     lt_y[28], float fv[84]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        fv[k] = static_cast<float>(k) + 3.0F;
        fv[k + 28] = lt_y[k];
        fv[k + 56] = 1.0F;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Tinv[9]
//                const float fv[84]
//                float U[84]
// Return Type  : void
//
static __global__ __launch_bounds__(96, 1) void detect_lane_kernel15(const
                                                                     double Tinv[9], const float fv[84], float U[84]) {
    unsigned long threadId;
    float f;
    int ibcol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    ibcol = static_cast<int>(threadId % 3UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(ibcol)) / 3UL);
    if ((static_cast<int>(k < 28)) && (static_cast<int>(ibcol < 3))) {
        f = 0.0F;
        for (int jtilecol = 0; jtilecol < 3; jtilecol++) {
            f += fv[k + 28 * jtilecol] * static_cast<float>(Tinv[jtilecol + 3 * ibcol]);
        }

        U[k + 28 * ibcol] = f;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float U[84]
//                float b[56]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel16(const float
                                                                     U[84], float b[56]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    k = static_cast<int>(threadId % 28UL);
    jtilecol = static_cast<int>((threadId - static_cast<unsigned long>(k)) / 28UL);
    if ((static_cast<int>(jtilecol < 2)) && (static_cast<int>(k < 28))) {
        b[jtilecol * 28 + k] = U[k + 56];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float b[56]
//                float U[84]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel17(const float
                                                                     b[56], float U[84]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    jtilecol = static_cast<int>(threadId % 28UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(jtilecol)) / 28UL);
    if ((static_cast<int>(k < 2)) && (static_cast<int>(jtilecol < 28))) {
        U[jtilecol + 28 * k] /= b[jtilecol + 28 * k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float U[84]
//                float ltPts[56]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel18(const float
                                                                     U[84], float ltPts[56]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    jtilecol = static_cast<int>(threadId % 28UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(jtilecol)) / 28UL);
    if ((static_cast<int>(k < 2)) && (static_cast<int>(jtilecol < 28))) {
        ltPts[jtilecol + 28 * k] = U[jtilecol + 28 * k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Tinv[9]
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel19(const
                                                                     double Tinv[9], double b_x[9]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 9) {
        b_x[k] = Tinv[k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double laneCoeffMeans[6]
//                const double laneCoeffStds[6]
//                float lanecoeffsNetworkOutput[6]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel2(const double
                                                                    laneCoeffMeans[6], const double laneCoeffStds[6],
                                                                    float
                                                                    lanecoeffsNetworkOutput[6]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 6) {
        //  Recover original coeffs by reversing the normalization steps
        lanecoeffsNetworkOutput[k] = lanecoeffsNetworkOutput[k] * static_cast<float>
        (laneCoeffStds[k]) + static_cast<float>(laneCoeffMeans[k]);
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel20(double b_x
[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        b_x[1] /= 1.1512965678044422;
        b_x[2] = -1.718788847108661E-19;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel21(double b_x
[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        b_x[4] -= b_x[1] * 1.0305949982581226;
        b_x[5] = -0.0032316839464807288 - b_x[2] * 1.0305949982581226;
        b_x[7] -= b_x[1] * -0.22205377950113064;
        b_x[8] = 1.9916790026632809E-35 - b_x[2] * -0.22205377950113064;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel22(const
                                                                     double t1, const double t2, double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[6] = ((1.0 - 1.0305949982581226 * t2) - -0.22205377950113064 * t1) /
                  1.1512965678044422;
        Tinv[7] = t2;
        Tinv[8] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                const int p2
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel23(const
                                                                     double t1, const double t2, const int p2,
                                                                     double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[p2] = -(1.0305949982581226 * t2 + -0.22205377950113064 * t1) /
                   1.1512965678044422;
        Tinv[p2 + 1] = t2;
        Tinv[p2 + 2] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double t1
//                const double t2
//                const int p3
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel24(const
                                                                     double t1, const double t2, const int p3,
                                                                     double Tinv[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[p3] = -(1.0305949982581226 * t2 + -0.22205377950113064 * t1) /
                   1.1512965678044422;
        Tinv[p3 + 1] = t2;
        Tinv[p3 + 2] = t1;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float rt_y[28]
//                float fv[84]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel25(const float
                                                                     rt_y[28], float fv[84]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        fv[k] = static_cast<float>(k) + 3.0F;
        fv[k + 28] = rt_y[k];
        fv[k + 56] = 1.0F;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Tinv[9]
//                const float fv[84]
//                float U[84]
// Return Type  : void
//
static __global__ __launch_bounds__(96, 1) void detect_lane_kernel26(const
                                                                     double Tinv[9], const float fv[84], float U[84]) {
    unsigned long threadId;
    float f;
    int ibcol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    ibcol = static_cast<int>(threadId % 3UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(ibcol)) / 3UL);
    if ((static_cast<int>(k < 28)) && (static_cast<int>(ibcol < 3))) {
        f = 0.0F;
        for (int jtilecol = 0; jtilecol < 3; jtilecol++) {
            f += fv[k + 28 * jtilecol] * static_cast<float>(Tinv[jtilecol + 3 * ibcol]);
        }

        U[k + 28 * ibcol] = f;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float U[84]
//                float b[56]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel27(const float
                                                                     U[84], float b[56]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    k = static_cast<int>(threadId % 28UL);
    jtilecol = static_cast<int>((threadId - static_cast<unsigned long>(k)) / 28UL);
    if ((static_cast<int>(jtilecol < 2)) && (static_cast<int>(k < 28))) {
        b[jtilecol * 28 + k] = U[k + 56];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float b[56]
//                float U[84]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel28(const float
                                                                     b[56], float U[84]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    jtilecol = static_cast<int>(threadId % 28UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(jtilecol)) / 28UL);
    if ((static_cast<int>(k < 2)) && (static_cast<int>(jtilecol < 28))) {
        U[jtilecol + 28 * k] /= b[jtilecol + 28 * k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float U[84]
//                float rtPts[56]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void detect_lane_kernel29(const float
                                                                     U[84], float rtPts[56]) {
    unsigned long threadId;
    int jtilecol;
    int k;
    threadId = mwGetGlobalThreadIndex();
    jtilecol = static_cast<int>(threadId % 28UL);
    k = static_cast<int>((threadId - static_cast<unsigned long>(jtilecol)) / 28UL);
    if ((static_cast<int>(k < 2)) && (static_cast<int>(jtilecol < 28))) {
        rtPts[jtilecol + 28 * k] = U[jtilecol + 28 * k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float lanecoeffsNetworkOutput[6]
//                float rt_y[28]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel3(const float
                                                                    lanecoeffsNetworkOutput[6], float rt_y[28]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        rt_y[k] = lanecoeffsNetworkOutput[3];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float lanecoeffsNetworkOutput
//                float rt_y[28]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel4(const float
                                                                    lanecoeffsNetworkOutput, float rt_y[28]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        rt_y[k] = (static_cast<float>(k) + 3.0F) * rt_y[k] + lanecoeffsNetworkOutput;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float lanecoeffsNetworkOutput[6]
//                float lt_y[28]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel5(const float
                                                                    lanecoeffsNetworkOutput[6], float lt_y[28]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        lt_y[k] = lanecoeffsNetworkOutput[0];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const float lanecoeffsNetworkOutput
//                float lt_y[28]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel6(const float
                                                                    lanecoeffsNetworkOutput, float lt_y[28]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 28) {
        lt_y[k] = (static_cast<float>(k) + 3.0F) * lt_y[k] + lanecoeffsNetworkOutput;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                double Tinv[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel7(double Tinv
[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        Tinv[2] = 1.1512965678044422;
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Tinv[9]
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel8(const double
                                                                    Tinv[9], double b_x[9]) {
    int k;
    k = static_cast<int>(mwGetGlobalThreadIndex());
    if (k < 9) {
        //  map vehicle to image coordinates
        b_x[k] = Tinv[k];
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                double b_x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void detect_lane_kernel9(double b_x[9]) {
    int tmpIdx;
    tmpIdx = static_cast<int>(mwGetGlobalThreadIndex());
    if (tmpIdx < 1) {
        b_x[1] /= 1.1512965678044422;
        b_x[2] = -1.718788847108661E-19;
    }
}

//
// From the networks output, compute left and right lane points in the
//  image coordinates. The camera coordinates are described by the caltech
//  mono camera model.
// Arguments    : const float frame[154587]
//                const double laneCoeffMeans[6]
//                const double laneCoeffStds[6]
//                boolean_T *laneFound
//                float ltPts[56]
//                float rtPts[56]
// Return Type  : void
//
void detect_lane(const float frame[154587], const double laneCoeffMeans[6],
                 const double laneCoeffStds[6], boolean_T *laneFound, float
                 ltPts[56], float rtPts[56]) {
    static float b[154587];
    double Tinv[9];
    double b_x[9];
    double (*b_gpu_Tinv)[9];
    double (*gpu_Tinv)[9];
    double (*gpu_x)[9];
    double (*gpu_laneCoeffMeans)[6];
    double (*gpu_laneCoeffStds)[6];
    float (*gpu_b)[154587];
    float (*gpu_frame)[154587];
    float (*gpu_U)[84];
    float (*gpu_fv)[84];
    float (*b_gpu_b)[56];
    float (*gpu_ltPts)[56];
    float (*gpu_rtPts)[56];
    float (*gpu_lt_y)[28];
    float (*gpu_rt_y)[28];
    float lanecoeffsNetworkOutput[6];
    float (*gpu_lanecoeffsNetworkOutput)[6];
    boolean_T rtPts_dirtyOnGpu;
    if (!isInitialized_detect_lane) {
        detect_lane_initialize();
    }

    cudaMalloc(&gpu_rtPts, 224UL);
    cudaMalloc(&gpu_ltPts, 224UL);
    cudaMalloc(&b_gpu_b, 224UL);
    cudaMalloc(&gpu_U, 336UL);
    cudaMalloc(&gpu_fv, 336UL);
    cudaMalloc(&b_gpu_Tinv, 72UL);
    cudaMalloc(&gpu_x, 72UL);
    cudaMalloc(&gpu_Tinv, 72UL);
    cudaMalloc(&gpu_lt_y, 112UL);
    cudaMalloc(&gpu_rt_y, 112UL);
    cudaMalloc(&gpu_lanecoeffsNetworkOutput, 24UL);
    cudaMalloc(&gpu_laneCoeffStds, 48UL);
    cudaMalloc(&gpu_laneCoeffMeans, 48UL);
    cudaMalloc(&gpu_b, 618348UL);
    cudaMalloc(&gpu_frame, 618348UL);
    rtPts_dirtyOnGpu = false;

    //  A persistent object mynet is used to load the series network object.
    //  At the first call to this function, the persistent object is constructed and
    //  setup. When the function is called subsequent times, the same object is reused
    //  to call predict on inputs, thus avoiding reconstructing and reloading the
    //  network object.
    if (!lanenet_not_empty) {
        coder::DeepLearningNetwork_setup(&lanenet);
        lanenet_not_empty = true;
    }

    cudaMemcpy(gpu_frame, (void *) &frame[0], 618348UL, cudaMemcpyHostToDevice);
    detect_lane_kernel1<<<dim3(302U, 1U, 1U), dim3(512U, 1U, 1U)>>>(*gpu_frame,
                                                                    *gpu_b);
    cudaMemcpy(&b[0], gpu_b, 618348UL, cudaMemcpyDeviceToHost);
    coder::DeepLearningNetwork_predict(&lanenet, b, lanecoeffsNetworkOutput);

    //  Recover original coeffs by reversing the normalization steps
    cudaMemcpy(gpu_laneCoeffMeans, (void *) &laneCoeffMeans[0], 48UL,
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_laneCoeffStds, (void *) &laneCoeffStds[0], 48UL,
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_lanecoeffsNetworkOutput, &lanecoeffsNetworkOutput[0], 24UL,
               cudaMemcpyHostToDevice);
    detect_lane_kernel2<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
            (*gpu_laneCoeffMeans, *gpu_laneCoeffStds, *gpu_lanecoeffsNetworkOutput);

    // c should be more than 0.5 for it to be a right lane
    // meters, ahead of the sensor
    cudaMemcpy(&lanecoeffsNetworkOutput[0], gpu_lanecoeffsNetworkOutput, 24UL,
               cudaMemcpyDeviceToHost);
    if ((std::abs(lanecoeffsNetworkOutput[5]) > 0.5F) && (std::abs
                                                                  (lanecoeffsNetworkOutput[2]) > 0.5F)) {
        double t1;
        int p2;
        int p3;
        detect_lane_kernel3<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
                (*gpu_lanecoeffsNetworkOutput, *gpu_rt_y);
        for (p2 = 0; p2 < 2; p2++) {
            detect_lane_kernel4<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
                    (lanecoeffsNetworkOutput[p2 + 4], *gpu_rt_y);
        }

        detect_lane_kernel5<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
                (*gpu_lanecoeffsNetworkOutput, *gpu_lt_y);
        for (p2 = 0; p2 < 2; p2++) {
            detect_lane_kernel6<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>
                    (lanecoeffsNetworkOutput[p2 + 1], *gpu_lt_y);
        }

        //  Visualize lane boundaries of the ego vehicle
        //  Compute extrinsics based on camera setup
        //  pitch of the camera in degrees
        //  Construct a camera matrix
        //  Turn camMatrix into 2-D homography
        //  drop Z
        Tinv[3] = -0.0032316839464807288;
        Tinv[4] = -1.2852132429203174E-19;
        Tinv[5] = 1.0305949982581226;
        Tinv[6] = 1.9916790026632809E-35;
        Tinv[7] = 0.0012931719938928032;
        Tinv[8] = -0.22205377950113064;
        Tinv[0] = -1.9788357004567556E-19;
        Tinv[1] = -0.00070281981464454381;
        cudaMemcpy(gpu_Tinv, &Tinv[0], 72UL, cudaMemcpyHostToDevice);
        detect_lane_kernel7<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Tinv);

        //  map vehicle to image coordinates
        detect_lane_kernel8<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Tinv,
                                                                     *gpu_x);
        p2 = 3;
        p3 = 0;
        detect_lane_kernel9<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x);
        detect_lane_kernel10<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x);
        cudaMemcpy(&b_x[0], gpu_x, 72UL, cudaMemcpyDeviceToHost);
        if (std::abs(b_x[5]) > std::abs(b_x[4])) {
            p2 = 0;
            p3 = 3;
            t1 = b_x[1];
            b_x[1] = b_x[2];
            b_x[2] = t1;
            t1 = b_x[4];
            b_x[4] = b_x[5];
            b_x[5] = t1;
            t1 = b_x[7];
            b_x[7] = b_x[8];
            b_x[8] = t1;
        }

        b_x[5] /= b_x[4];
        b_x[8] -= b_x[5] * b_x[7];
        t1 = (b_x[5] * b_x[1] - b_x[2]) / b_x[8];
        detect_lane_kernel11<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, -(b_x[1] +
                                                                            b_x[7] * t1) / b_x[4], *b_gpu_Tinv);
        t1 = -b_x[5] / b_x[8];
        detect_lane_kernel12<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, (1.0 -
                                                                           b_x[7] * t1) / b_x[4], p2, *b_gpu_Tinv);
        t1 = 1.0 / b_x[8];
        detect_lane_kernel13<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, -b_x[7] *
                                                                          t1 / b_x[4], p3, *b_gpu_Tinv);
        detect_lane_kernel14<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_lt_y,
                                                                      *gpu_fv);
        detect_lane_kernel15<<<dim3(1U, 1U, 1U), dim3(96U, 1U, 1U)>>>(*b_gpu_Tinv,
                                                                      *gpu_fv, *gpu_U);
        detect_lane_kernel16<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_U,
                                                                      *b_gpu_b);
        detect_lane_kernel17<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_b,
                                                                      *gpu_U);
        detect_lane_kernel18<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_U,
                                                                      *gpu_ltPts);
        cudaMemcpy(gpu_x, &b_x[0], 72UL, cudaMemcpyHostToDevice);
        detect_lane_kernel19<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Tinv,
                                                                      *gpu_x);
        p2 = 3;
        p3 = 0;
        detect_lane_kernel20<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x);
        detect_lane_kernel21<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x);
        cudaMemcpy(&b_x[0], gpu_x, 72UL, cudaMemcpyDeviceToHost);
        if (std::abs(b_x[5]) > std::abs(b_x[4])) {
            p2 = 0;
            p3 = 3;
            t1 = b_x[1];
            b_x[1] = b_x[2];
            b_x[2] = t1;
            t1 = b_x[4];
            b_x[4] = b_x[5];
            b_x[5] = t1;
            t1 = b_x[7];
            b_x[7] = b_x[8];
            b_x[8] = t1;
        }

        b_x[5] /= b_x[4];
        b_x[8] -= b_x[5] * b_x[7];
        t1 = (b_x[5] * b_x[1] - b_x[2]) / b_x[8];
        detect_lane_kernel22<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, -(b_x[1] +
                                                                            b_x[7] * t1) / b_x[4], *gpu_Tinv);
        t1 = -b_x[5] / b_x[8];
        detect_lane_kernel23<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, (1.0 -
                                                                           b_x[7] * t1) / b_x[4], p2, *gpu_Tinv);
        t1 = 1.0 / b_x[8];
        detect_lane_kernel24<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(t1, -b_x[7] *
                                                                          t1 / b_x[4], p3, *gpu_Tinv);
        detect_lane_kernel25<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_rt_y,
                                                                      *gpu_fv);
        detect_lane_kernel26<<<dim3(1U, 1U, 1U), dim3(96U, 1U, 1U)>>>(*gpu_Tinv,
                                                                      *gpu_fv, *gpu_U);
        detect_lane_kernel27<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_U,
                                                                      *b_gpu_b);
        detect_lane_kernel28<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_b,
                                                                      *gpu_U);
        detect_lane_kernel29<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_U,
                                                                      *gpu_rtPts);
        rtPts_dirtyOnGpu = true;
        *laneFound = true;
        cudaMemcpy(&ltPts[0], gpu_ltPts, 224UL, cudaMemcpyDeviceToHost);
    } else {
        *laneFound = false;
    }

    if (rtPts_dirtyOnGpu) {
        cudaMemcpy(&rtPts[0], gpu_rtPts, 224UL, cudaMemcpyDeviceToHost);
    }

    cudaFree(*gpu_frame);
    cudaFree(*gpu_b);
    cudaFree(*gpu_laneCoeffMeans);
    cudaFree(*gpu_laneCoeffStds);
    cudaFree(*gpu_lanecoeffsNetworkOutput);
    cudaFree(*gpu_rt_y);
    cudaFree(*gpu_lt_y);
    cudaFree(*gpu_Tinv);
    cudaFree(*gpu_x);
    cudaFree(*b_gpu_Tinv);
    cudaFree(*gpu_fv);
    cudaFree(*gpu_U);
    cudaFree(*b_gpu_b);
    cudaFree(*gpu_ltPts);
    cudaFree(*gpu_rtPts);
}

//
// From the networks output, compute left and right lane points in the
//  image coordinates. The camera coordinates are described by the caltech
//  mono camera model.
// Arguments    : void
// Return Type  : void
//
void detect_lane_init() {
    lanenet_not_empty = false;
}

//
// File trailer for detect_lane.cu
//
// [EOF]
//
