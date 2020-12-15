//
// File: detect_lane.h
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//
#ifndef DETECT_LANE_H
#define DETECT_LANE_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void detect_lane(const float frame[154587], const double laneCoeffMeans[6],
                        const double laneCoeffStds[6], boolean_T *laneFound, float ltPts[56], float
                        rtPts[56]);

void detect_lane_init();

#endif

//
// File trailer for detect_lane.h
//
// [EOF]
//
