//
// File: detect_lane_initialize.cu
//
// GPU Coder version                    : 2.0
// CUDA/C/C++ source code generated on  : 15-Dec-2020 12:44:50
//

// Include Files
#include "detect_lane_initialize.h"
#include "detect_lane.h"
#include "detect_lane_data.h"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void detect_lane_initialize() {
    detect_lane_init();
    isInitialized_detect_lane = true;
}

//
// File trailer for detect_lane_initialize.cu
//
// [EOF]
//
