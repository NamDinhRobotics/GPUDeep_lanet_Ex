//
// File: main.cu
//
// NET TO CHANGE , directory of .bin file Exactly in File: DeepLearningNetwork.cu
// "/home/dinhnambkhn/CLionProjects/GPUDeep_lanenet/detect_lane/cnn_lanenet0_0_conv2_w.bin",

//***********************************************************************

// Include Files
#include "main.h"
#include "detect_lane.h"
#include "detect_lane_terminate.h"
#include <istream>
// Function Declarations
static void argInit_1x6_real_T(double result[6]);
static void argInit_227x227x3_real32_T(float result[154587]);
static float argInit_real32_T();
static double argInit_real_T();
static void main_detect_lane();

// Function Definitions
//
// Arguments    : double result[6]
// Return Type  : void
//
static void argInit_1x6_real_T(double result[6])
{
    // Loop over the array to initialize each element.
    for (int idx1 = 0; idx1 < 6; idx1++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[idx1] = argInit_real_T();
    }
}

//
// Arguments    : float result[154587]
// Return Type  : void
//
static void argInit_227x227x3_real32_T(float result[154587])
{
    // Loop over the array to initialize each element.
    for (int idx0 = 0; idx0 < 227; idx0++) {
        for (int idx1 = 0; idx1 < 227; idx1++) {
            for (int idx2 = 0; idx2 < 3; idx2++) {
                // Set the value of the array element.
                // Change this value to the value that the application requires.
                result[(idx0 + 227 * idx1) + 51529 * idx2] = argInit_real32_T();
            }
        }
    }
}

//
// Arguments    : void
// Return Type  : float
//
static float argInit_real32_T()
{
    return 1.2F;
}

//
// Arguments    : void
// Return Type  : double
//
static double argInit_real_T()
{
    return 1.0;
}

//
// Arguments    : void
// Return Type  : void
//
static void main_detect_lane()
{
    static float b[154587];
    double c[6];
    double d[6];
    float ltPts[56];
    float rtPts[56];
    boolean_T laneFound;

    // Initialize function 'detect_lane' input arguments.
    // Initialize function input argument 'frame'.
    // Initialize function input argument 'laneCoeffMeans'.
    // Initialize function input argument 'laneCoeffStds'.
    // Call the entry-point 'detect_lane'.
    argInit_227x227x3_real32_T(b);
    argInit_1x6_real_T(c);
    argInit_1x6_real_T(d);
    detect_lane(b, c, d, &laneFound, ltPts, rtPts);
    printf("detect lane \n");
}

//
// Arguments    : int argc
//                const char * const argv[]
// Return Type  : int
//
int main(int, const char * const [])
{
    // The initialize function is being called automatically from your entry-point function. So, a call to initialize is not included here.
    // Invoke the entry-point functions.
    // You can call entry-point functions multiple times.
    main_detect_lane();

    // Terminate the application.
    // You do not need to do this more than one time.
    detect_lane_terminate();
    return 0;
}

//
// File trailer for main.cu
//
// [EOF]
//
