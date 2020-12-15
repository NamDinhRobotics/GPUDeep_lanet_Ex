cmake_minimum_required(VERSION 3.15)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(DeepLaneNet)

set(CMAKE_CXX_STANDARD 14)
# The version of CMake which is not compatible with the old CUDA CMake commands.
set(CMAKE_VERSION_THRESHOLD "3.15.0")

find_package(CUDA REQUIRED)
#find_package(BLAS REQUIRED)

# enable_language("CUDA")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")


message(${TENSORRT_INCLUDE_DIR})
message(STATUS ${TENSORRT_ROOT})
message(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
message(CUDA_LIBRARIES ${CUDA_LIBRARIES})
message(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARY})
message(CUDA_cusolver_LIBRARY ${CUDA_cusolver_LIBRARY})
message(CUDA_CUFFT_LIBRARIES ${CUDA_CUFFT_LIBRARIES})
message(TENSORRT_LIBRARY_INFER ${TENSORRT_LIBRARY_INFER})

#set all cuda lib
SET(ALL_CUDA_LIBS ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")
message(STATUS "LIBS: ${LIBS}")

include_directories(${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
#        ${BLAS_INCLUDE_DIRS}
        ./detect_lane
#        ./mulMatrix/interface
        )

#link_directories(/usr/local/cuda-10.2 /usr/local/cuda-10.2/lib /usr/local/cuda-10.2/lib64)

file(GLOB GPU_CPP ./detect_lane/*.cu )

file(GLOB GPU_CCC ./detect_lane/*.cpp)
#file(GLOB GPU_CCC ./mulMatrix/MWCUBLASUtils.cpp)

cuda_add_executable(deepLaneNet main.cpp ${GPU_CPP} ${GPU_CCC})
#add_library(gpu_check ${GPU_CPP}) # create a GPU library from MATLAB
cuda_add_cublas_to_target(deepLaneNet)

# target the lib with others LIBS
# need to aad more lib in https://cmake.org/cmake/help/latest/module/FindCUDA.html
target_link_libraries(deepLaneNet
        ${TENSORRT_LIBRARY_INFER}
        ${OpenCV_LIBS}
        ${CPP_FS_LIB}
        ${LIBS} # for many exmaple
        /usr/local/cuda-10.2/lib64/libcudnn.so
#        cudnn
#        cublas
#        ${CUDA_LIBRARIES}
#        ${CUDA_cusolver_LIBRARY}
#        cudnn
       )
#target_link_libraries( your_projectSolution_name ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} ${CUDA_nvcuvenc_LIBRARY} ${CUDA_nvcuvid_LIBRARY})

#make run file and target lib