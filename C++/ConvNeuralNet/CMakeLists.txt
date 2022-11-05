cmake_minimum_required(VERSION 3.21)
project(CNN CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CNN src/main.cpp src/alg_stopwatch.cpp include/alg_stopwatch.h include/ConvNet.h include/ConvLayer.h src/ConvNet.cpp src/ConvLayer.cpp)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set_target_properties(CNN PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
