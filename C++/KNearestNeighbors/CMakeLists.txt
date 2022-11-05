cmake_minimum_required(VERSION 3.21)
project(KNearestNeighbors)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_executable(KNearestNeighbors
        include/KNNClassifier.h
        src/KNNClassifier.cpp
        src/main.cpp)
