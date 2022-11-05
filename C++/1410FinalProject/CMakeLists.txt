cmake_minimum_required(VERSION 3.21)
project(1410FinalProject)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_executable(1410FinalProject
        include/Car.h
        include/carFileReader.h
        src/Car.cpp
        src/carFileReader.cpp
        src/main.cpp)
