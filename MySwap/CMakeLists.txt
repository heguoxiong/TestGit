cmake_minimum_required(VERSION 3.1)

project(MySwap)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(my_cmake_swap main.cpp swap.cpp)