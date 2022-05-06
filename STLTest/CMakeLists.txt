cmake_minimum_required(VERSION 3.1)

project(STLTest)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11")

add_executable(my_cmake_stl main.cpp SeqContainer.cpp)