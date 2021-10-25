cmake_minimum_required(VERSION 2.8)
project(GCN)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")
add_executable(GCN utilities.c kernels.c main.c)
