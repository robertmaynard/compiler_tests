

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(list_for_each_constructor CUDA)

add_executable(list_for_each_constructor main.cu)
target_compile_options(list_for_each_constructor PRIVATE -std=c++11 -arch=sm_60)
