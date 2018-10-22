

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(function_scope__host_device_class CUDA)

add_executable(function_scope__host_device_class main.cu)
set_target_properties(function_scope__host_device_class PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(function_scope__host_device_class PRIVATE -std=c++11 -arch=sm_60)
