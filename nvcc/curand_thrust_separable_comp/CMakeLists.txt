

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(curand_thrust_separable_comp CUDA)

add_executable(curand_thrust_separable_comp main.cu)
set_target_properties(curand_thrust_separable_comp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(curand_thrust_separable_comp PRIVATE -std=c++11 -arch=sm_60)
