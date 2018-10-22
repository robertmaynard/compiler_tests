
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(multi_std_compile_flags CUDA)

add_executable(multi_std_compile_flags main.cu)

#this needs to be the long and short forms so that CMake doesn't de-duplicate them
target_compile_options(multi_std_compile_flags PRIVATE -std=c++11 --std=c++11)
