cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nvcc_ VERSION 0.1.0 LANGUAGES C CXX CUDA)

enable_testing()
include(../cmake/compiler_tests.cmake)


add_compile_test(CUDA at_c)
add_compile_test(CUDA optfile_quote_parsing)
