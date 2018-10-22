
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(generate_deps_with_spaces CUDA)


set(src "${CMAKE_CURRENT_SOURCE_DIR}/dir\ with\ spaces/main.cu")
set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/dir\ with\ spaces/")

execute_process(COMMAND ${CMAKE_CUDA_COMPILER} ${src} -I ${include_path} --generate-dependencies
                OUTPUT_VARIABLE text)


string(FIND ${text} "main.cu" first)
string(FIND ${text} "main.cu" last REVERSE)

message(STATUS "first loc ${first}")
message(STATUS "last loc ${last}")

if(NOT first EQUAL last)
  message(FATAL_ERROR "nvcc generate-dependencies doesn't handle spaces in path")
endif()
