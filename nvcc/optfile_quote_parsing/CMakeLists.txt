
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(optfile_quote_parsing CUDA)

add_executable(optfile_quote_parsing main.cu)
target_compile_options(optfile_quote_parsing PRIVATE "-optf=${CMAKE_CURRENT_SOURCE_DIR}/response_file")
