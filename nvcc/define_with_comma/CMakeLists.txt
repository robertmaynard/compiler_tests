

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(def_with_comma CUDA)

add_executable(def_with_comma main.cu)
target_compile_definitions(def_with_comma PRIVATE [==[PRAG_PUSH_WITH_WORKAROUND=__pragma(warning(push\,0))]==])
target_compile_definitions(def_with_comma PRIVATE [==[PRAG_PUSH_NO_WORKAROUND=__pragma(warning(push,0))]==])
target_compile_options(def_with_comma PRIVATE "-v")
