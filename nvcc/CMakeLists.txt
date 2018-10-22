cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nvcc_ VERSION 0.1.0 LANGUAGES C CXX CUDA)

enable_testing()
include(../cmake/compiler_tests.cmake)

#Tests that need to be compiled and a test run
add_compile_run_test(CUDA copy_if_with_large_obj)


#Tests that just need to be compiled
add_compile_test(CUDA argument_pack)
add_compile_test(CUDA at_c)
add_compile_test(CUDA curand_thrust_separable_comp)
add_compile_test(CUDA function_scope__host_device_class)
add_compile_test(CUDA generate_deps_with_spaces)
add_compile_test(CUDA list_for_each_constructor)
add_compile_test(CUDA multi_std_compile_flags)
add_compile_test(CUDA optfile_quote_parsing)
add_compile_test(CUDA particle_advec_ice_ptxas)
