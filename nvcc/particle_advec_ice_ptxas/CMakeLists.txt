
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(partice_advec_ice_ptxas CUDA)

#get the location of PTXAS
get_filename_component(ptxas_location ${CMAKE_CUDA_COMPILER} DIRECTORY)


execute_process(COMMAND ${ptxas_location}/ptxas
                        ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestParticleAdvection.ptx
                        --output-file ${CMAKE_CURRENT_BINARY_DIR}/builds.cubin
                        -arch=sm_60
                        --opt-level 3
                        )
