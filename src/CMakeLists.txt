# Get list of all source files with given pattern
# file(GLOB_RECURSE bamboosmp_sources "${PROJECT_SOURCE_DIR}/src/*.cu")

# iterate each source file to create library file individually
# add_library(bamboosmp_lib SHARED ${bamboosmp_sources})
# set_property(TARGET bamboosmp_lib PROPERTY CUDA_ARCHITECTURES 80)

# Get list of all source files with given pattern
file(GLOB_RECURSE bamboosmp_sources "${PROJECT_SOURCE_DIR}/src/*.cu")

# iterate each source file to create object files individually
# add_library(bamboosmp_lib OBJECT ${bamboosmp_sources})
add_library(bamboosmp_lib SHARED ${bamboosmp_sources}
        smp/BambooSMPExecCPUSingleGPU.cu)

# set_property(TARGET bamboosmp_lib PROPERTY CUDA_ARCHITECTURES 70)
set_property(TARGET bamboosmp_lib PROPERTY CUDA_ARCHITECTURES 80)

target_compile_options(bamboosmp_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Add the new workload generator executable
add_executable(bamboosmp
        "${PROJECT_SOURCE_DIR}/src/bamboosmp/bamboosmp.cu"
)

# Link the new executable with the necessary libraries
target_link_libraries(bamboosmp PRIVATE
        bamboosmp_lib
        pthread)


