cmake_minimum_required(VERSION 3.8)

set(project_name bambooSMP)
set(version 2024.7)

project(${project_name}
        VERSION ${version}
        LANGUAGES C CXX CUDA
)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_ARCHITECTURES 80)


# both source files and test files 
# will try to include header files from this directory
set(CUDA_HOME $ENV{CUDA_HOME})
set(HOME $ENV{HOME})

include_directories(${PROJECT_SOURCE_DIR}/include)
# export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# src files
add_subdirectory(src)

add_executable(bamboosmpTest test/testbase.cpp test/testbase.h
        test/bamboosmpTest.cu)
target_link_libraries(bamboosmpTest bamboosmp_lib)
