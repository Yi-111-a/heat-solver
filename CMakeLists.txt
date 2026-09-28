cmake_minimum_required(VERSION 3.18)
project(heat_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(heat src/main.cpp)
target_link_libraries(heat PRIVATE OpenMP::OpenMP_CXX)
# No -ffast-math: the float/double comparison needs standard IEEE arithmetic.
target_compile_options(heat PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)

# The GPU back end is built only when a CUDA compiler is available.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  # "native" compiles for the GPU in this machine; override e.g. with -DHEAT_CUDA_ARCH=75 on Colab.
  set(HEAT_CUDA_ARCH native CACHE STRING "CUDA architectures to build for")
  set_target_properties(heat PROPERTIES CUDA_ARCHITECTURES "${HEAT_CUDA_ARCH}")
  target_sources(heat PRIVATE src/heat_cuda.cu)
  target_compile_definitions(heat PRIVATE HEAT_WITH_CUDA)
  target_compile_options(heat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  message(STATUS "CUDA found: building the GPU back end")
else()
  message(STATUS "CUDA not found: CPU back ends only")
endif()
