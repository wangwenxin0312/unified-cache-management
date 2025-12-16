cmake_minimum_required(VERSION 3.18)
project(ucm_sm_copy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE Torch_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Torch cmake prefix: ${Torch_CMAKE_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${Torch_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

get_filename_component(TORCH_SHARE_DIR "${Torch_CMAKE_PREFIX}" DIRECTORY)  # .../torch/share
get_filename_component(TORCH_ROOT_DIR  "${TORCH_SHARE_DIR}" DIRECTORY)     # .../torch
set(TORCH_LIB_DIR "${TORCH_ROOT_DIR}/lib")

message(STATUS "Torch lib dir (guessed): ${TORCH_LIB_DIR}")

find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED)

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
# ===============================================================

add_library(ucm_sm_copy MODULE
    cuda_sm_pybind.cpp
    cuda_sm_kernel.cu
)

target_include_directories(ucm_sm_copy PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(ucm_sm_copy PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Python3::Python
    CUDA::cudart
)

set_target_properties(ucm_sm_copy PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
