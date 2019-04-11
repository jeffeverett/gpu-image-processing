cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(hw10)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)

list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fopenmp")

# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCL REQUIRED)

# Populate a CMake variable with the sources
file(GLOB hw10_srcs
    "*.hpp"
    "*.cpp"
    "*.cu"
)
# Tell CMake to create the executable
cuda_add_executable(hw10 ${hw10_srcs})
set_target_properties(hw10
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(hw10 PUBLIC cxx_std_11)

# Link needed libraries
target_link_libraries(hw10 Qt5::Widgets)
target_link_libraries(hw10 gomp)
target_link_libraries(hw10 OpenCL::OpenCL)