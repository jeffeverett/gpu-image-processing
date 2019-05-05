cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(gpu-image-processing)

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
file(GLOB gpu-image-processing_srcs
    "*.hpp"
    "*.cpp"
    "*.cu"
)
# Tell CMake to create the executable
cuda_add_executable(gpu-image-processing ${gpu-image-processing_srcs})
set_target_properties(gpu-image-processing
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(gpu-image-processing PUBLIC cxx_std_11)

# Link needed libraries
target_link_libraries(gpu-image-processing Qt5::Widgets)
target_link_libraries(gpu-image-processing gomp)
target_link_libraries(gpu-image-processing OpenCL::OpenCL)