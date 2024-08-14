cmake_minimum_required(VERSION 2.8)
project(ImageMatching)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++11;-DVERBOSE")
#list(APPEND CUDA_NVCC_FLAGS "-arch=compute_20;-std=c++11;-DVERBOSE")
#set(CUDA_ARCH "-arch=compute_20" CACHE STRING "Value of the NVCC -arch option.")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

file(GLOB  cu  *.cu *.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})  
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

#set(SOURCE_FILES main.cpp filter.cpp keyPointDetector.cpp brief.cpp stitcher.cpp)
CUDA_ADD_EXECUTABLE(ImageMatching src/main.cpp 
                             src/filter.cpp 
                             src/keyPointDetector.cpp 
                             src/brief.cpp 
                             src/Util.cpp 
                             src/bitarray.cpp 
                             src/cudaFilter.cu 
                             src/cudaMatcher.cu)
#add_executable(ImageMatching ${SOURCE_FILES})

target_link_libraries(ImageMatching ${CUDA_LIBRARIES} ${OpenCV_LIBS})
