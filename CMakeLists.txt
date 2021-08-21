cmake_minimum_required(VERSION 3.20)
project(joiss2021 CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_FLAGS -arch=sm_75 -std=c++23 -Wall)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "C:/UserProgramFiles/opencv")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

# add_executable(joiss2021 main.cu)
add_executable(joiss2021 main.cu)

# linking
target_link_libraries(joiss2021 ${OpenCV_LIBS})
#target_link_libraries(joiss2021 C:/UserProgramFiles/opencv/build/x64/vc15/lib/opencv_world453.lib)
#target_link_libraries(joiss2021 C:/UserProgramFiles/opencv/build/x64/vc15/lib/opencv_world453d.lib)
target_link_libraries(joiss2021 C:/UserProgramFiles/opencv/install/x64/vc16/lib/opencv_world453.lib)
target_link_libraries(joiss2021 C:/UserProgramFiles/opencv/install/x64/vc16/lib/opencv_world453d.lib)
set_target_properties(joiss2021 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

