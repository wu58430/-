cmake_minimum_required(VERSION 3.22)
project(Dedistortion)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(Dedistortion main.cpp ty.h file.h file.cpp all.cpp all.h distorted.cpp distorted.h)

target_link_libraries(Dedistortion ${OpenCV_LIBS})



