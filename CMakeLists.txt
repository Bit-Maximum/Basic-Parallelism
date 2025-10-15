cmake_minimum_required(VERSION 3.28)
project(Parallelism)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")

add_executable(Parallelism main.cpp
)
