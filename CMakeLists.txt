cmake_minimum_required(VERSION 3.0)
project(atan2)

set(CMAKE_CXX_STANDARD 11)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fstack-protector-all)
    add_compile_options(-Wall -Wextra -pedantic)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-ffp-contract=fast -ffast-math -freciprocal-math -fmerge-all-constants)
    add_compile_options(-O3 -march=native -funroll-loops -finline-functions -ftree-vectorize)
endif ()

add_executable(atan2
        atan2.cpp)
