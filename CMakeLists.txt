cmake_minimum_required(VERSION 3.16)
project(adapTQ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
  add_compile_options(/O2 /W3)
else()
  add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

# Core library sources
set(CORE_SOURCES
    core/fwht.cpp
    core/codebook.cpp
    core/quantizer.cpp
    cache/ring_buffer.cpp
    attention/attention.cpp
    utils/timer.cpp
)

add_library(adapTQ_core STATIC ${CORE_SOURCES})
target_include_directories(adapTQ_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main executable
add_executable(adapTQ_demo main.cpp)
target_link_libraries(adapTQ_demo adapTQ_core)
target_include_directories(adapTQ_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link math library on Linux
if(UNIX AND NOT APPLE)
  target_link_libraries(adapTQ_core m)
endif()
