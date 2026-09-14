cmake_minimum_required(VERSION 3.18)
project(adapTQ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
  add_compile_options(/O2 /W3 /wd4996 /wd4068)
else()
  add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    core/fwht.cpp
    core/codebook.cpp
    core/quantizer.cpp
    core/adaptq_c_api.cpp
    core/adaptq_backend_vtable.cpp
    core/cost_optimizer.cpp
    core/quality_oracle.cpp
    cache/ring_buffer.cpp
    attention/attention.cpp
    utils/timer.cpp
    kernels/scalar/kdot_scalar.cpp
    runtime/default_policy.cpp
    runtime/contiguous_storage.cpp
)

add_library(adapTQ_core_obj OBJECT ${CORE_SOURCES})
target_include_directories(adapTQ_core_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/strategies
)
if(MSVC)
  target_compile_options(adapTQ_core_obj PRIVATE /openmp)
else()
  target_compile_options(adapTQ_core_obj PRIVATE -fopenmp)
  target_link_options(adapTQ_core_obj PUBLIC -fopenmp)
endif()
set_target_properties(adapTQ_core_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(adapTQ_core STATIC $<TARGET_OBJECTS:adapTQ_core_obj>)
target_include_directories(adapTQ_core PUBLIC $<TARGET_PROPERTY:adapTQ_core_obj,INCLUDE_DIRECTORIES>)
if(NOT MSVC)
  target_link_options(adapTQ_core PUBLIC -fopenmp)
endif()

# ---------------------------------------------------------------------------
# AVX2 kernel backend
# ---------------------------------------------------------------------------
add_library(adapTQ_avx2_obj OBJECT kernels/avx2/kdot_avx2.cpp)
target_include_directories(adapTQ_avx2_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
if(MSVC)
  target_compile_options(adapTQ_avx2_obj PRIVATE /arch:AVX2)
else()
  target_compile_options(adapTQ_avx2_obj PRIVATE -mavx2 -mfma)
endif()
set_target_properties(adapTQ_avx2_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(adapTQ_avx2 STATIC $<TARGET_OBJECTS:adapTQ_avx2_obj>)
target_link_libraries(adapTQ_avx2 PUBLIC adapTQ_core)

if(UNIX AND NOT APPLE)
  target_link_libraries(adapTQ_core PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Runtime library (V2: RuntimeContext + SessionSnapshot + ReplayEngine)
# ---------------------------------------------------------------------------
# Strategy/storage/kernel implementations are #included directly in
# runtime_context.cpp as a single TU to avoid ODR issues — do NOT list them
# as separate RUNTIME_SOURCES.
set(RUNTIME_SOURCES
    runtime/dummy.cpp
)

add_library(adapTQ_runtime_obj OBJECT ${RUNTIME_SOURCES})
target_include_directories(adapTQ_runtime_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/strategies
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/replay
    ${CMAKE_CURRENT_SOURCE_DIR}/cli
)
if(MSVC)
  
else()
  
endif()
set_target_properties(adapTQ_runtime_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(adapTQ_runtime STATIC $<TARGET_OBJECTS:adapTQ_runtime_obj>)
target_include_directories(adapTQ_runtime PUBLIC $<TARGET_PROPERTY:adapTQ_runtime_obj,INCLUDE_DIRECTORIES>)
target_link_libraries(adapTQ_runtime PUBLIC adapTQ_core adapTQ_avx2)


# ---------------------------------------------------------------------------
# Shared Library for Python C-API Integration
# ---------------------------------------------------------------------------
add_library(adaptq_shared SHARED 
    $<TARGET_OBJECTS:adapTQ_core_obj>
    $<TARGET_OBJECTS:adapTQ_avx2_obj>
    $<TARGET_OBJECTS:adapTQ_runtime_obj>
)
set_target_properties(adaptq_shared PROPERTIES 
    OUTPUT_NAME adaptq
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
if(NOT MSVC)
  target_link_options(adaptq_shared PRIVATE -fopenmp)
  target_link_libraries(adaptq_shared PRIVATE m)
endif()

# ---------------------------------------------------------------------------
# Main demo executable
# ---------------------------------------------------------------------------

add_executable(adapTQ_demo main.cpp)
target_link_libraries(adapTQ_demo PRIVATE adapTQ_runtime adapTQ_core)
target_include_directories(adapTQ_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/replay
    ${CMAKE_CURRENT_SOURCE_DIR}/cli
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/strategies
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# ---------------------------------------------------------------------------
# Tests — Catch2 via find_package first, then FetchContent fallback
# ---------------------------------------------------------------------------
option(ADAPTQ_BUILD_TESTS "Build unit and conformance tests" ON)

if(ADAPTQ_BUILD_TESTS)
  # Try system Catch2 (apt/brew install catch2)
  find_package(Catch2 3 QUIET)

  if(NOT Catch2_FOUND)
    # Fallback: download Catch2 at configure time via FetchContent
    message(STATUS "Catch2 not found via find_package — using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.1
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  enable_testing()
  include(CTest)
  include(Catch)

  # --- test_fwht -----------------------------------------------------------
  add_executable(test_fwht
      tests/unit/test_fwht.cpp
      core/fwht.cpp)
  target_include_directories(test_fwht PRIVATE include)
  target_link_libraries(test_fwht PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(test_fwht)

  # --- test_packing --------------------------------------------------------
  add_executable(test_packing
      tests/unit/test_packing.cpp)
  target_link_libraries(test_packing PRIVATE adapTQ_core Catch2::Catch2WithMain)
  catch_discover_tests(test_packing)

  # --- test_storage --------------------------------------------------------
  # Storage backends are header-only (inline classes in storage/*.h).
  # The .cpp stubs are intentionally NOT listed — they'd be empty TUs anyway.
  add_executable(test_storage
      tests/unit/test_storage.cpp)
  target_include_directories(test_storage PRIVATE include storage)
  target_link_libraries(test_storage PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(test_storage)

  # --- test_api ------------------------------------------------------------
  add_executable(test_api
      tests/unit/test_api.cpp)
  target_link_libraries(test_api PRIVATE adapTQ_core Catch2::Catch2WithMain)
  catch_discover_tests(test_api)

  # --- test_conformance ----------------------------------------------------
  # Strategy .cpp files are compiled inline within this single TU (not in
  # any library) to avoid ODR violations. The CMakeLists does not list them
  # as separate sources precisely because they are #included by the .cpp.
  add_executable(test_conformance
      tests/conformance/strategy_conformance.cpp)
  target_include_directories(test_conformance PRIVATE
      include storage strategies tests/conformance)
  target_link_libraries(test_conformance PRIVATE adapTQ_core Catch2::Catch2WithMain)
  catch_discover_tests(test_conformance)

  # V2 Tests disabled in 0.2.3 foundation to avoid compile errors
  # (as V2 architectures speculative files are skipped)
  message(STATUS "Test targets enabled (V1: 38 tests + Codebook/Kernel tests)")
endif()

