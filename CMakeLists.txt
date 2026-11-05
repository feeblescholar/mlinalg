cmake_minimum_required(VERSION 3.25)

project(mlinalg LANGUAGES CXX)

option(MLINALG_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
set(MLINALG_SANITIZER "" CACHE STRING "Value for -fsanitize=, e.g. address,undefined or thread")

add_library(mlinalg INTERFACE)
add_library(mlinalg::mlinalg ALIAS mlinalg)

target_include_directories(mlinalg INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(mlinalg INTERFACE cxx_std_23)

# packet.hpp silently falls back to scalar code if these are missing, so consumers inherit them.
# NEON is part of the AArch64 baseline and needs no flag.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(mlinalg INTERFACE -mavx2 -mfma)

    # MinGW GCC does not realign the stack for 32-byte locals (GCC PR 54412), so aligned AVX moves
    # of stack values can fault. This makes the assembler emit the unaligned forms instead.
    if(MINGW)
        target_compile_options(mlinalg INTERFACE -Wa,-muse-unaligned-vector-move)
    endif()
endif()

if(MLINALG_SANITIZER)
    if(WIN32)
        message(FATAL_ERROR "MLINALG_SANITIZER is not supported on Windows")
    endif()

    # Directory-scoped so GoogleTest is instrumented too: TSan reports false positives otherwise.
    # UBSan only logs and carries on by default, which would leave the test passing.
    add_compile_options(-fsanitize=${MLINALG_SANITIZER} -fno-sanitize-recover=undefined
                        -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${MLINALG_SANITIZER} -fno-sanitize-recover=undefined)
endif()

if(MLINALG_BUILD_TESTS)
    enable_testing()

    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(lib/googletest EXCLUDE_FROM_ALL SYSTEM)

    add_subdirectory(test)
endif()
