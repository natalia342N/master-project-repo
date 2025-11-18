cmake_minimum_required(VERSION 3.12)

project(queuetest LANGUAGES CXX CUDA)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUDA_BUILD_TARGET_ARCHITECTURES "75" CACHE STRING "Semicolon-separated list of target architectures to build for")
option(CUDA_BUILD_INFO "build with kernel statistics and line numbers" FALSE)
option(CUDA_BUILD_DEBUG "build with kernel debug" FALSE)

# ============================================================
# Existing target: queuetest
# ============================================================
add_executable(queuetest
    include/helper.h
    include/helper.cuh
    include/kernels.cuh
    include/datatypes.h
    include/queuetest.cuh
    include/queues/broker.cuh
    include/queues/gottlieb.cuh
    include/queues/bwd.cuh
    source/main.cpp
    source/queuetest.cu
)

set_target_properties(queuetest PROPERTIES
    CXX_STANDARD 14
    CUDA_ARCHITECTURES "${CUDA_BUILD_TARGET_ARCHITECTURES}"
    CUDA_STANDARD 14
)

# Add support for driver API
target_link_libraries(queuetest cuda)

if (CUDA_BUILD_INFO)
    target_compile_options(queuetest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-keep --ptxas-options=-v -lineinfo>)
endif ()

if (CUDA_BUILD_DEBUG)
    target_compile_options(queuetest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif ()


# ============================================================
# NEW target: taskmanager_test (GPU broker-queue task manager)
# ============================================================
add_executable(taskmanager_test
    include/helper.h
    include/helper.cuh
    include/kernels.cuh
    include/datatypes.h

    include/queues/broker.cuh
    include/queues/job_definitions.cuh
    include/queues/job_kernels.cuh
    include/queues/taskmanager_gpu.cuh
    include/queues/taskmanager_gpu.cu

    source/main_taskmanager.cu
)

set_target_properties(taskmanager_test PROPERTIES
    CXX_STANDARD 14
    CUDA_ARCHITECTURES "${CUDA_BUILD_TARGET_ARCHITECTURES}"
    CUDA_STANDARD 14
)

target_link_libraries(taskmanager_test cuda)

if (CUDA_BUILD_INFO)
    target_compile_options(taskmanager_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-keep --ptxas-options=-v -lineinfo>)
endif ()

if (CUDA_BUILD_DEBUG)
    target_compile_options(taskmanager_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif ()
