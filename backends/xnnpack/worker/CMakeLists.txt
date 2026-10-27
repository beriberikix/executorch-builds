# Builds xnnpack-text-worker against an installed ExecuTorch (the `llm`
# preset) plus its source tree, which provides the llm_server worker loop
# headers that are not part of the install.
#
#   cmake -S . -B build \
#     -DEXECUTORCH_SOURCE_DIR=/path/to/executorch \
#     -DCMAKE_PREFIX_PATH=/path/to/executorch-install

cmake_minimum_required(VERSION 3.24) # WHOLE_ARCHIVE
project(xnnpack_text_worker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_SOURCE_DIR
    ""
    CACHE PATH "ExecuTorch source tree (directory must be named executorch)"
)
if(NOT EXISTS ${EXECUTORCH_SOURCE_DIR}/examples/llm_server/cpp/worker_loop.h)
  message(FATAL_ERROR "Set EXECUTORCH_SOURCE_DIR to an ExecuTorch checkout")
endif()
get_filename_component(_et_dir_name ${EXECUTORCH_SOURCE_DIR} NAME)
if(NOT _et_dir_name STREQUAL "executorch")
  message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR must end in /executorch")
endif()

include(${EXECUTORCH_SOURCE_DIR}/tools/cmake/Utils.cmake)

find_package(executorch CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Kernel and backend libraries register themselves from static initializers,
# so they are linked whole-archive.
set(_whole_archive_libs
    executorch
    optimized_native_cpu_ops_lib
    quantized_ops_lib
    custom_ops
    xnnpack_backend
)
foreach(_lib ${_whole_archive_libs})
  if(NOT TARGET ${_lib})
    message(FATAL_ERROR "ExecuTorch install is missing ${_lib}; "
                        "build it with the llm preset")
  endif()
  executorch_target_link_options_shared_lib(${_lib})
endforeach()

set(link_libraries
    ${_whole_archive_libs}
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
    quantized_kernels
    extension_llm_runner
    extension_module
    extension_data_loader
    extension_tensor
    extension_flat_tensor
    extension_threadpool
    pthreadpool
    cpuinfo
    XNNPACK
    xnnpack-microkernels-prod
    tokenizers::tokenizers
    Threads::Threads
)
if(TARGET kleidiai)
  list(APPEND link_libraries kleidiai)
endif()
if(TARGET extension_llm_session)
  list(APPEND link_libraries extension_llm_session)
endif()

add_executable(
  xnnpack-text-worker xnnpack_text_worker.cpp xnnpack_text_engine.cpp
)
target_include_directories(
  xnnpack-text-worker
  PRIVATE
    ${EXECUTORCH_SOURCE_DIR}/..
    ${EXECUTORCH_SOURCE_DIR}/extension/llm/tokenizers/third-party/json/single_include
)
target_compile_options(xnnpack-text-worker PRIVATE -Wno-deprecated-declarations)
target_link_libraries(xnnpack-text-worker PRIVATE ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(xnnpack-text-worker)
  target_link_options(xnnpack-text-worker PRIVATE "LINKER:-s")
endif()

install(TARGETS xnnpack-text-worker RUNTIME DESTINATION bin)

option(BUILD_LOGITS_CHECK "Build logits-check for the numerics gate" OFF)
if(BUILD_LOGITS_CHECK)
  add_executable(logits-check logits_check.cpp)
  target_include_directories(
    logits-check
    PRIVATE
      ${EXECUTORCH_SOURCE_DIR}/..
      ${EXECUTORCH_SOURCE_DIR}/extension/llm/tokenizers/third-party/json/single_include
  )
  target_link_libraries(logits-check PRIVATE ${link_libraries})
endif()
