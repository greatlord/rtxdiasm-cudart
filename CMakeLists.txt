cmake_minimum_required(VERSION 3.21)
project(rtxdiasm_guard LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA Runtime API shim (cudart) split from ACuda.
# Builds requested output names:
#   Windows x64:   cudart64_13.dll
#   Windows x86:   cudart32_13.dll
#   Linux:         libcudart.so, libcudart.so.13, libcudart.so.13.0.96

set(RTXCUDART_SOURCES
  src/cudart_device.c
  src/cudart_error.c
  src/cudart_init.c
  src/cudart_memory.c
  src/cudart_module_fatbin.c
  src/cudart_module_launch.c
  src/cudart_module_load.c
  src/cudart_stream_event.c
  src/cudart_stubs.c
  src/cudart_texture.c
)

# --- External: lz4 (v1.11.1 submodule) for fatbin LZ4 extraction ---
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
add_subdirectory(external/lz4/build/cmake ${CMAKE_CURRENT_BINARY_DIR}/external/lz4 EXCLUDE_FROM_ALL)
if(TARGET lz4_static)
  set(RTXCUDART_LZ4_TARGET lz4_static)
elseif(TARGET LZ4::lz4_static)
  set(RTXCUDART_LZ4_TARGET LZ4::lz4_static)
elseif(TARGET lz4)
  set(RTXCUDART_LZ4_TARGET lz4)
else()
  message(FATAL_ERROR "lz4 library target not found in external/lz4 submodule")
endif()

if(NOT WIN32)
  # Trace hook source may be provided by the integrating build (e.g. acuda driver trace.c).
  if(DEFINED ACUDA_TRACE_SOURCE AND NOT "${ACUDA_TRACE_SOURCE}" STREQUAL "")
    list(APPEND RTXCUDART_SOURCES ${ACUDA_TRACE_SOURCE})
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/trace.c")
    list(APPEND RTXCUDART_SOURCES src/trace.c)
  endif()
endif()

add_library(rtxcudart SHARED ${RTXCUDART_SOURCES})

target_include_directories(rtxcudart PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(rtxcudart PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/external/lz4/lib
)

target_link_libraries(rtxcudart PRIVATE ${RTXCUDART_LZ4_TARGET})

# When built inside ACuda, link the in-tree driver target.
if(TARGET acuda_driver)
  target_link_libraries(rtxcudart PRIVATE acuda_driver)
elseif(TARGET cuda)
  target_link_libraries(rtxcudart PRIVATE cuda)
endif()

# Output naming
set(RTXCUDART_OUTPUT_NAME cudart)
if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(RTXCUDART_OUTPUT_NAME cudart32_13)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Use 64-bit naming for x64/ARM64 builds; arch-specific staging is handled by integrator.
    set(RTXCUDART_OUTPUT_NAME cudart64_13)
  endif()
endif()

set_target_properties(rtxcudart PROPERTIES
  OUTPUT_NAME ${RTXCUDART_OUTPUT_NAME}
)

if(NOT WIN32)
  set_target_properties(rtxcudart PROPERTIES
    VERSION 13.40
    SOVERSION 13
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
  target_link_options(rtxcudart PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/cudart.map"
    "LINKER:-Bsymbolic"
  )
  if(DEFINED ACUDA_TRACE_SOURCE AND NOT "${ACUDA_TRACE_SOURCE}" STREQUAL "")
    target_compile_options(rtxcudart PRIVATE
      -finstrument-functions
      -finstrument-functions-exclude-file-list=trace.c
    )
  endif()
else()
  set_target_properties(rtxcudart PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
endif()

# Warnings: inherit from integrator if provided, else reasonable defaults.
if(DEFINED ACUDA_WARNINGS)
  target_compile_options(rtxcudart PRIVATE ${ACUDA_WARNINGS})
else()
  if(MSVC)
    target_compile_options(rtxcudart PRIVATE /W3)
    target_compile_definitions(rtxcudart PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(rtxcudart PRIVATE -Wall -Wextra)
  endif()
endif()

# Compatibility target name for ACuda integration/tests.
if(NOT TARGET acuda_cudart)
  add_library(acuda_cudart ALIAS rtxcudart)
endif()
