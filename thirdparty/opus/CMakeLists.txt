cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(OpusPackageVersion)
get_package_version(PACKAGE_VERSION PROJECT_VERSION)

project(Opus LANGUAGES C VERSION ${PROJECT_VERSION})

include(OpusFunctions)
include(OpusBuildtype)
include(OpusConfig)
include(OpusSources)
include(GNUInstallDirs)
include(CMakeDependentOption)
include(FeatureSummary)

# Detect clang-cl
set(MSVC_CLANG_CL false BOOL)
if (MSVC)
    if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
        set(MSVC_CLANG_CL true BOOL)
    endif()
endif()

set(OPUS_BUILD_SHARED_LIBRARY_HELP_STR "build shared library.")
option(OPUS_BUILD_SHARED_LIBRARY ${OPUS_BUILD_SHARED_LIBRARY_HELP_STR} OFF)
if(OPUS_BUILD_SHARED_LIBRARY OR BUILD_SHARED_LIBS OR OPUS_BUILD_FRAMEWORK)
  # Global flag to cause add_library() to create shared libraries if on.
  set(BUILD_SHARED_LIBS ON)
  set(OPUS_BUILD_SHARED_LIBRARY ON)
endif()
add_feature_info(OPUS_BUILD_SHARED_LIBRARY OPUS_BUILD_SHARED_LIBRARY ${OPUS_BUILD_SHARED_LIBRARY_HELP_STR})

set(OPUS_BUILD_TESTING_HELP_STR "build tests.")
option(OPUS_BUILD_TESTING ${OPUS_BUILD_TESTING_HELP_STR} OFF)
if(OPUS_BUILD_TESTING OR BUILD_TESTING)
  set(OPUS_BUILD_TESTING ON)
  set(BUILD_TESTING ON)
endif()
add_feature_info(OPUS_BUILD_TESTING OPUS_BUILD_TESTING ${OPUS_BUILD_TESTING_HELP_STR})

set(OPUS_CUSTOM_MODES_HELP_STR "enable non-Opus modes, e.g. 44.1 kHz & 2^n frames.")
option(OPUS_CUSTOM_MODES ${OPUS_CUSTOM_MODES_HELP_STR} OFF)
add_feature_info(OPUS_CUSTOM_MODES OPUS_CUSTOM_MODES ${OPUS_CUSTOM_MODES_HELP_STR})

set(OPUS_BUILD_PROGRAMS_HELP_STR "build programs.")
option(OPUS_BUILD_PROGRAMS ${OPUS_BUILD_PROGRAMS_HELP_STR} OFF)
add_feature_info(OPUS_BUILD_PROGRAMS OPUS_BUILD_PROGRAMS ${OPUS_BUILD_PROGRAMS_HELP_STR})

set(OPUS_DISABLE_INTRINSICS_HELP_STR "disable all intrinsics optimizations.")
option(OPUS_DISABLE_INTRINSICS ${OPUS_DISABLE_INTRINSICS_HELP_STR} OFF)
add_feature_info(OPUS_DISABLE_INTRINSICS OPUS_DISABLE_INTRINSICS ${OPUS_DISABLE_INTRINSICS_HELP_STR})

set(OPUS_FIXED_POINT_HELP_STR "compile as fixed-point (for machines without a fast enough FPU).")
option(OPUS_FIXED_POINT ${OPUS_FIXED_POINT_HELP_STR} OFF)
add_feature_info(OPUS_FIXED_POINT OPUS_FIXED_POINT ${OPUS_FIXED_POINT_HELP_STR})

set(OPUS_ENABLE_FLOAT_API_HELP_STR "compile with the floating point API (for machines with float library).")
option(OPUS_ENABLE_FLOAT_API ${OPUS_ENABLE_FLOAT_API_HELP_STR} ON)
add_feature_info(OPUS_ENABLE_FLOAT_API OPUS_ENABLE_FLOAT_API ${OPUS_ENABLE_FLOAT_API_HELP_STR})

set(OPUS_FLOAT_APPROX_HELP_STR "enable floating point approximations (Ensure your platform supports IEEE 754 before enabling).")
option(OPUS_FLOAT_APPROX ${OPUS_FLOAT_APPROX_HELP_STR} OFF)
add_feature_info(OPUS_FLOAT_APPROX OPUS_FLOAT_APPROX ${OPUS_FLOAT_APPROX_HELP_STR})

set(OPUS_ASSERTIONS_HELP_STR "additional software error checking.")
option(OPUS_ASSERTIONS ${OPUS_ASSERTIONS_HELP_STR} OFF)
add_feature_info(OPUS_ASSERTIONS OPUS_ASSERTIONS ${OPUS_ASSERTIONS_HELP_STR})

set(OPUS_HARDENING_HELP_STR "run-time checks that are cheap and safe for use in production.")
option(OPUS_HARDENING ${OPUS_HARDENING_HELP_STR} ON)
add_feature_info(OPUS_HARDENING OPUS_HARDENING ${OPUS_HARDENING_HELP_STR})

set(OPUS_FUZZING_HELP_STR "causes the encoder to make random decisions (do not use in production).")
option(OPUS_FUZZING ${OPUS_FUZZING_HELP_STR} OFF)
add_feature_info(OPUS_FUZZING OPUS_FUZZING ${OPUS_FUZZING_HELP_STR})

set(OPUS_CHECK_ASM_HELP_STR "enable bit-exactness checks between optimized and c implementations.")
option(OPUS_CHECK_ASM ${OPUS_CHECK_ASM_HELP_STR} OFF)
add_feature_info(OPUS_CHECK_ASM OPUS_CHECK_ASM ${OPUS_CHECK_ASM_HELP_STR})

set(OPUS_DNN_FLOAT_DEBUG_HELP_STR "Run DNN computations as float for debugging purposes.")
option(OPUS_DNN_FLOAT_DEBUG ${OPUS_DNN_FLOAT_DEBUG_HELP_STR} OFF)
add_feature_info(OPUS_DNN_FLOAT_DEBUG OPUS_DNN_FLOAT_DEBUG ${OPUS_DNN_FLOAT_DEBUG_HELP_STR})

set(OPUS_INSTALL_PKG_CONFIG_MODULE_HELP_STR "install pkg-config module.")
option(OPUS_INSTALL_PKG_CONFIG_MODULE ${OPUS_INSTALL_PKG_CONFIG_MODULE_HELP_STR} ON)
add_feature_info(OPUS_INSTALL_PKG_CONFIG_MODULE OPUS_INSTALL_PKG_CONFIG_MODULE ${OPUS_INSTALL_PKG_CONFIG_MODULE_HELP_STR})

set(OPUS_INSTALL_CMAKE_CONFIG_MODULE_HELP_STR "install CMake package config module.")
option(OPUS_INSTALL_CMAKE_CONFIG_MODULE ${OPUS_INSTALL_CMAKE_CONFIG_MODULE_HELP_STR} ON)
add_feature_info(OPUS_INSTALL_CMAKE_CONFIG_MODULE OPUS_INSTALL_CMAKE_CONFIG_MODULE ${OPUS_INSTALL_CMAKE_CONFIG_MODULE_HELP_STR})

set(OPUS_DRED_HELP_STR "enable DRED.")
option(OPUS_DRED ${OPUS_DRED_HELP_STR} OFF)
add_feature_info(OPUS_DRED OPUS_DRED ${OPUS_DRED_HELP_STR})

set(OPUS_OSCE_HELP_STR "enable OSCE.")
option(OPUS_OSCE ${OPUS_OSCE_HELP_STR} OFF)
add_feature_info(OPUS_OSCE OPUS_OSCE ${OPUS_OSCE_HELP_STR})

if(APPLE)
  set(OPUS_BUILD_FRAMEWORK_HELP_STR "build Framework bundle for Apple systems.")
  option(OPUS_BUILD_FRAMEWORK ${OPUS_BUILD_FRAMEWORK_HELP_STR} OFF)
  add_feature_info(OPUS_BUILD_FRAMEWORK OPUS_BUILD_FRAMEWORK ${OPUS_BUILD_FRAMEWORK_HELP_STR})
endif()

if(MSVC)
  set(OPUS_STATIC_RUNTIME_HELP_STR "build with static runtime library.")
  option(OPUS_STATIC_RUNTIME ${OPUS_STATIC_RUNTIME_HELP_STR} OFF)
  add_feature_info(OPUS_STATIC_RUNTIME OPUS_STATIC_RUNTIME ${OPUS_STATIC_RUNTIME_HELP_STR})
endif()

set(OPUS_FIXED_POINT_DEBUG_HELP_STR "debug fixed-point implementation.")
cmake_dependent_option(OPUS_FIXED_POINT_DEBUG
                      ${OPUS_FIXED_POINT_DEBUG_HELP_STR}
                      ON
                      "OPUS_FIXED_POINT; OPUS_FIXED_POINT_DEBUG"
                      OFF)
add_feature_info(OPUS_FIXED_POINT_DEBUG OPUS_FIXED_POINT_DEBUG ${OPUS_FIXED_POINT_DEBUG_HELP_STR})

set(OPUS_VAR_ARRAYS_HELP_STR "use variable length arrays for stack arrays.")
cmake_dependent_option(OPUS_VAR_ARRAYS
                      ${OPUS_VAR_ARRAYS_HELP_STR}
                      ON
                      "VLA_SUPPORTED; NOT OPUS_USE_ALLOCA; NOT OPUS_NONTHREADSAFE_PSEUDOSTACK"
                      OFF)
add_feature_info(OPUS_VAR_ARRAYS OPUS_VAR_ARRAYS ${OPUS_VAR_ARRAYS_HELP_STR})

set(OPUS_USE_ALLOCA_HELP_STR "use alloca for stack arrays (on non-C99 compilers).")
cmake_dependent_option(OPUS_USE_ALLOCA
                       ${OPUS_USE_ALLOCA_HELP_STR}
                       ON
                       "USE_ALLOCA_SUPPORTED; NOT OPUS_VAR_ARRAYS; NOT OPUS_NONTHREADSAFE_PSEUDOSTACK"
                       OFF)
add_feature_info(OPUS_USE_ALLOCA OPUS_USE_ALLOCA ${OPUS_USE_ALLOCA_HELP_STR})

set(OPUS_NONTHREADSAFE_PSEUDOSTACK_HELP_STR "use a non threadsafe pseudostack when neither variable length arrays or alloca is supported.")
cmake_dependent_option(OPUS_NONTHREADSAFE_PSEUDOSTACK
                       ${OPUS_NONTHREADSAFE_PSEUDOSTACK_HELP_STR}
                       ON
                       "NOT OPUS_VAR_ARRAYS; NOT OPUS_USE_ALLOCA"
                       OFF)
add_feature_info(OPUS_NONTHREADSAFE_PSEUDOSTACK OPUS_NONTHREADSAFE_PSEUDOSTACK ${OPUS_NONTHREADSAFE_PSEUDOSTACK_HELP_STR})

set(OPUS_FAST_MATH_HELP_STR "enable fast math (unsupported and discouraged use, as code is not well tested with this build option).")
cmake_dependent_option(OPUS_FAST_MATH
                       ${OPUS_FAST_MATH_HELP_STR}
                       ON
                       "OPUS_FLOAT_APPROX; OPUS_FAST_MATH; FAST_MATH_SUPPORTED"
                       OFF)
add_feature_info(OPUS_FAST_MATH OPUS_FAST_MATH ${OPUS_FAST_MATH_HELP_STR})

set(OPUS_STACK_PROTECTOR_HELP_STR "use stack protection.")
cmake_dependent_option(OPUS_STACK_PROTECTOR
                       ${OPUS_STACK_PROTECTOR_HELP_STR}
                       ON
                       "STACK_PROTECTOR_SUPPORTED"
                       OFF)
add_feature_info(OPUS_STACK_PROTECTOR OPUS_STACK_PROTECTOR ${OPUS_STACK_PROTECTOR_HELP_STR})

if(NOT MSVC)
  set(OPUS_FORTIFY_SOURCE_HELP_STR "add protection against buffer overflows.")
  cmake_dependent_option(OPUS_FORTIFY_SOURCE
                         ${OPUS_FORTIFY_SOURCE_HELP_STR}
                         ON
                         "FORTIFY_SOURCE_SUPPORTED"
                         OFF)
  add_feature_info(OPUS_FORTIFY_SOURCE OPUS_FORTIFY_SOURCE ${OPUS_FORTIFY_SOURCE_HELP_STR})
endif()

if(MINGW AND (OPUS_FORTIFY_SOURCE OR OPUS_STACK_PROTECTOR))
  # ssp lib is needed for security features for MINGW
  list(APPEND OPUS_REQUIRED_LIBRARIES ssp)
endif()

if(OPUS_CPU_X86 OR OPUS_CPU_X64)
  set(OPUS_X86_MAY_HAVE_SSE_HELP_STR "does runtime check for SSE1 support.")
  cmake_dependent_option(OPUS_X86_MAY_HAVE_SSE
                         ${OPUS_X86_MAY_HAVE_SSE_HELP_STR}
                         ON
                         "SSE1_SUPPORTED; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_MAY_HAVE_SSE OPUS_X86_MAY_HAVE_SSE ${OPUS_X86_MAY_HAVE_SSE_HELP_STR})

  set(OPUS_X86_MAY_HAVE_SSE2_HELP_STR "does runtime check for SSE2 support.")
  cmake_dependent_option(OPUS_X86_MAY_HAVE_SSE2
                         ${OPUS_X86_MAY_HAVE_SSE2_HELP_STR}
                         ON
                         "SSE2_SUPPORTED; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_MAY_HAVE_SSE2 OPUS_X86_MAY_HAVE_SSE2 ${OPUS_X86_MAY_HAVE_SSE2_HELP_STR})

  set(OPUS_X86_MAY_HAVE_SSE4_1_HELP_STR "does runtime check for SSE4.1 support.")
  cmake_dependent_option(OPUS_X86_MAY_HAVE_SSE4_1
                         ${OPUS_X86_MAY_HAVE_SSE4_1_HELP_STR}
                         ON
                         "SSE4_1_SUPPORTED; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_MAY_HAVE_SSE4_1 OPUS_X86_MAY_HAVE_SSE4_1 ${OPUS_X86_MAY_HAVE_SSE4_1_HELP_STR})

  set(OPUS_X86_MAY_HAVE_AVX2_HELP_STR "does runtime check for AVX FMA AVX2 support.")
  cmake_dependent_option(OPUS_X86_MAY_HAVE_AVX2
                         ${OPUS_X86_MAY_HAVE_AVX2_HELP_STR}
                         ON
                         "AVX2_SUPPORTED; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_MAY_HAVE_AVX2 OPUS_X86_MAY_HAVE_AVX2 ${OPUS_X86_MAY_HAVE_AVX2_HELP_STR})

  # PRESUME depends on MAY HAVE, but PRESUME will override runtime detection
  set(OPUS_X86_PRESUME_SSE_HELP_STR "assume target CPU has SSE1 support (override runtime check).")
  set(OPUS_X86_PRESUME_SSE2_HELP_STR "assume target CPU has SSE2 support (override runtime check).")
  if(OPUS_CPU_X64) # Assume x86_64 has up to SSE2 support
    cmake_dependent_option(OPUS_X86_PRESUME_SSE
                           ${OPUS_X86_PRESUME_SSE_HELP_STR}
                           ON
                           "OPUS_X86_MAY_HAVE_SSE; NOT OPUS_DISABLE_INTRINSICS"
                           OFF)

    cmake_dependent_option(OPUS_X86_PRESUME_SSE2
                           ${OPUS_X86_PRESUME_SSE2_HELP_STR}
                           ON
                           "OPUS_X86_MAY_HAVE_SSE2; NOT OPUS_DISABLE_INTRINSICS"
                           OFF)
  else()
    cmake_dependent_option(OPUS_X86_PRESUME_SSE
                           ${OPUS_X86_PRESUME_SSE_HELP_STR}
                           OFF
                           "OPUS_X86_MAY_HAVE_SSE; NOT OPUS_DISABLE_INTRINSICS"
                           OFF)

    cmake_dependent_option(OPUS_X86_PRESUME_SSE2
                           ${OPUS_X86_PRESUME_SSE2_HELP_STR}
                           OFF
                           "OPUS_X86_MAY_HAVE_SSE2; NOT OPUS_DISABLE_INTRINSICS"
                           OFF)
  endif()
  add_feature_info(OPUS_X86_PRESUME_SSE OPUS_X86_PRESUME_SSE ${OPUS_X86_PRESUME_SSE_HELP_STR})
  add_feature_info(OPUS_X86_PRESUME_SSE2 OPUS_X86_PRESUME_SSE2 ${OPUS_X86_PRESUME_SSE2_HELP_STR})

  set(OPUS_X86_PRESUME_SSE4_1_HELP_STR "assume target CPU has SSE4.1 support (override runtime check).")
  cmake_dependent_option(OPUS_X86_PRESUME_SSE4_1
                         ${OPUS_X86_PRESUME_SSE4_1_HELP_STR}
                         OFF
                         "OPUS_X86_MAY_HAVE_SSE4_1; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_PRESUME_SSE4_1 OPUS_X86_PRESUME_SSE4_1 ${OPUS_X86_PRESUME_SSE4_1_HELP_STR})

  set(OPUS_X86_PRESUME_AVX2_HELP_STR "assume target CPU has AVX FMA AVX2 support (override runtime check).")
  cmake_dependent_option(OPUS_X86_PRESUME_AVX2
                         ${OPUS_X86_PRESUME_AVX2_HELP_STR}
                         OFF
                         "OPUS_X86_MAY_HAVE_AVX2; NOT OPUS_DISABLE_INTRINSICS"
                         OFF)
  add_feature_info(OPUS_X86_PRESUME_AVX2 OPUS_X86_PRESUME_AVX2 ${OPUS_X86_PRESUME_AVX2_HELP_STR})
endif()

feature_summary(WHAT ALL)

set_package_properties(Git
                       PROPERTIES
                       TYPE
                       REQUIRED
                       DESCRIPTION
                       "fast, scalable, distributed revision control system"
                       URL
                       "https://git-scm.com/"
                       PURPOSE
                       "required to set up package version")

set(Opus_PUBLIC_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/opus.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_multistream.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_projection.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_types.h)

if(OPUS_CUSTOM_MODES)
  list(APPEND Opus_PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_custom.h)
endif()

if(MSVC)
  if(OPUS_STATIC_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  endif()
endif()

add_library(opus ${opus_headers} ${opus_sources} ${opus_sources_float} ${Opus_PUBLIC_HEADER})
add_library(Opus::opus ALIAS opus)

get_library_version(OPUS_LIBRARY_VERSION OPUS_LIBRARY_VERSION_MAJOR)
message(DEBUG "Opus library version: ${OPUS_LIBRARY_VERSION}")

set_target_properties(opus
                      PROPERTIES SOVERSION
                                 ${OPUS_LIBRARY_VERSION_MAJOR}
                                 VERSION
                                 ${OPUS_LIBRARY_VERSION}
                                 PUBLIC_HEADER
                                 "${Opus_PUBLIC_HEADER}")

target_include_directories(
  opus
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/opus>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/dnn
          celt
          silk)

target_link_libraries(opus PRIVATE ${OPUS_REQUIRED_LIBRARIES})
target_compile_definitions(opus PRIVATE OPUS_BUILD)

if(OPUS_FIXED_POINT_DEBUG)
  target_compile_definitions(opus PRIVATE FIXED_DEBUG)
endif()

if(OPUS_FORTIFY_SOURCE AND NOT MSVC)
  target_compile_definitions(opus PRIVATE
                             $<$<NOT:$<CONFIG:debug>>:_FORTIFY_SOURCE=2>)
endif()

if(OPUS_FLOAT_APPROX)
  target_compile_definitions(opus PRIVATE FLOAT_APPROX)
endif()

if(OPUS_ASSERTIONS)
  target_compile_definitions(opus PRIVATE ENABLE_ASSERTIONS)
endif()

if(OPUS_HARDENING)
  target_compile_definitions(opus PRIVATE ENABLE_HARDENING)
endif()

if(OPUS_FUZZING)
  target_compile_definitions(opus PRIVATE FUZZING)
endif()

if(OPUS_CHECK_ASM)
  target_compile_definitions(opus PRIVATE OPUS_CHECK_ASM)
endif()

if(NOT OPUS_DNN_FLOAT_DEBUG)
  target_compile_definitions(opus PRIVATE DISABLE_DEBUG_FLOAT)
endif()

if(OPUS_VAR_ARRAYS)
  target_compile_definitions(opus PRIVATE VAR_ARRAYS)
elseif(OPUS_USE_ALLOCA)
  target_compile_definitions(opus PRIVATE USE_ALLOCA)
elseif(OPUS_NONTHREADSAFE_PSEUDOSTACK)
  target_compile_definitions(opus PRIVATE NONTHREADSAFE_PSEUDOSTACK)
else()
  message(ERROR "Need to set a define for stack allocation")
endif()

if(OPUS_CUSTOM_MODES)
  target_compile_definitions(opus PRIVATE CUSTOM_MODES)
endif()

if(OPUS_FAST_MATH)
  if(MSVC)
    target_compile_options(opus PRIVATE /fp:fast)
  else()
    target_compile_options(opus PRIVATE -ffast-math)
  endif()
endif()

if(OPUS_STACK_PROTECTOR)
  if(MSVC)
    target_compile_options(opus PRIVATE /GS)
  else()
    target_compile_options(opus PRIVATE -fstack-protector-strong)
  endif()
elseif(STACK_PROTECTOR_DISABLED_SUPPORTED)
  target_compile_options(opus PRIVATE /GS-)
endif()

if(BUILD_SHARED_LIBS)
  if(WIN32)
    target_compile_definitions(opus PRIVATE DLL_EXPORT)
  elseif(HIDDEN_VISIBILITY_SUPPORTED)
    set_target_properties(opus PROPERTIES C_VISIBILITY_PRESET hidden)
  endif()
endif()

add_sources_group(opus silk ${silk_headers} ${silk_sources})
add_sources_group(opus celt ${celt_headers} ${celt_sources})

if(OPUS_FIXED_POINT)
  add_sources_group(opus silk ${silk_sources_fixed})
  target_include_directories(opus PRIVATE silk/fixed)
  target_compile_definitions(opus PRIVATE FIXED_POINT=1)
else()
  add_sources_group(opus silk ${silk_sources_float})
  target_include_directories(opus PRIVATE silk/float)
endif()

if(NOT OPUS_ENABLE_FLOAT_API)
  target_compile_definitions(opus PRIVATE DISABLE_FLOAT_API)
endif()

if (OPUS_DEEP_PLC OR OPUS_DRED OR OPUS_OSCE)
  add_sources_group(opus lpcnet ${deep_plc_headers} ${deep_plc_sources})
  set(OPUS_DNN TRUE)
else()
  set(OPUS_DNN FALSE)
endif()

if (OPUS_DNN)
  add_sources_group(opus lpcnet ${deep_plc_headers} ${deep_plc_sources})
  target_compile_definitions(opus PRIVATE ENABLE_DEEP_PLC)
endif()

if (OPUS_DRED)
  add_sources_group(opus lpcnet ${dred_headers} ${dred_sources})
  target_compile_definitions(opus PRIVATE ENABLE_DRED)
endif()

if (OPUS_OSCE)
  add_sources_group(opus lpcnet ${osce_headers} ${osce_sources})
  target_compile_definitions(opus PRIVATE ENABLE_OSCE)
endif()

if(NOT OPUS_DISABLE_INTRINSICS)
  if(((OPUS_X86_MAY_HAVE_SSE AND NOT OPUS_X86_PRESUME_SSE) OR
     (OPUS_X86_MAY_HAVE_SSE2 AND NOT OPUS_X86_PRESUME_SSE2) OR
     (OPUS_X86_MAY_HAVE_SSE4_1 AND NOT OPUS_X86_PRESUME_SSE4_1) OR
     (OPUS_X86_MAY_HAVE_AVX2 AND NOT OPUS_X86_PRESUME_AVX2)) AND
      RUNTIME_CPU_CAPABILITY_DETECTION)
    target_compile_definitions(opus PRIVATE OPUS_HAVE_RTCD)
    if(NOT MSVC)
      if(CPU_INFO_BY_ASM_SUPPORTED)
        target_compile_definitions(opus PRIVATE CPU_INFO_BY_ASM)
      elseif(CPU_INFO_BY_C_SUPPORTED)
        target_compile_definitions(opus PRIVATE CPU_INFO_BY_C)
      else()
        message(ERROR "Runtime cpu capability detection is enabled while CPU_INFO is not supported")
      endif()
    endif()
    add_sources_group(opus celt ${celt_sources_x86_rtcd})
    add_sources_group(opus silk ${silk_sources_x86_rtcd})
    if (OPUS_DNN)
      add_sources_group(opus lpcnet ${dnn_sources_x86_rtcd})
    endif()
  endif()

  if(SSE1_SUPPORTED)
    if(OPUS_X86_MAY_HAVE_SSE)
      add_sources_group(opus celt ${celt_sources_sse})
      target_compile_definitions(opus PRIVATE OPUS_X86_MAY_HAVE_SSE)
      if(NOT MSVC)
        set_source_files_properties(${celt_sources_sse} PROPERTIES COMPILE_FLAGS -msse)
      endif()
    endif()
    if(OPUS_X86_PRESUME_SSE)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_SSE)
      if(NOT MSVC)
        target_compile_options(opus PRIVATE -msse)
      endif()
    endif()
  endif()

  if(SSE2_SUPPORTED)
    if(OPUS_X86_MAY_HAVE_SSE2)
      add_sources_group(opus celt ${celt_sources_sse2})
      if (OPUS_DNN)
        add_sources_group(opus lpcnet ${dnn_sources_sse2})
      endif()
      target_compile_definitions(opus PRIVATE OPUS_X86_MAY_HAVE_SSE2)
      if(NOT MSVC)
        set_source_files_properties(${celt_sources_sse2} ${dnn_sources_sse2} PROPERTIES COMPILE_FLAGS -msse2)
      endif()
    endif()
    if(OPUS_X86_PRESUME_SSE2)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_SSE2)
      if(NOT MSVC)
        target_compile_options(opus PRIVATE -msse2)
      endif()
    endif()
  endif()

  if(SSE4_1_SUPPORTED)
    if(OPUS_X86_MAY_HAVE_SSE4_1)
      add_sources_group(opus celt ${celt_sources_sse4_1})
      add_sources_group(opus silk ${silk_sources_sse4_1})
      if (OPUS_DNN)
        add_sources_group(opus lpcnet ${dnn_sources_sse4_1})
      endif()
      target_compile_definitions(opus PRIVATE OPUS_X86_MAY_HAVE_SSE4_1)
      if(NOT MSVC OR MSVC_CLANG_CL)
        set_source_files_properties(${celt_sources_sse4_1} ${silk_sources_sse4_1} ${dnn_sources_sse4_1} PROPERTIES COMPILE_FLAGS -msse4.1)
      endif()

      if(OPUS_FIXED_POINT)
        add_sources_group(opus silk ${silk_sources_fixed_sse4_1})
        if(NOT MSVC OR MSVC_CLANG_CL)
          set_source_files_properties(${silk_sources_fixed_sse4_1} PROPERTIES COMPILE_FLAGS -msse4.1)
        endif()
      endif()
    endif()
    if(OPUS_X86_PRESUME_SSE4_1)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_SSE4_1)
      if(NOT MSVC OR MSVC_CLANG_CL)
        target_compile_options(opus PRIVATE -msse4.1)
      endif()
    endif()
  endif()

  if(AVX2_SUPPORTED)
    if(OPUS_X86_MAY_HAVE_AVX2)
      add_sources_group(opus celt ${celt_sources_avx2})
      add_sources_group(opus silk ${silk_sources_avx2})
      if (NOT OPUS_FIXED_POINT)
        add_sources_group(opus silk ${silk_sources_float_avx2})
      endif()
      if (OPUS_DNN)
        add_sources_group(opus lpcnet ${dnn_sources_avx2})
      endif()
      target_compile_definitions(opus PRIVATE OPUS_X86_MAY_HAVE_AVX2)
      if(MSVC)
        set(AVX2_FLAGS "${AVX2_FLAGS} /arch:AVX2")
      else()
        set(AVX2_FLAGS "${AVX2_FLAGS} -mavx2 -mfma -mavx")
      endif()
      set_source_files_properties(${celt_sources_avx2} PROPERTIES COMPILE_FLAGS ${AVX2_FLAGS})
      set_source_files_properties(${silk_sources_avx2} PROPERTIES COMPILE_FLAGS ${AVX2_FLAGS})
      if (NOT OPUS_FIXED_POINT)
        set_source_files_properties(${silk_sources_float_avx2} PROPERTIES COMPILE_FLAGS ${AVX2_FLAGS})
      endif()
      set_source_files_properties(${dnn_sources_avx2} PROPERTIES COMPILE_FLAGS ${AVX2_FLAGS})
    endif()
    if(OPUS_X86_PRESUME_AVX2)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_AVX2)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_SSE4_1)
      if(NOT MSVC)
        target_compile_options(opus PRIVATE -mavx2 -mfma -mavx)
      endif()
    endif()
  endif()

  if(MSVC)
    if(AVX2_SUPPORTED AND OPUS_X86_PRESUME_AVX2) # on 64 bit and 32 bits
      add_definitions(/arch:AVX2)
    elseif(OPUS_CPU_X86) # if AVX not supported then set SSE flag
      if((SSE4_1_SUPPORTED AND OPUS_X86_PRESUME_SSE4_1)
         OR (SSE2_SUPPORTED AND OPUS_X86_PRESUME_SSE2))
        target_compile_definitions(opus PRIVATE /arch:SSE2)
      elseif(SSE1_SUPPORTED AND OPUS_X86_PRESUME_SSE)
        target_compile_definitions(opus PRIVATE /arch:SSE)
      endif()
    endif()
  endif()

  if(COMPILER_SUPPORT_NEON)
    if(OPUS_MAY_HAVE_NEON)
      if(RUNTIME_CPU_CAPABILITY_DETECTION)
        message(STATUS "OPUS_MAY_HAVE_NEON enabling runtime detection")
        target_compile_definitions(opus PRIVATE OPUS_HAVE_RTCD)
        add_sources_group(opus celt ${celt_sources_arm_rtcd})
        add_sources_group(opus silk ${silk_sources_arm_rtcd})
        add_sources_group(opus lpcnet ${dnn_sources_arm_rtcd})
      else()
        message(ERROR "Runtime cpu capability detection needed for MAY_HAVE_NEON")
      endif()
      # Do runtime check for NEON
      target_compile_definitions(opus
                                 PRIVATE
                                 OPUS_ARM_MAY_HAVE_NEON
                                 OPUS_ARM_MAY_HAVE_NEON_INTR)
    endif()

    add_sources_group(opus celt ${celt_sources_arm_neon_intr})
    add_sources_group(opus silk ${silk_sources_arm_neon_intr})
    if (OPUS_DNN)
      add_sources_group(opus lpcnet ${dnn_sources_arm_neon})
    endif()

    # silk arm neon depends on main_Fix.h
    target_include_directories(opus PRIVATE silk/fixed)

    if(OPUS_FIXED_POINT)
      add_sources_group(opus silk ${silk_sources_fixed_arm_neon_intr})
    endif()

    if(OPUS_PRESUME_NEON)
      target_compile_definitions(opus
                                 PRIVATE
                                 OPUS_ARM_PRESUME_NEON
                                 OPUS_ARM_PRESUME_NEON_INTR)
    endif()
  endif()
endif()

target_compile_definitions(opus
                           PRIVATE
                           $<$<BOOL:${HAVE_LRINT}>:HAVE_LRINT>
                           $<$<BOOL:${HAVE_LRINTF}>:HAVE_LRINTF>
                           $<$<BOOL:${HAVE_ELF_AUX_INFO}>:HAVE_ELF_AUX_INFO>)

if(OPUS_BUILD_FRAMEWORK)
  set_target_properties(opus PROPERTIES
                        FRAMEWORK TRUE
                        FRAMEWORK_VERSION ${PROJECT_VERSION}
                        MACOSX_FRAMEWORK_IDENTIFIER org.xiph.opus
                        MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${PROJECT_VERSION}
                        MACOSX_FRAMEWORK_BUNDLE_VERSION ${PROJECT_VERSION}
                        XCODE_ATTRIBUTE_INSTALL_PATH "@rpath"
                        OUTPUT_NAME Opus)
endif()

install(TARGETS opus
        EXPORT OpusTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opus)

if(OPUS_INSTALL_PKG_CONFIG_MODULE)
  set(prefix ${CMAKE_INSTALL_PREFIX})
  set(exec_prefix ${CMAKE_INSTALL_PREFIX})
  set(libdir ${CMAKE_INSTALL_FULL_LIBDIR})
  set(includedir ${CMAKE_INSTALL_FULL_INCLUDEDIR})
  set(VERSION ${PACKAGE_VERSION})
  if(HAVE_LIBM)
    set(LIBM "-lm")
  endif()
  configure_file(opus.pc.in opus.pc)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/opus.pc
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

if(OPUS_INSTALL_CMAKE_CONFIG_MODULE)
  set(CPACK_GENERATOR TGZ)
  include(CPack)
  set(CMAKE_INSTALL_PACKAGEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
  install(EXPORT OpusTargets
          NAMESPACE Opus::
          DESTINATION ${CMAKE_INSTALL_PACKAGEDIR})

  include(CMakePackageConfigHelpers)

  set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/OpusConfig.cmake.in
                                OpusConfig.cmake
                                INSTALL_DESTINATION
                                ${CMAKE_INSTALL_PACKAGEDIR}
                                PATH_VARS
                                INCLUDE_INSTALL_DIR
                                INSTALL_PREFIX
                                ${CMAKE_INSTALL_PREFIX})
  write_basic_package_version_file(OpusConfigVersion.cmake
                                   VERSION ${PROJECT_VERSION}
                                   COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OpusConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/OpusConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_PACKAGEDIR})
endif()

if(OPUS_BUILD_PROGRAMS)
  # demo
  if(OPUS_CUSTOM_MODES)
    add_executable(opus_custom_demo ${opus_custom_demo_sources})
    target_include_directories(opus_custom_demo
                               PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(opus_custom_demo PRIVATE opus)
    target_compile_definitions(opus_custom_demo PRIVATE OPUS_BUILD)
  endif()

  add_executable(opus_demo ${opus_demo_sources})
  target_include_directories(opus_demo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_include_directories(opus_demo PRIVATE silk) # debug.h
  target_include_directories(opus_demo PRIVATE celt) # arch.h
  target_include_directories(opus_demo PRIVATE dnn)
  target_link_libraries(opus_demo PRIVATE opus ${OPUS_REQUIRED_LIBRARIES})
  target_compile_definitions(opus_demo PRIVATE OPUS_BUILD)

  # compare
  add_executable(opus_compare ${opus_compare_sources})
  target_include_directories(opus_compare PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(opus_compare PRIVATE opus ${OPUS_REQUIRED_LIBRARIES})
endif()

if(BUILD_TESTING AND NOT BUILD_SHARED_LIBS)
  enable_testing()

  # tests
  add_executable(test_opus_decode ${test_opus_decode_sources})
  target_include_directories(test_opus_decode
                             PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(test_opus_decode PRIVATE opus)
  target_compile_definitions(test_opus_decode PRIVATE OPUS_BUILD)
  if(OPUS_FIXED_POINT)
    target_compile_definitions(test_opus_decode PRIVATE DISABLE_FLOAT_API)
  endif()
  add_test(NAME test_opus_decode COMMAND ${CMAKE_COMMAND}
           -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_decode>
           -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
           -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")

  add_executable(test_opus_padding ${test_opus_padding_sources})
  target_include_directories(test_opus_padding
                             PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(test_opus_padding PRIVATE opus)
  add_test(NAME test_opus_padding COMMAND ${CMAKE_COMMAND}
           -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_padding>
           -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
           -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")

  add_executable(test_opus_api ${test_opus_api_sources})
  target_include_directories(test_opus_api
                            PRIVATE ${CMAKE_CURRENT_BINARY_DIR} celt)
  target_link_libraries(test_opus_api PRIVATE opus)
  target_compile_definitions(test_opus_api PRIVATE OPUS_BUILD)
  if(OPUS_FIXED_POINT)
    target_compile_definitions(test_opus_api PRIVATE DISABLE_FLOAT_API)
  endif()
  add_test(NAME test_opus_api COMMAND ${CMAKE_COMMAND}
        -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_api>
        -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
        -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")

  add_executable(test_opus_encode ${test_opus_encode_sources})
  target_include_directories(test_opus_encode
                            PRIVATE ${CMAKE_CURRENT_BINARY_DIR} celt dnn)
  target_link_libraries(test_opus_encode PRIVATE opus)
  target_compile_definitions(test_opus_encode PRIVATE OPUS_BUILD)
  add_test(NAME test_opus_encode COMMAND ${CMAKE_COMMAND}
        -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_encode>
        -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
        -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")

  add_executable(test_opus_extensions ${test_opus_extensions_sources})
  target_include_directories(test_opus_extensions
                            PRIVATE ${CMAKE_CURRENT_BINARY_DIR} celt dnn)
  target_link_libraries(test_opus_extensions PRIVATE opus)
  target_compile_definitions(test_opus_extensions PRIVATE OPUS_BUILD)
  add_test(NAME test_opus_extensions COMMAND ${CMAKE_COMMAND}
        -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_extensions>
        -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
        -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")
  if(OPUS_DRED)
    add_executable(test_opus_dred ${test_opus_dred_sources})
    target_include_directories(test_opus_dred
                              PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(test_opus_dred PRIVATE opus)
    target_compile_definitions(test_opus_dred PRIVATE OPUS_BUILD)
    add_test(NAME test_opus_dred COMMAND ${CMAKE_COMMAND}
          -DTEST_EXECUTABLE=$<TARGET_FILE:test_opus_dred>
          -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
          -P "${PROJECT_SOURCE_DIR}/cmake/RunTest.cmake")
  endif()
  if(OPUS_CUSTOM_MODES)
    add_executable(test_opus_custom ${test_opus_custom_sources})
    target_include_directories(test_opus_custom
                              PRIVATE ${CMAKE_CURRENT_BINARY_DIR} celt dnn)
    target_link_libraries(test_opus_custom PRIVATE opus)
    target_compile_definitions(test_opus_custom PRIVATE OPUS_BUILD)
  endif()
endif()
