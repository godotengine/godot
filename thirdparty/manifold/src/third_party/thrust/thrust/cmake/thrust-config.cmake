#
# find_package(Thrust) config file.
#
# Provided by NVIDIA under the same license as the associated Thrust library.
#
# Reply-To: Allison Vacanti <alliepiper16@gmail.com>
#
# *****************************************************************************
# **     The following is a short reference to using Thrust from CMake.      **
# ** For more details, see the README.md in the same directory as this file. **
# *****************************************************************************
#
# # General Usage:
# find_package(Thrust REQUIRED CONFIG)
# thrust_create_target(Thrust [options])
# target_link_libraries(some_project_lib Thrust)
#
# # Create default target with: HOST=CPP DEVICE=CUDA
# thrust_create_target(TargetName)
#
# # Create target with: HOST=CPP DEVICE=TBB
# thrust_create_target(TargetName DEVICE TBB)
#
# # Create target with: HOST=TBB DEVICE=OMP
# thrust_create_target(TargetName HOST TBB DEVICE OMP)
#
# # Create CMake cache options THRUST_[HOST|DEVICE]_SYSTEM and configure a
# # target from them. This allows these systems to be changed by developers at
# # configure time, per build.
# thrust_create_target(TargetName FROM_OPTIONS
#   [HOST_OPTION <option_name>]      # Optionally rename the host system option
#   [DEVICE_OPTION <option_name>]    # Optionally rename the device system option
#   [HOST_OPTION_DOC <doc_string>]   # Optionally change the cache label
#   [DEVICE_OPTION_DOC <doc_string>] # Optionally change the cache label
#   [HOST <default system>]          # Optionally change the default backend
#   [DEVICE <default system>]        # Optionally change the default backend
#   [ADVANCED]                       # Optionally mark options as advanced
# )
#
# # Use a custom TBB, CUB, and/or OMP
# # (Note that once set, these cannot be changed. This includes COMPONENT
# # preloading and lazy lookups in thrust_create_target)
# find_package(Thrust REQUIRED)
# thrust_set_CUB_target(MyCUBTarget)  # MyXXXTarget contains an existing
# thrust_set_TBB_target(MyTBBTarget)  # interface to XXX for Thrust to use.
# thrust_set_OMP_target(MyOMPTarget)
# thrust_create_target(ThrustWithMyCUB DEVICE CUDA)
# thrust_create_target(ThrustWithMyTBB DEVICE TBB)
# thrust_create_target(ThrustWithMyOMP DEVICE OMP)
#
# # Create target with HOST=CPP DEVICE=CUDA and some advanced flags set
# thrust_create_target(TargetName
#   IGNORE_DEPRECATED_API         # Silence build warnings about deprecated APIs
#   IGNORE_DEPRECATED_CPP_DIALECT # Silence build warnings about deprecated compilers and C++ standards
#   IGNORE_DEPRECATED_CPP_11      # Only silence deprecation warnings for C++11
#   IGNORE_DEPRECATED_COMPILER    # Only silence deprecation warnings for old compilers
#   IGNORE_CUB_VERSION            # Skip configure-time and compile-time CUB version checks
# )
#
# # Test if a particular system has been loaded. ${var_name} is set to TRUE or
# # FALSE to indicate if "system" is found.
# thrust_is_system_found(<system> <var_name>)
# thrust_is_cuda_system_found(<var_name>)
# thrust_is_tbb_system_found(<var_name>)
# thrust_is_omp_system_found(<var_name>)
# thrust_is_cpp_system_found(<var_name>)
#
# # Define / update THRUST_${system}_FOUND flags in current scope
# thrust_update_system_found_flags()
#
# # View verbose log with target and dependency information:
# $ cmake . --log-level=VERBOSE (CMake 3.15.7 and above)
#
# # Print debugging output to status channel:
# thrust_debug_internal_targets()
# thrust_debug_target(TargetName "${THRUST_VERSION}")

cmake_minimum_required(VERSION 3.15)

################################################################################
# User variables and APIs. Users can rely on these:
#

# Advertise system options:
set(THRUST_HOST_SYSTEM_OPTIONS
  CPP OMP TBB
  CACHE INTERNAL "Valid Thrust host systems."
)
set(THRUST_DEVICE_SYSTEM_OPTIONS
  CUDA CPP OMP TBB
  CACHE INTERNAL "Valid Thrust device systems"
)

# Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
set(THRUST_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "")
set(THRUST_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "")
set(THRUST_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "")
set(THRUST_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "")
set(THRUST_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "")
set(THRUST_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "")

function(thrust_create_target target_name)
  thrust_debug("Assembling target ${target_name}. Options: ${ARGN}" internal)
  set(options
    ADVANCED
    FROM_OPTIONS
    IGNORE_CUB_VERSION_CHECK
    IGNORE_DEPRECATED_API
    IGNORE_DEPRECATED_COMPILER
    IGNORE_DEPRECATED_CPP_11
    IGNORE_DEPRECATED_CPP_DIALECT
    )
  set(keys
    DEVICE
    DEVICE_OPTION
    DEVICE_OPTION_DOC
    HOST
    HOST_OPTION
    HOST_OPTION_DOC
    )
  cmake_parse_arguments(TCT "${options}" "${keys}" "" ${ARGN})
  if (TCT_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Unrecognized arguments passed to thrust_create_target: "
      ${TCT_UNPARSED_ARGUMENTS}
      )
  endif()

  # Check that the main Thrust internal target is available
  # (functions have global scope, targets have directory scope, so this
  # might happen)
  if (NOT TARGET Thrust::Thrust)
    message(AUTHOR_WARNING
      "The `thrust_create_target` function was called outside the scope of the "
      "thrust targets. Call find_package again to recreate targets."
      )
  endif()

  _thrust_set_if_undefined(TCT_HOST CPP)
  _thrust_set_if_undefined(TCT_DEVICE CUDA)
  _thrust_set_if_undefined(TCT_HOST_OPTION THRUST_HOST_SYSTEM)
  _thrust_set_if_undefined(TCT_DEVICE_OPTION THRUST_DEVICE_SYSTEM)
  _thrust_set_if_undefined(TCT_HOST_OPTION_DOC "Thrust host system.")
  _thrust_set_if_undefined(TCT_DEVICE_OPTION_DOC "Thrust device system.")

  if (NOT TCT_HOST IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
    message(FATAL_ERROR
      "Requested HOST=${TCT_HOST}; must be one of ${THRUST_HOST_SYSTEM_OPTIONS}")
  endif()

  if (NOT TCT_DEVICE IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
    message(FATAL_ERROR
      "Requested DEVICE=${TCT_DEVICE}; must be one of ${THRUST_DEVICE_SYSTEM_OPTIONS}")
  endif()

  if (TCT_FROM_OPTIONS)
    _thrust_create_cache_options(
      ${TCT_HOST} ${TCT_DEVICE}
      ${TCT_HOST_OPTION} ${TCT_DEVICE_OPTION}
      ${TCT_HOST_OPTION_DOC} ${TCT_DEVICE_OPTION_DOC}
      ${TCT_ADVANCED}
    )
    set(TCT_HOST ${${TCT_HOST_OPTION}})
    set(TCT_DEVICE ${${TCT_DEVICE_OPTION}})
    thrust_debug("Current option settings:" internal)
    thrust_debug("  - ${TCT_HOST_OPTION}=${TCT_HOST}" internal)
    thrust_debug("  - ${TCT_DEVICE_OPTION}=${TCT_DEVICE}" internal)
  endif()

  _thrust_find_backend(${TCT_HOST} REQUIRED)
  _thrust_find_backend(${TCT_DEVICE} REQUIRED)

  # We can just create an INTERFACE IMPORTED target here instead of going
  # through _thrust_declare_interface_alias as long as we aren't hanging any
  # Thrust/CUB include paths on ${target_name}.
  add_library(${target_name} INTERFACE IMPORTED)
  target_link_libraries(${target_name}
    INTERFACE
    Thrust::${TCT_HOST}::Host
    Thrust::${TCT_DEVICE}::Device
  )

  # This would be nice to enforce, but breaks when using old cmake + new
  # compiler, since cmake doesn't know what features the new compiler version
  # supports.
  # Leaving this here as a reminder not to add it back. Just let the
  # compile-time checks in thrust/detail/config/cpp_dialect.h handle it.
  #
  #  if (NOT TCT_IGNORE_DEPRECATED_CPP_DIALECT)
  #    if (TCT_IGNORE_DEPRECATED_CPP_11)
  #      target_compile_features(${target_name} INTERFACE cxx_std_11)
  #    else()
  #      target_compile_features(${target_name} INTERFACE cxx_std_14)
  #    endif()
  #  endif()

  if (TCT_IGNORE_DEPRECATED_CPP_DIALECT)
    target_compile_definitions(${target_name} INTERFACE "THRUST_IGNORE_DEPRECATED_CPP_DIALECT")
  endif()

  if (TCT_IGNORE_DEPRECATED_API)
    target_compile_definitions(${target_name} INTERFACE "THRUST_IGNORE_DEPRECATED_API")
  endif()

  if (TCT_IGNORE_DEPRECATED_CPP_11)
    target_compile_definitions(${target_name} INTERFACE "THRUST_IGNORE_DEPRECATED_CPP_11")
  endif()

  if (TCT_IGNORE_DEPRECATED_COMPILER)
    target_compile_definitions(${target_name} INTERFACE "THRUST_IGNORE_DEPRECATED_COMPILER")
  endif()

  if (TCT_IGNORE_CUB_VERSION_CHECK)
    target_compile_definitions(${target_name} INTERFACE "THRUST_IGNORE_CUB_VERSION_CHECK")
  else()
    if (("${TCT_HOST}" STREQUAL "CUDA" OR "${TCT_DEVICE}" STREQUAL "CUDA") AND
    (NOT THRUST_VERSION VERSION_EQUAL THRUST_CUB_VERSION))
      message(FATAL_ERROR
        "The version of CUB found by CMake is not compatible with this release of Thrust. "
        "CUB is now included in the CUDA Toolkit, so you no longer need to use your own checkout of CUB. "
        "Pass IGNORE_CUB_VERSION_CHECK to thrust_create_target to ignore. "
        "(CUB ${THRUST_CUB_VERSION}, Thrust ${THRUST_VERSION})."
        )
    endif()
  endif()

  thrust_debug_target(${target_name} "Thrust ${THRUST_VERSION}"  internal)
endfunction()

function(thrust_is_system_found system var_name)
  if (TARGET Thrust::${system})
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(thrust_is_cpp_system_found var_name)
  thrust_is_system_found(CPP ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(thrust_is_cuda_system_found var_name)
  thrust_is_system_found(CUDA ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(thrust_is_tbb_system_found var_name)
  thrust_is_system_found(TBB ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

function(thrust_is_omp_system_found var_name)
  thrust_is_system_found(OMP ${var_name})
  set(${var_name} ${${var_name}} PARENT_SCOPE)
endfunction()

# Since components are loaded lazily, this will refresh the
# THRUST_${component}_FOUND flags in the current scope.
# Alternatively, check system states individually using the
# thrust_is_system_found functions.
macro(thrust_update_system_found_flags)
  set(THRUST_FOUND TRUE)
  thrust_is_system_found(CPP  THRUST_CPP_FOUND)
  thrust_is_system_found(CUDA THRUST_CUDA_FOUND)
  thrust_is_system_found(TBB  THRUST_TBB_FOUND)
  thrust_is_system_found(OMP  THRUST_OMP_FOUND)
endmacro()

function(thrust_debug msg)
  # Use the VERBOSE channel when called internally
  # Run `cmake . --log-level=VERBOSE` to view.
  if ("${ARGN}" STREQUAL "internal")
    # If CMake is too old to know about the VERBOSE channel, just be silent.
    # Users reproduce much the same output on the STATUS channel by using:
    # thrust_create_target(Thrust [...])
    # thrust_debug_internal_targets()
    # thrust_debug_target(Thrust)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.15.7")
      set(channel VERBOSE)
    else()
      return()
    endif()
  else()
    set(channel STATUS)
  endif()

  message(${channel} "Thrust: ${msg}")
endfunction()

# Print details of the specified target.
function(thrust_debug_target target_name version)
  if (NOT TARGET ${target_name})
    return()
  endif()

  set(is_internal "${ARGN}")

  if (version)
    set(version "(${version})")
  endif()

  thrust_debug("TargetInfo: ${target_name}: ${version}" ${is_internal})

  function(_thrust_print_prop_if_set target_name prop)
    get_target_property(value ${target_name} ${prop})
    if (value)
      thrust_debug("TargetInfo: ${target_name} > ${prop}: ${value}" ${is_internal})
    endif()
  endfunction()

  function(_thrust_print_imported_prop_if_set target_name prop)
    get_target_property(imported ${target_name} IMPORTED)
    get_target_property(type ${target_name} TYPE)
    if (imported AND NOT ${type} STREQUAL "INTERFACE_LIBRARY")
      _thrust_print_prop_if_set(${target_name} ${prop})
    endif()
  endfunction()

  _thrust_print_prop_if_set(${target_name} ALIASED_TARGET)
  _thrust_print_prop_if_set(${target_name} IMPORTED)
  _thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_DEFINITIONS)
  _thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_FEATURES)
  _thrust_print_prop_if_set(${target_name} INTERFACE_COMPILE_OPTIONS)
  _thrust_print_prop_if_set(${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  _thrust_print_prop_if_set(${target_name} INTERFACE_LINK_DEPENDS)
  _thrust_print_prop_if_set(${target_name} INTERFACE_LINK_DIRECTORIES)
  _thrust_print_prop_if_set(${target_name} INTERFACE_LINK_LIBRARIES)
  _thrust_print_prop_if_set(${target_name} INTERFACE_LINK_OPTIONS)
  _thrust_print_prop_if_set(${target_name} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  _thrust_print_prop_if_set(${target_name} INTERFACE_THRUST_HOST)
  _thrust_print_prop_if_set(${target_name} INTERFACE_THRUST_DEVICE)
  _thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION)
  _thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION_DEBUG)
  _thrust_print_imported_prop_if_set(${target_name} IMPORTED_LOCATION_RELEASE)
endfunction()

function(thrust_debug_internal_targets)
  function(_thrust_debug_backend_targets backend version)
    thrust_debug_target(Thrust::${backend} "${version}")
    thrust_debug_target(Thrust::${backend}::Host "${version}")
    thrust_debug_target(Thrust::${backend}::Device "${version}")
  endfunction()

  thrust_debug_target(Thrust::Thrust "${THRUST_VERSION}")

  _thrust_debug_backend_targets(CPP "Thrust ${THRUST_VERSION}")

  _thrust_debug_backend_targets(CUDA "CUB ${THRUST_CUB_VERSION}")
  thrust_debug_target(CUB::CUB "${THRUST_CUB_VERSION}")

  _thrust_debug_backend_targets(TBB "${THRUST_TBB_VERSION}")
  thrust_debug_target(TBB:tbb "${THRUST_TBB_VERSION}")

  _thrust_debug_backend_targets(OMP "${THRUST_OMP_VERSION}")
  thrust_debug_target(OpenMP::OpenMP_CXX "${THRUST_OMP_VERSION}")
endfunction()

################################################################################
# Internal utilities. Subject to change.
#

function(_thrust_set_if_undefined var)
  if (NOT DEFINED ${var})
    set(${var} ${ARGN} PARENT_SCOPE)
  endif()
endfunction()

function(_thrust_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) nvcc will automatically check the CUDA Toolkit include path *before* the
  #    system includes. This means that the Toolkit Thrust will *always* be used
  #    during compilation, and the include paths of an IMPORTED Thrust::Thrust
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to Thrust::Thrust. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong thrust!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)
  add_library(${alias_name} ALIAS ${ugly_name})
endfunction()

# Create cache options for selecting the user/device systems with ccmake/cmake-gui.
function(_thrust_create_cache_options host device host_option device_option host_doc device_doc advanced)
  thrust_debug("Creating system cache options: (advanced=${advanced})" internal)
  thrust_debug("  - Host Option=${host_option} Default=${host} Doc='${host_doc}'" internal)
  thrust_debug("  - Device Option=${device_option} Default=${device} Doc='${device_doc}'" internal)
  set(${host_option} ${host} CACHE STRING "${host_doc}")
  set_property(CACHE ${host_option} PROPERTY STRINGS ${THRUST_HOST_SYSTEM_OPTIONS})
  set(${device_option} ${device} CACHE STRING "${device_doc}")
  set_property(CACHE ${device_option} PROPERTY STRINGS ${THRUST_DEVICE_SYSTEM_OPTIONS})
  if (advanced)
    mark_as_advanced(${host_option} ${device_option})
  endif()
endfunction()

# Create Thrust::${backend}::Host and Thrust::${backend}::Device targets.
# Assumes that `Thrust::${backend}` and `_Thrust_${backend}` have been created
# by _thrust_declare_interface_alias and configured to bring in system
# dependency interfaces (including Thrust::Thrust).
function(_thrust_setup_system backend)
  set(backend_target_alias "Thrust::${backend}")

  if (backend IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
    set(host_target "_Thrust_${backend}_Host")
    set(host_target_alias "Thrust::${backend}::Host")
    if (NOT TARGET ${host_target_alias})
      _thrust_declare_interface_alias(${host_target_alias} ${host_target})
      target_compile_definitions(${host_target} INTERFACE
        "THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${backend}")
      target_link_libraries(${host_target} INTERFACE ${backend_target_alias})
      set_property(TARGET ${host_target} PROPERTY INTERFACE_THRUST_HOST ${backend})
      set_property(TARGET ${host_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING THRUST_HOST)
      thrust_debug_target(${host_target_alias} "" internal)
    endif()
  endif()

  if (backend IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
    set(device_target "_Thrust_${backend}_Device")
    set(device_target_alias "Thrust::${backend}::Device")
    if (NOT TARGET ${device_target_alias})
      _thrust_declare_interface_alias(${device_target_alias} ${device_target})
      target_compile_definitions(${device_target} INTERFACE
        "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${backend}")
      target_link_libraries(${device_target} INTERFACE ${backend_target_alias})
      set_property(TARGET ${device_target} PROPERTY INTERFACE_THRUST_DEVICE ${backend})
      set_property(TARGET ${device_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING THRUST_DEVICE)
      thrust_debug_target(${device_target_alias} "" internal)
    endif()
  endif()
endfunction()

# Use the provided cub_target for the CUDA backend. If Thrust::CUDA already
# exists, this call has no effect.
function(thrust_set_CUB_target cub_target)
  if (NOT TARGET Thrust::CUDA)
    thrust_debug("Setting CUB target to ${cub_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(THRUST_CUB_VERSION ${CUB_VERSION} CACHE INTERNAL "CUB version used by Thrust")
    _thrust_declare_interface_alias(Thrust::CUDA _Thrust_CUDA)
    target_link_libraries(_Thrust_CUDA INTERFACE Thrust::Thrust ${cub_target})
    thrust_debug_target(${cub_target} "${THRUST_CUB_VERSION}" internal)
    thrust_debug_target(Thrust::CUDA "CUB ${THRUST_CUB_VERSION}" internal)
    _thrust_setup_system(CUDA)
  endif()
endfunction()

# Use the provided tbb_target for the TBB backend. If Thrust::TBB already
# exists, this call has no effect.
function(thrust_set_TBB_target tbb_target)
  if (NOT TARGET Thrust::TBB)
    thrust_debug("Setting TBB target to ${tbb_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(THRUST_TBB_VERSION ${TBB_VERSION} CACHE INTERNAL "TBB version used by Thrust")
    _thrust_declare_interface_alias(Thrust::TBB _Thrust_TBB)
    target_link_libraries(_Thrust_TBB INTERFACE Thrust::Thrust ${tbb_target})
    thrust_debug_target(${tbb_target} "${THRUST_TBB_VERSION}" internal)
    thrust_debug_target(Thrust::TBB "${THRUST_TBB_VERSION}" internal)
    _thrust_setup_system(TBB)
  endif()
endfunction()

# Use the provided omp_target for the OMP backend. If Thrust::OMP already
# exists, this call has no effect.
function(thrust_set_OMP_target omp_target)
  if (NOT TARGET Thrust::OMP)
    thrust_debug("Setting OMP target to ${omp_target}" internal)
    # Workaround cmake issue #20670 https://gitlab.kitware.com/cmake/cmake/-/issues/20670
    set(THRUST_OMP_VERSION ${OpenMP_CXX_VERSION} CACHE INTERNAL "OpenMP version used by Thrust")
    _thrust_declare_interface_alias(Thrust::OMP _Thrust_OMP)
    target_link_libraries(_Thrust_OMP INTERFACE Thrust::Thrust ${omp_target})
    thrust_debug_target(${omp_target} "${THRUST_OMP_VERSION}" internal)
    thrust_debug_target(Thrust::OMP "${THRUST_OMP_VERSION}" internal)
    _thrust_setup_system(OMP)
  endif()
endfunction()

function(_thrust_find_CPP required)
  if (NOT TARGET Thrust::CPP)
    thrust_debug("Generating CPP targets." internal)
    _thrust_declare_interface_alias(Thrust::CPP _Thrust_CPP)
    target_link_libraries(_Thrust_CPP INTERFACE Thrust::Thrust)
    thrust_debug_target(Thrust::CPP "Thrust ${THRUST_VERSION}" internal)
    _thrust_setup_system(CPP)
  endif()
endfunction()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like CUB_VERSION, etc won't be in the caller's
# scope.
macro(_thrust_find_CUDA required)
  if (NOT TARGET Thrust::CUDA)
    thrust_debug("Searching for CUB ${required}" internal)
    find_package(CUB ${THRUST_VERSION} CONFIG
      ${_THRUST_QUIET_FLAG}
      ${required}
      NO_DEFAULT_PATH # Only check the explicit HINTS below:
      HINTS
        "${_THRUST_INCLUDE_DIR}/dependencies/cub" # Source layout (GitHub)
        "${_THRUST_INCLUDE_DIR}/../cub/cub/cmake" # Source layout (Perforce)
        "${_THRUST_CMAKE_DIR}/.."                 # Install layout
    )

    if (TARGET CUB::CUB)
      thrust_set_CUB_target(CUB::CUB)
    else()
      thrust_debug("CUB not found!" internal)
    endif()
  endif()
endmacro()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like TBB_VERSION, etc won't be in the caller's
# scope.
macro(_thrust_find_TBB required)
  if(NOT TARGET Thrust::TBB)
    thrust_debug("Searching for TBB ${required}" internal)
    # Swap in a temporary module path to make sure we use our FindTBB.cmake
    set(_THRUST_STASH_MODULE_PATH "${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "${_THRUST_CMAKE_DIR}")

    # Push policy CMP0074 to silence warnings about TBB_ROOT being set. This
    # var is used unconventionally in this FindTBB.cmake module.
    # Someday we'll have a suitable TBB cmake configuration and can avoid this.
    cmake_policy(PUSH)
    cmake_policy(SET CMP0074 OLD)
    set(THRUST_TBB_ROOT "" CACHE PATH "Path to the root of the TBB installation.")
    if (TBB_ROOT AND NOT THRUST_TBB_ROOT)
      message(
        "Warning: TBB_ROOT is set. "
        "Thrust uses THRUST_TBB_ROOT to avoid issues with CMake Policy CMP0074. "
        "Please set this variable instead when using Thrust with TBB."
      )
    endif()
    set(TBB_ROOT "${THRUST_TBB_ROOT}")
    set(_THRUST_STASH_TBB_ROOT "${TBB_ROOT}")

    find_package(TBB
      ${_THRUST_QUIET_FLAG}
      ${required}
    )

    cmake_policy(POP)
    set(TBB_ROOT "${_THRUST_STASH_TBB_ROOT}")
    set(CMAKE_MODULE_PATH "${_THRUST_STASH_MODULE_PATH}")

    if (TARGET TBB::tbb)
      thrust_set_TBB_target(TBB::tbb)
    else()
      thrust_debug("TBB not found!" internal)
    endif()
  endif()
endmacro()

# Wrap the OpenMP flags for CUDA targets
function(thrust_fixup_omp_target omp_target)
  get_target_property(opts ${omp_target} INTERFACE_COMPILE_OPTIONS)
  if (opts MATCHES "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^>]*)>")
    target_compile_options(${omp_target} INTERFACE
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=${CMAKE_MATCH_1}>
    )
  endif()
endfunction()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like OpenMP_CXX_VERSION, etc won't be in the caller's
# scope.
macro(_thrust_find_OMP required)
  if (NOT TARGET Thrust::OMP)
    thrust_debug("Searching for OMP ${required}" internal)
    find_package(OpenMP
      ${_THRUST_QUIET_FLAG}
      ${_THRUST_REQUIRED_FLAG_OMP}
      COMPONENTS CXX
    )

    if (TARGET OpenMP::OpenMP_CXX)
      thrust_fixup_omp_target(OpenMP::OpenMP_CXX)
      thrust_set_OMP_target(OpenMP::OpenMP_CXX)
    else()
      thrust_debug("OpenMP::OpenMP_CXX not found!" internal)
    endif()
  endif()
endmacro()

# This must be a macro instead of a function to ensure that backends passed to
# find_package(Thrust COMPONENTS [...]) have their full configuration loaded
# into the current scope. This provides at least some remedy for CMake issue
# #20670 -- otherwise variables like CUB_VERSION, etc won't be in the caller's
# scope.
macro(_thrust_find_backend backend required)
  # Unfortunately, _thrust_find_${backend}(req) is not valid CMake syntax. Hence
  # why this function exists.
  if ("${backend}" STREQUAL "CPP")
    _thrust_find_CPP("${required}")
  elseif ("${backend}" STREQUAL "CUDA")
    _thrust_find_CUDA("${required}")
  elseif ("${backend}" STREQUAL "TBB")
    _thrust_find_TBB("${required}")
  elseif ("${backend}" STREQUAL "OMP")
    _thrust_find_OMP("${required}")
  else()
    message(FATAL_ERROR "_thrust_find_backend: Invalid system: ${backend}")
  endif()
endmacro()

################################################################################
# Initialization. Executed inside find_package(Thrust) call.
#

if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_THRUST_QUIET ON CACHE INTERNAL "Quiet mode enabled for Thrust find_package calls.")
  set(_THRUST_QUIET_FLAG "QUIET" CACHE INTERNAL "")
else()
  unset(_THRUST_QUIET CACHE)
  unset(_THRUST_QUIET_FLAG CACHE)
endif()

set(_THRUST_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Location of thrust-config.cmake")

# Internal target that actually holds the Thrust interface. Used by all other Thrust targets.
if (NOT TARGET Thrust::Thrust)
  _thrust_declare_interface_alias(Thrust::Thrust _Thrust_Thrust)
  # Pull in the include dir detected by thrust-config-version.cmake
  set(_THRUST_INCLUDE_DIR "${_THRUST_VERSION_INCLUDE_DIR}"
    CACHE INTERNAL "Location of Thrust headers."
  )
  unset(_THRUST_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache
  target_include_directories(_Thrust_Thrust INTERFACE "${_THRUST_INCLUDE_DIR}")
  thrust_debug_target(Thrust::Thrust "${THRUST_VERSION}" internal)
endif()

# Handle find_package COMPONENT requests:
foreach(component ${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})
  if (NOT component IN_LIST THRUST_HOST_SYSTEM_OPTIONS AND
      NOT component IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
    message(FATAL_ERROR "Invalid component requested: '${component}'")
  endif()

  unset(req)
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${component})
    set(req "REQUIRED")
  endif()

  thrust_debug("Preloading COMPONENT '${component}' ${req}" internal)
  _thrust_find_backend(${component} "${req}")
endforeach()

thrust_update_system_found_flags()

include(FindPackageHandleStandardArgs)
if (NOT Thrust_CONFIG)
  set(Thrust_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(Thrust CONFIG_MODE)
