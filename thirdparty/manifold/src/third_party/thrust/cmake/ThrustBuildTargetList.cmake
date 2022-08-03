# This file provides utilities for building and working with thrust
# configuration targets.
#
# THRUST_TARGETS
#  - Built by the calling the `thrust_build_target_list()` function.
#  - Each item is the name of a thrust interface target that is configured for a
#    certain combination of host/device/dialect.
#
# thrust_build_target_list()
# - Creates the THRUST_TARGETS list.
#
# The following functions can be used to test/set metadata on a thrust target:
#
# thrust_get_target_property(<prop_var> <target_name> <prop>)
#   - Checks the ${prop} target property on thrust target ${target_name}
#     and sets the ${prop_var} variable in the caller's scope.
#   - <prop_var> is any valid cmake identifier.
#   - <target_name> is the name of a thrust target.
#   - <prop> is one of the following:
#     - HOST: The host system. Valid values: CPP, OMP, TBB.
#     - DEVICE: The device system. Valid values: CUDA, CPP, OMP, TBB.
#     - DIALECT: The C++ dialect. Valid values: 11, 14, 17.
#     - PREFIX: A unique prefix that should be used to name all
#       targets/tests/examples that use this configuration.
#
# thrust_get_target_properties(<target_name>)
#   - Defines ${target_name}_${prop} in the caller's scope, for `prop` in:
#     HOST, DEVICE, DIALECT, PREFIX. See above for details.
#
# thrust_clone_target_properties(<dst_target> <src_target>)
#   - Set the HOST, DEVICE, DIALECT, PREFIX metadata on ${dst_target} to match
#     ${src_target}. See above for details.
#   - This *MUST* be called on any targets that link to another thrust target
#     to ensure that dialect information is updated correctly, e.g.
#     `thrust_clone_target_properties(${my_thrust_test} ${some_thrust_target})`

define_property(TARGET PROPERTY _THRUST_HOST
  BRIEF_DOCS "A target's host system: CPP, TBB, or OMP."
  FULL_DOCS "A target's host system: CPP, TBB, or OMP."
)
define_property(TARGET PROPERTY _THRUST_DEVICE
  BRIEF_DOCS "A target's device system: CUDA, CPP, TBB, or OMP."
  FULL_DOCS "A target's device system: CUDA, CPP, TBB, or OMP."
)
define_property(TARGET PROPERTY _THRUST_DIALECT
  BRIEF_DOCS "A target's C++ dialect: 11, 14, or 17."
  FULL_DOCS "A target's C++ dialect: 11, 14, or 17."
)
define_property(TARGET PROPERTY _THRUST_PREFIX
  BRIEF_DOCS "A prefix describing the config, eg. 'thrust.cpp.cuda.cpp14'."
  FULL_DOCS "A prefix describing the config, eg. 'thrust.cpp.cuda.cpp14'."
)

function(thrust_set_target_properties target_name host device dialect prefix)
  set_target_properties(${target_name}
    PROPERTIES
      _THRUST_HOST ${host}
      _THRUST_DEVICE ${device}
      _THRUST_DIALECT ${dialect}
      _THRUST_PREFIX ${prefix}
  )

  get_property(langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  set(standard_features)
  if (CUDA IN_LIST langs)
    list(APPEND standard_features cuda_std_${dialect})
  endif()
  if (CXX IN_LIST langs)
    list(APPEND standard_features cxx_std_${dialect})
  endif()

  get_target_property(type ${target_name} TYPE)
  if (${type} STREQUAL "INTERFACE_LIBRARY")
    target_compile_features(${target_name} INTERFACE
      ${standard_features}
    )
  else()
    target_compile_features(${target_name} PUBLIC
      ${standard_features}
    )
    set_target_properties(${target_name}
      PROPERTIES
        CXX_STANDARD ${dialect}
        CUDA_STANDARD ${dialect}
        # Must manually request that the standards above are actually respected
        # or else CMake will silently fail to configure the targets correctly...
        # Note that this doesn't actually work as of CMake 3.16:
        # https://gitlab.kitware.com/cmake/cmake/-/issues/20953
        # We'll leave these properties enabled in hopes that they will someday
        # work.
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD_REQUIRED ON
        ARCHIVE_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${THRUST_LIBRARY_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${THRUST_EXECUTABLE_OUTPUT_DIR}"
    )

    # CMake still emits errors about empty CUDA_ARCHITECTURES when CMP0104
    # is set to OLD. This suppresses the errors for good.
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      set_target_properties(${target_name}
        PROPERTIES
          CUDA_ARCHITECTURES OFF
      )
    endif()

    if ("CUDA" STREQUAL "${device}" AND
        "NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
      set_target_properties(${target_name} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      )
    endif()
  endif()
endfunction()

# Get a thrust property from a target and store it in var_name
# thrust_get_target_property(<var_name> <target_name> [HOST|DEVICE|DIALECT|PREFIX]
macro(thrust_get_target_property prop_var target_name prop)
  get_property(${prop_var} TARGET ${target_name} PROPERTY _THRUST_${prop})
endmacro()

# Defines the following string variables in the caller's scope:
# - ${target_name}_HOST
# - ${target_name}_DEVICE
# - ${target_name}_DIALECT
# - ${target_name}_PREFIX
macro(thrust_get_target_properties target_name)
  thrust_get_target_property(${target_name}_HOST ${target_name} HOST)
  thrust_get_target_property(${target_name}_DEVICE ${target_name} DEVICE)
  thrust_get_target_property(${target_name}_DIALECT ${target_name} DIALECT)
  thrust_get_target_property(${target_name}_PREFIX ${target_name} PREFIX)
endmacro()

# Set one target's THRUST_* properties to match another target
function(thrust_clone_target_properties dst_target src_target)
  thrust_get_target_properties(${src_target})
  thrust_set_target_properties(${dst_target}
    ${${src_target}_HOST}
    ${${src_target}_DEVICE}
    ${${src_target}_DIALECT}
    ${${src_target}_PREFIX}
  )
endfunction()

# Set ${var_name} to TRUE or FALSE in the caller's scope
function(_thrust_is_config_valid var_name host device dialect)
  if (THRUST_MULTICONFIG_ENABLE_SYSTEM_${host} AND
      THRUST_MULTICONFIG_ENABLE_SYSTEM_${device} AND
      THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${dialect} AND
      "${host}_${device}" IN_LIST THRUST_MULTICONFIG_WORKLOAD_${THRUST_MULTICONFIG_WORKLOAD}_CONFIGS)
    set(${var_name} TRUE PARENT_SCOPE)
  else()
    set(${var_name} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_thrust_init_target_list)
  set(THRUST_TARGETS "" CACHE INTERNAL "" FORCE)
endfunction()

function(_thrust_add_target_to_target_list target_name host device dialect prefix)
  thrust_set_target_properties(${target_name} ${host} ${device} ${dialect} ${prefix})

  target_link_libraries(${target_name} INTERFACE
    thrust.compiler_interface
    thrust.compiler_interface_cpp${dialect}
  )

  # Workaround Github issue #1174. cudafe promote TBB header warnings to
  # errors, even when they're -isystem includes.
  if ((NOT host STREQUAL "TBB") OR (NOT device STREQUAL "CUDA"))
    target_link_libraries(${target_name} INTERFACE
      thrust.promote_cudafe_warnings
    )
  endif()

  set(THRUST_TARGETS ${THRUST_TARGETS} ${target_name} CACHE INTERNAL "" FORCE)

  set(label "${host}.${device}.cpp${dialect}")
  string(TOLOWER "${label}" label)
  message(STATUS "Enabling Thrust configuration: ${label}")
endfunction()

function(_thrust_build_target_list_multiconfig)
  # Detect supported dialects if requested -- this must happen after CUDA is
  # enabled, if it's going to be enabled.
  if (THRUST_MULTICONFIG_ENABLE_DIALECT_ALL OR
      THRUST_MULTICONFIG_ENABLE_DIALECT_LATEST)
    message(STATUS "Testing for supported language standards...")
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DetectSupportedStandards.cmake")
    detect_supported_standards(THRUST CXX ${THRUST_CPP_DIALECT_OPTIONS})
    if (THRUST_CUDA_FOUND)
      detect_supported_standards(THRUST CUDA ${THRUST_CPP_DIALECT_OPTIONS})
    endif()

    # Take the union of supported standards in CXX and CUDA:
    set(supported_dialects)
    set(latest_dialect 11)
    foreach(standard IN LISTS THRUST_CPP_DIALECT_OPTIONS)
      if ((THRUST_CXX_${standard}_SUPPORTED) AND
          ((NOT THRUST_CUDA_FOUND) OR THRUST_CUDA_${standard}_SUPPORTED))

        # MSVC silently promotes C++11 to C++14 -- skip it:
        if ((${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC) AND (standard EQUAL 11))
          continue()
        endif()

        list(APPEND supported_dialects ${standard})
        if (latest_dialect LESS standard)
          set(latest_dialect ${standard})
        endif()
      endif()
    endforeach()

    if (THRUST_MULTICONFIG_ENABLE_DIALECT_ALL)
      foreach(standard IN LISTS THRUST_CPP_DIALECT_OPTIONS)
        if (standard IN_LIST supported_dialects)
          set(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${standard} ON CACHE BOOL
              "Generate C++${dialect} build configurations." FORCE
          )
        else()
          set(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${standard} OFF CACHE BOOL
            "Generate C++${dialect} build configurations." FORCE
            )
        endif()
      endforeach()
    elseif(THRUST_MULTICONFIG_ENABLE_DIALECT_LATEST)
      foreach(standard IN LISTS THRUST_CPP_DIALECT_OPTIONS)
        if (standard EQUAL latest_dialect)
          set(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${standard} ON CACHE BOOL
            "Generate C++${dialect} build configurations." FORCE
            )
        else()
          set(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${standard} OFF CACHE BOOL
            "Generate C++${dialect} build configurations." FORCE
            )
        endif()
      endforeach()
    endif()
  endif()

  # Supported versions of MSVC do not distinguish between C++11 and C++14.
  # Warn the user that they may be generating a ton of redundant targets if
  # they explicitly requested this configuration.
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}" AND
      THRUST_MULTICONFIG_ENABLE_DIALECT_CPP11)
    message(WARNING
      "Supported versions of MSVC (2017+) do not distinguish between C++11 "
      "and C++14. The requested C++11 targets may be redundant."
    )
  endif()

  # Build THRUST_TARGETS
  foreach(host IN LISTS THRUST_HOST_SYSTEM_OPTIONS)
    foreach(device IN LISTS THRUST_DEVICE_SYSTEM_OPTIONS)
      foreach(dialect IN LISTS THRUST_CPP_DIALECT_OPTIONS)
        _thrust_is_config_valid(config_valid ${host} ${device} ${dialect})
        if (config_valid)
          set(prefix "thrust.${host}.${device}.cpp${dialect}")
          string(TOLOWER "${prefix}" prefix)

          # Configure a thrust interface target for this host/device
          set(target_name "${prefix}")
          thrust_create_target(${target_name}
            HOST ${host}
            DEVICE ${device}
            ${THRUST_TARGET_FLAGS}
          )

          # Set configuration metadata for this thrust interface target:
          _thrust_add_target_to_target_list(${target_name}
            ${host} ${device} ${dialect} ${prefix}
          )

          # Create a meta target for all targets in this configuration:
          add_custom_target(${prefix}.all)
          add_dependencies(thrust.all ${prefix}.all)
        endif()
      endforeach() # dialects
    endforeach() # devices
  endforeach() # hosts

  list(LENGTH THRUST_TARGETS count)
  message(STATUS "${count} unique thrust.host.device.dialect configurations generated")
endfunction()

function(_thrust_build_target_list_singleconfig)
  set(host ${THRUST_HOST_SYSTEM})
  set(device ${THRUST_DEVICE_SYSTEM})
  set(dialect ${THRUST_CPP_DIALECT})
  set(prefix "thrust") # single config

  _thrust_add_target_to_target_list(thrust ${host} ${device} ${dialect} ${prefix})
endfunction()

# Build a ${THRUST_TARGETS} list containing target names for all
# requested configurations
function(thrust_build_target_list)
  # Clear the list of targets:
  _thrust_init_target_list()

  # Generic config flags:
  set(THRUST_TARGET_FLAGS)
  macro(add_flag_option flag docstring default)
    set(opt "THRUST_${flag}")
    option(${opt} "${docstring}" "${default}")
    mark_as_advanced(${opt})
    if (${${opt}})
      list(APPEND THRUST_TARGET_FLAGS ${flag})
    endif()
  endmacro()
  add_flag_option(IGNORE_DEPRECATED_CPP_DIALECT "Don't warn about any deprecated C++ standards and compilers." OFF)
  add_flag_option(IGNORE_DEPRECATED_CPP_11 "Don't warn about deprecated C++11." OFF)
  add_flag_option(IGNORE_DEPRECATED_COMPILER "Don't warn about deprecated compilers." OFF)
  add_flag_option(IGNORE_CUB_VERSION_CHECK "Don't warn about mismatched CUB versions." OFF)
  add_flag_option(IGNORE_DEPRECATED_API "Don't warn about deprecated Thrust or CUB APIs." OFF)

  # Top level meta-target. Makes it easier to just build thrust targets when
  # building both CUB and Thrust. Add all project files here so IDEs will be
  # aware of them. This will not generate build rules.
  file(GLOB_RECURSE all_sources
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    "${Thrust_SOURCE_DIR}/thrust/*.h"
    "${Thrust_SOURCE_DIR}/thrust/*.inl"
  )
  add_custom_target(thrust.all SOURCES ${all_sources})

  if (THRUST_ENABLE_MULTICONFIG)
    _thrust_build_target_list_multiconfig()
  else()
    _thrust_build_target_list_singleconfig()
  endif()
endfunction()
