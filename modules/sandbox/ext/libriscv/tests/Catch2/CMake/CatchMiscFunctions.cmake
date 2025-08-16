
#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

include(CheckCXXCompilerFlag)
function(add_cxx_flag_if_supported_to_targets flagname targets)
  string(MAKE_C_IDENTIFIER ${flagname} flag_identifier)
  check_cxx_compiler_flag("${flagname}" HAVE_FLAG_${flag_identifier})

  if(HAVE_FLAG_${flag_identifier})
    foreach(target ${targets})
      target_compile_options(${target} PRIVATE ${flagname})
    endforeach()
  endif()
endfunction()

# Assumes that it is only called for development builds, where warnings
# and Werror is desired, so it also enables Werror.
function(add_warnings_to_targets targets)
  LIST(LENGTH targets TARGETS_LEN)
  # For now we just assume 2 possibilities: msvc and msvc-like compilers,
  # and other.
  if(MSVC)
    foreach(target ${targets})
      # Force MSVC to consider everything as encoded in utf-8
      target_compile_options(${target} PRIVATE /utf-8)
      # Enable Werror equivalent
      if(CATCH_ENABLE_WERROR)
        target_compile_options(${target} PRIVATE /WX)
      endif()

      # MSVC is currently handled specially
      if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        STRING(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}) # override default warning level
        target_compile_options(${target} PRIVATE /w44265 /w44061 /w44062 /w45038)
      endif()
    endforeach()
  endif()

  if(NOT MSVC)
    set(CHECKED_WARNING_FLAGS
      "-Wabsolute-value"
      "-Wall"
      "-Wcall-to-pure-virtual-from-ctor-dtor"
      "-Wcast-align"
      "-Wcatch-value"
      "-Wdangling"
      "-Wdeprecated"
      "-Wdeprecated-register"
      "-Wexceptions"
      "-Wexit-time-destructors"
      "-Wextra"
      "-Wextra-semi"
      "-Wfloat-equal"
      "-Wglobal-constructors"
      "-Winit-self"
      "-Wmisleading-indentation"
      "-Wmismatched-new-delete"
      "-Wmismatched-return-types"
      "-Wmismatched-tags"
      "-Wmissing-braces"
      "-Wmissing-declarations"
      "-Wmissing-noreturn"
      "-Wmissing-prototypes"
      "-Wmissing-variable-declarations"
      "-Wnon-virtual-dtor"
      "-Wnull-dereference"
      "-Wold-style-cast"
      "-Woverloaded-virtual"
      "-Wparentheses"
      "-Wpedantic"
      "-Wredundant-decls"
      "-Wreorder"
      "-Wreturn-std-move"
      "-Wshadow"
      "-Wstrict-aliasing"
      "-Wsubobject-linkage"
      "-Wsuggest-destructor-override"
      "-Wsuggest-override"
      "-Wundef"
      "-Wuninitialized"
      "-Wunneeded-internal-declaration"
      "-Wunreachable-code-aggressive"
      "-Wunused"
      "-Wunused-function"
      "-Wunused-parameter"
      "-Wvla"
      "-Wweak-vtables"

      # This is a useful warning, but our tests sometimes rely on
      # functions being present, but not picked (e.g. various checks
      # for stringification implementation ordering).
      # Ergo, we should use it every now and then, but we cannot
      # enable it by default.
      # "-Wunused-member-function"
    )
    foreach(warning ${CHECKED_WARNING_FLAGS})
      add_cxx_flag_if_supported_to_targets(${warning} "${targets}")
    endforeach()

    if(CATCH_ENABLE_WERROR)
      foreach(target ${targets})
        # Enable Werror equivalent
        target_compile_options(${target} PRIVATE -Werror)
      endforeach()
    endif()
  endif()
endfunction()

# Adds flags required for reproducible build to the target
# Currently only supports GCC and Clang
function(add_build_reproducibility_settings target)
  # Make the build reproducible on versions of g++ and clang that supports -ffile-prefix-map
  if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_cxx_flag_if_supported_to_targets("-ffile-prefix-map=${CATCH_DIR}/=" "${target}")
  endif()
endfunction()
