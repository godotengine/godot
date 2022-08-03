#
# This file defines the `thrust_build_compiler_targets()` function, which
# creates the following interface targets:
#
# thrust.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   Thrust's tests, examples, etc.
#
# thrust.compiler_interface_cpp11
# thrust.compiler_interface_cpp14
# thrust.compiler_interface_cpp17
# - Interface targets providing compiler-specific options that should only be
#   applied to certain dialects of C++.
#
# thrust.promote_cudafe_warnings
# - Interface target that adds warning promotion for NVCC cudafe invocations.
# - Only exists to work around github issue #1174 on tbb.cuda configurations.
# - May be combined with thrust.compiler_interface when #1174 is fully resolved.
#
# thrust.silence_unreachable_code_warnings
# - Interface target that silences unreachable code warnings.
# - Used to selectively disable such warnings in unit tests caused by
#   unconditionally thrown exceptions.

function(thrust_build_compiler_targets)
  set(cxx_compile_definitions)
  set(cxx_compile_options)

  thrust_update_system_found_flags()

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    append_option_if_available("/W4" cxx_compile_options)

    # Treat all warnings as errors. This is only supported on Release builds,
    # as `nv_exec_check_disable` doesn't seem to work with MSVC debug iterators
    # and spurious warnings are emitted.
    # See NVIDIA/thrust#1273, NVBug 3129879.
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
      append_option_if_available("/WX" cxx_compile_options)
    endif()

    # Suppress overly-pedantic/unavoidable warnings brought in with /W4:
    # C4324: structure was padded due to alignment specifier
    append_option_if_available("/wd4324" cxx_compile_options)
    # C4505: unreferenced local function has been removed
    # The CUDA `host_runtime.h` header emits this for
    # `__cudaUnregisterBinaryUtil`.
    append_option_if_available("/wd4505" cxx_compile_options)
    # C4706: assignment within conditional expression
    # MSVC doesn't provide an opt-out for this warning when the assignment is
    # intentional. Clang will warn for these, but suppresses the warning when
    # double-parentheses are used around the assignment. We'll let Clang catch
    # unintentional assignments and suppress all such warnings on MSVC.
    append_option_if_available("/wd4706" cxx_compile_options)

    # Disabled loss-of-data conversion warnings.
    # TODO Re-enable.
    append_option_if_available("/wd4244" cxx_compile_options)

    # Disable warning about applying unary operator- to unsigned type.
    # TODO Re-enable.
    append_option_if_available("/wd4146" cxx_compile_options)

    # MSVC STL assumes that `allocator_traits`'s allocator will use raw pointers,
    # and the `__DECLSPEC_ALLOCATOR` macro causes issues with thrust's universal
    # allocators:
    #   warning C4494: 'std::allocator_traits<_Alloc>::allocate' :
    #      Ignoring __declspec(allocator) because the function return type is not
    #      a pointer or reference
    # See https://github.com/microsoft/STL/issues/696
    append_option_if_available("/wd4494" cxx_compile_options)

    # Some of the async tests require /bigobj to fit all their sections into the
    # object files:
    append_option_if_available("/bigobj" cxx_compile_options)

    # "Oh right, this is Visual Studio."
    list(APPEND cxx_compile_definitions "NOMINMAX")
  else()
    append_option_if_available("-Werror" cxx_compile_options)
    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wpointer-arith" cxx_compile_options)
    append_option_if_available("-Wunused-local-typedef" cxx_compile_options)
    append_option_if_available("-Wvla" cxx_compile_options)

    # Disable GNU extensions (flag is clang only)
    append_option_if_available("-Wgnu" cxx_compile_options)
    # Calling a variadic macro with zero args is a GNU extension until C++20,
    # but the THRUST_PP_ARITY macro is used with zero args. Need to see if this
    # is a real problem worth fixing.
    append_option_if_available("-Wno-gnu-zero-variadic-macro-arguments" cxx_compile_options)

    # This complains about functions in CUDA system headers when used with nvcc.
    append_option_if_available("-Wno-unused-function" cxx_compile_options)
  endif()

  if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
      # GCC 7.3 complains about name mangling changes due to `noexcept`
      # becoming part of the type system; we don't care.
      append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
    endif()
  endif()

  if ("Intel" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # Disable warning that inlining is inhibited by compiler thresholds.
    append_option_if_available("-diag-disable=11074" cxx_compile_options)
    append_option_if_available("-diag-disable=11076" cxx_compile_options)
  endif()

  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    # Today:
    # * NVCC accepts CUDA C++ in .cu files but not .cpp files.
    # * NVC++ accepts CUDA C++ in .cpp files but not .cu files.
    # TODO: This won't be necessary in the future.
    list(APPEND cxx_compile_options -cppsuffix=cu)
  endif()

  add_library(thrust.compiler_interface INTERFACE)

  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(thrust.compiler_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVCXX>>:${cxx_option}>
      # Only use -Xcompiler with NVCC, not NVC++.
      #
      # CMake can't split genexs, so this can't be formatted better :(
      # This is:
      # if (using CUDA and CUDA_COMPILER is NVCC) add -Xcompiler=opt:
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=${cxx_option}>
    )
  endforeach()

  foreach (cxx_definition IN LISTS cxx_compile_definitions)
    # Add these for both CUDA and CXX targets:
    target_compile_definitions(thrust.compiler_interface INTERFACE
      ${cxx_definition}
    )
  endforeach()

  # Display warning numbers from nvcc cudafe errors:
  target_compile_options(thrust.compiler_interface INTERFACE
    # If using CUDA w/ NVCC...
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--display_error_number>
  )

  # Tell NVCC to be quiet about deprecated GPU targets:
  target_compile_options(thrust.compiler_interface INTERFACE
    # If using CUDA w/ NVCC...
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Wno-deprecated-gpu-targets>
  )

  # This is kept separate for Github issue #1174.
  add_library(thrust.promote_cudafe_warnings INTERFACE)
  target_compile_options(thrust.promote_cudafe_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--promote_warnings>
  )

  # Some of our unit tests unconditionally throw exceptions, and compilers will
  # detect that the following instructions are unreachable. This is intentional
  # and unavoidable in these cases. This target can be used to silence
  # unreachable code warnings.
  add_library(thrust.silence_unreachable_code_warnings INTERFACE)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(thrust.silence_unreachable_code_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4702>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=/wd4702>
    )
  endif()

  # These targets are used for dialect-specific options:
  add_library(thrust.compiler_interface_cpp11 INTERFACE)
  add_library(thrust.compiler_interface_cpp14 INTERFACE)
  add_library(thrust.compiler_interface_cpp17 INTERFACE)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4127: conditional expression is constant
    # Disable this MSVC warning for C++11/C++14. In C++17, we can use
    # THRUST_IF_CONSTEXPR to address these warnings.
    target_compile_options(thrust.compiler_interface_cpp11 INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4127>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=/wd4127>
    )
    target_compile_options(thrust.compiler_interface_cpp14 INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/wd4127>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=/wd4127>
    )
  endif()

endfunction()
