#[=======================================================================[.rst:
Windows
-------
This file contains functions for options and configuration for targeting the
Windows platform

Because this file is included into the top level CMakelists.txt before the
project directive, it means that

* ``CMAKE_CURRENT_SOURCE_DIR`` is the location of godot-cpp's CMakeLists.txt
* ``CMAKE_SOURCE_DIR`` is the location where any prior ``project(...)``
    directive was

MSVC Runtime Selection
----------------------

There are two main ways to set the msvc runtime library;
Using ``target_compile_options()`` to add the flags
or using the ``CMAKE_MSVC_RUNTIME_LIBRARY`` property_ abstraction, introduced
in CMake version 3.15 with the policy CMP0091_ to remove the flags from
``CMAKE_<LANG>_FLAGS_<CONFIG>``.

Default: ``CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"``

This initializes each target's ``MSVC_RUNTIME_LIBRARY`` property at the time of
target creation.

it is stated in the msvc_ documentation that: "All modules passed to a given
invocation of the linker must have been compiled with the same runtime library
compiler option (/MD, /MT, /LD)."

This creates a conundrum for us, the ``CMAKE_MSVC_RUNTIME_LIBRARY`` needs to be
correct at the time the target is created, but we have no control over the
consumers CMake scripts, and the per-target ``MSVC_RUNTIME_LIBRARY`` property
is not transient.

It has been raised that not using ``CMAKE_MSVC_RUNTIME_LIBRARY`` can also cause
issues_ when a dependency( independent to godot-cpp ) that doesn't set any
runtime flags, which relies purely on the ``CMAKE_MSVC_RUNTIME_LIBRARY``
variable will very likely not have the correct msvc runtime flags set.

So we'll set ``CMAKE_MSVC_RUNTIME_LIBRARY`` as CACHE STRING so that it will be
available for consumer target definitions, but also be able to be overridden if
needed.

Additionally we message consumers notifying them and pointing to this
documentation.

.. _CMP0091:https://cmake.org/cmake/help/latest/policy/CMP0091.html
.. _property:https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html
.. https://discourse.cmake.org/t/mt-staticrelease-doesnt-match-value-md-dynamicrelease/5428/4
.. _msvc: https://learn.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library
.. _issues: https://github.com/godotengine/godot-cpp/issues/1699

]=======================================================================]

#[============================[ Windows Options ]============================]
function(windows_options)
    #[[ Options from SCons

    TODO silence_msvc: Silence MSVC's cl/link stdout bloat, redirecting errors to stderr
        Default: True

    These three options will not implemented as compiler selection is managed
    by CMake toolchain files. Look to doc/cmake.rst for examples.
    use_mingw: Use the MinGW compiler instead of MSVC - only effective on Windows
    use_llvm: Use the LLVM compiler (MVSC or MinGW depending on the use_mingw flag
    mingw_prefix: MinGW prefix
    ]]

    option(GODOTCPP_USE_STATIC_CPP "Link MinGW/MSVC C++ runtime libraries statically" ON)
    option(GODOTCPP_DEBUG_CRT "Compile with MSVC's debug CRT (/MDd)" OFF)

    message(
        STATUS
        "If not already cached, setting CMAKE_MSVC_RUNTIME_LIBRARY.\n"
        "\tFor more information please read godot-cpp/cmake/windows.cmake"
    )

    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<IF:$<BOOL:${GODOTCPP_DEBUG_CRT}>,DebugDLL,$<$<NOT:$<BOOL:${GODOTCPP_USE_STATIC_CPP}>>:DLL>>"
        CACHE STRING
        "Select the MSVC runtime library for use by compilers targeting the MSVC ABI."
    )
endfunction()

#[===========================[ Target Generation ]===========================]
function(windows_generate)
    set(STATIC_CPP "$<BOOL:${GODOTCPP_USE_STATIC_CPP}>")

    set_target_properties(godot-cpp PROPERTIES PDB_OUTPUT_DIRECTORY "$<1:${CMAKE_SOURCE_DIR}/bin>")

    target_compile_definitions(
        godot-cpp
        PUBLIC WINDOWS_ENABLED $<${IS_MSVC}: TYPED_METHOD_BIND NOMINMAX >
    )

    # gersemi: off
    target_link_options(
        godot-cpp
        PUBLIC
            $<${NOT_MSVC}:
                -Wl,--no-undefined
                $<${STATIC_CPP}:
                    -static
                    -static-libgcc
                    -static-libstdc++
                >
            >

            $<${IS_CLANG}:-lstdc++>
    )
    # gersemi: on

    common_compiler_flags()
endfunction()
