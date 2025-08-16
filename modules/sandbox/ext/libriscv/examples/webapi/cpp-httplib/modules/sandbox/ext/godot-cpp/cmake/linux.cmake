#[=======================================================================[.rst:
Linux
-----

This file contains functions for options and configuration for targeting the
Linux platform

]=======================================================================]

#[=============================[ Linux Options ]=============================]
function(linux_options)
    #[[ Options from SCons
    use_llvm : Use the LLVM compiler
        Not implemented as compiler selection is managed by CMake. Look to
        doc/cmake.rst for examples.
    ]]
    option(GODOTCPP_USE_STATIC_CPP "Link libgcc and libstdc++ statically for better portability" ON)
endfunction()

#[===========================[ Target Generation ]===========================]
function(linux_generate)
    set(STATIC_CPP "$<BOOL:${GODOTCPP_USE_STATIC_CPP}>")

    target_compile_definitions(godot-cpp PUBLIC LINUX_ENABLED UNIX_ENABLED)

    # gersemi: off
    target_link_options(
        godot-cpp
        PUBLIC
            $<${STATIC_CPP}:
                -static-libgcc
                -static-libstdc++
            >
    )
    # gersemi: on

    common_compiler_flags()
endfunction()
