#[=======================================================================[.rst:
Web
---

This file contains functions for options and configuration for targeting the
Web platform

]=======================================================================]

# Emscripten requires this hack for use of the SHARED option
set(CMAKE_PROJECT_godot-cpp_INCLUDE cmake/emsdkHack.cmake)

#[==============================[ Web Options ]==============================]
function(web_options)
endfunction()

#[===========================[ Target Generation ]===========================]
function(web_generate)
    target_compile_definitions(godot-cpp PUBLIC WEB_ENABLED UNIX_ENABLED)

    target_compile_options(
        godot-cpp
        PUBLIC #
            -sSIDE_MODULE
            -sSUPPORT_LONGJMP=wasm
            $<${THREADS_ENABLED}:-sUSE_PTHREADS=1>
    )

    target_link_options(
        godot-cpp
        INTERFACE #
            -sWASM_BIGINT
            -sSUPPORT_LONGJMP=wasm
            -fvisibility=hidden
            -shared
    )

    common_compiler_flags()
endfunction()
