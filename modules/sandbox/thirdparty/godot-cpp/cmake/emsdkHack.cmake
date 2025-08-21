#[=======================================================================[.rst:
emsdkHack
---------

The Emscripten platform doesn't support the use of shared libraries as known by cmake.

* https://github.com/emscripten-core/emscripten/issues/15276
* https://github.com/emscripten-core/emscripten/issues/17804

This workaround only works due to the way the cmake scripts are loaded.

Prior to the use of ``project( ... )`` directive we need to set
``CMAKE_PROJECT_INCLUDE=cmake/emscripten.cmake``.
This file will be loaded after the toolchain overriding the settings that
prevent shared library building.

CMAKE_PROJECT_INCLUDE was Added in version 3.15.
``CMAKE_PROJECT_<projectName>_INCLUDE`` was Added in version 3.17:

More information on cmake's `code injection`_

.. _code injection:https://cmake.org/cmake/help/latest/command/project.html#code-injection

Overwrite Shared Library Properties to allow shared libs to be generated.
]=======================================================================]
if(EMSCRIPTEN)
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
    set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-sSIDE_MODULE=1")
    set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-sSIDE_MODULE=1")
    set(CMAKE_SHARED_LIBRARY_SUFFIX) # remove the suffix from the shared lib
    set(CMAKE_STRIP FALSE) # used by default in pybind11 on .so modules

    # The Emscripten toolchain sets the default value for EMSCRIPTEN_SYSTEM_PROCESSOR to x86
    # and copies that to CMAKE_SYSTEM_PROCESSOR. We don't want that.
    set(CMAKE_SYSTEM_PROCESSOR "wasm32")
    # the above prevents the need for logic like:
    #if( ${CMAKE_SYSTEM_NAME} STREQUAL Emscripten )
    #    set( SYSTEM_ARCH wasm32 )
    #endif ()
endif()
