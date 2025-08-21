#[=======================================================================[.rst:
Common Compiler Flags
---------------------

This file contains host platform toolchain and target platform agnostic
configuration. It includes flags like optimization levels, warnings, and
features. For target platform specific flags look to each of the
``cmake/<platform>.cmake`` files.

The default compile and link options CMake adds can be found in the
platform modules_. When a project is created it initializes its variables from
the ``CMAKE_*`` values. The cleanest way I have found to alter these defaults
is the use of the ``CMAKE_PROJECT_<PROJECT-NAME>_INCLUDE`` as demonstrated by
the emsdkHack.cmake to overcome the limitation on shared library creation.

So far the emsdkHack is the only modification to the defaults we have made.

.. _modules: https://github.com/Kitware/CMake/blob/master/Modules/Platform/

]=======================================================================]

#[[ Compiler Configuration, not to be confused with build targets ]]
set(DEBUG_SYMBOLS "$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>")

#[[ Compiler Identification ]]
set(IS_CLANG "$<CXX_COMPILER_ID:Clang>")
set(IS_APPLECLANG "$<CXX_COMPILER_ID:AppleClang>")
set(IS_GNU "$<CXX_COMPILER_ID:GNU>")
set(IS_MSVC "$<CXX_COMPILER_ID:MSVC>")
set(NOT_MSVC "$<NOT:$<CXX_COMPILER_ID:MSVC>>")

set(LT_V8 "$<VERSION_LESS:$<CXX_COMPILER_VERSION>,8>")
set(GE_V9 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,9>")
set(GT_V11 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>")
set(LT_V11 "$<VERSION_LESS:$<CXX_COMPILER_VERSION>,11>")
set(GE_V12 "$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>")

#[===========================[ compiler_detection ]===========================]
#[[ Check for clang-cl with MSVC frontend
The compiler is tested and set when the project command is called.
The variable CXX_COMPILER_FRONTEND_VARIANT was introduced in 3.14
The generator expression $<CXX_COMPILER_FRONTEND_VARIANT> wasn't introduced
until CMake 3.30 so we can't use it yet.

So to support clang downloaded from llvm.org which uses the MSVC frontend
by default, we need to test for it. ]]
function(compiler_detection)
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
        if(${CMAKE_CXX_COMPILER_FRONTEND_VARIANT} STREQUAL MSVC)
            message(STATUS "Using clang-cl")
            set(IS_CLANG "0" PARENT_SCOPE)
            set(IS_MSVC "1" PARENT_SCOPE)
            set(NOT_MSVC "0" PARENT_SCOPE)
        endif()
    endif()
endfunction()

#[=========================[ common_compiler_flags ]=========================]
#[[ This function assumes it is being called from within one of the platform
generate functions, with all the variables from lower scopes defined. ]]
function(common_compiler_flags)
    # gersemi: off
    # These compiler options reflect what is in godot/SConstruct.
    target_compile_options(
        godot-cpp
        # The public flag tells CMake that the following options are transient,
        # and will propagate to consumers.
        PUBLIC
            # Disable exception handling. Godot doesn't use exceptions anywhere, and this
            # saves around 20% of binary size and very significant build time.
            $<${DISABLE_EXCEPTIONS}:$<${NOT_MSVC}:-fno-exceptions>>

            # Enabling Debug Symbols
            $<${DEBUG_SYMBOLS}:
                # Adding dwarf-4 explicitly makes stacktraces work with clang builds,
                # otherwise addr2line doesn't understand them.
                $<${NOT_MSVC}:
                    -gdwarf-4
                    $<IF:${IS_DEV_BUILD},-g3,-g2>
                >
            >

            $<${IS_DEV_BUILD}:$<${NOT_MSVC}:-fno-omit-frame-pointer -O0>>

            $<${HOT_RELOAD}:$<${IS_GNU}:-fno-gnu-unique>>

            # MSVC only
            $<${IS_MSVC}:
                # /MP isn't valid for clang-cl with msvc frontend
                $<$<CXX_COMPILER_ID:MSVC>:/MP${PROC_N}>

                # Interpret source files as utf-8
                /utf-8
            >

        # Warnings below, these do not need to propagate to consumers.
        PRIVATE
            $<${IS_MSVC}:
                /W4      # Warning level 4 (informational) warnings that aren't off by default.

                # Disable warnings which we don't plan to fix.
                /wd4100  # C4100 (unreferenced formal parameter): Doesn't play nice with polymorphism.
                /wd4127  # C4127 (conditional expression is constant)
                /wd4201  # C4201 (non-standard nameless struct/union): Only relevant for C89.
                /wd4244  # C4244 C4245 C4267 (narrowing conversions): Unavoidable at this scale.
                /wd4245
                /wd4267
                /wd4305  # C4305 (truncation): double to float or real_t, too hard to avoid.
                /wd4514  # C4514 (unreferenced inline function has been removed)
                /wd4714  # C4714 (function marked as __forceinline not inlined)
                /wd4820  # C4820 (padding added after construct)
            >

            # Clang and GNU common options
            $<$<OR:${IS_CLANG},${IS_GNU}>:
                -Wall
                -Wctor-dtor-privacy
                -Wextra
                -Wno-unused-parameter
                -Wnon-virtual-dtor
                -Wwrite-strings
            >

            # Clang only
            $<${IS_CLANG}:
                -Wimplicit-fallthrough
                -Wno-ordered-compare-function-pointers
            >

            # GNU only
            $<${IS_GNU}:
                -Walloc-zero
                -Wduplicated-branches
                -Wduplicated-cond
                -Wno-misleading-indentation
                -Wplacement-new=1
                -Wshadow-local
                -Wstringop-overflow=4

                # Bogus warning fixed in 8+.
                $<${LT_V8}:-Wno-strict-overflow>

                $<${GE_V9}:-Wattribute-alias=2>

                # Broke on MethodBind templates before GCC 11.
                $<${GT_V11}:-Wlogical-op>

                # Regression in GCC 9/10, spams so much in our variadic templates that we need to outright disable it.
                $<${LT_V11}:-Wno-type-limits>

                # False positives in our error macros, see GH-58747.
                $<${GE_V12}:-Wno-return-type>
            >
    )

    target_compile_definitions(
        godot-cpp
        PUBLIC
            GDEXTENSION

            # features
            $<${DEBUG_FEATURES}:DEBUG_ENABLED>

            $<${IS_DEV_BUILD}:DEV_ENABLED>

            $<${HOT_RELOAD}:HOT_RELOAD_ENABLED>

            $<$<STREQUAL:${GODOTCPP_PRECISION},double>:REAL_T_IS_DOUBLE>

            $<${IS_MSVC}:$<${DISABLE_EXCEPTIONS}:_HAS_EXCEPTIONS=0>>

            $<${THREADS_ENABLED}:THREADS_ENABLED>
    )

    target_link_options(
        godot-cpp
        PUBLIC
            $<${DEBUG_SYMBOLS}:$<${IS_MSVC}:/DEBUG:FULL>>

            $<$<NOT:${DEBUG_SYMBOLS}>:
                $<${IS_GNU}:-s>
                $<${IS_CLANG}:-s>
                $<${IS_APPLECLANG}:-Wl,-S -Wl,-x -Wl,-dead_strip>
            >
        PRIVATE
            $<${IS_MSVC}:
                /WX             # treat link warnings as errors.
                /MANIFEST:NO    # We dont need a manifest
            >
    )
    # gersemi: on
endfunction()
