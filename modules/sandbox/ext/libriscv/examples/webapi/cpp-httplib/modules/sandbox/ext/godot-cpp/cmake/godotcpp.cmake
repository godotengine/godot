#[=======================================================================[.rst:
godotcpp.cmake
--------------

As godot-cpp is a C++ project, there are no C files, and detection of a C
compiler is unnecessary. When CMake performs the configure process, if a
C compiler is specified, like in a toolchain, or from an IDE, then it will
print a warning stating that the CMAKE_C_COMPILER compiler is unused.
This if statement simply silences that warning.
]=======================================================================]
if(CMAKE_C_COMPILER)
endif()

#[[ Include Platform Files
Because these files are included into the top level CMakeLists.txt before the
project directive, it means that

CMAKE_CURRENT_SOURCE_DIR is the location of godot-cpp's CMakeLists.txt
CMAKE_SOURCE_DIR is the location where any prior project() directive was ]]
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GodotCPPModule.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common_compiler_flags.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/android.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ios.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/macos.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/web.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows.cmake)

# Detect number of processors
include(ProcessorCount)
ProcessorCount(PROC_MAX)
message(STATUS "Auto-detected ${PROC_MAX} CPU cores available for build parallelism.")

# List of known platforms
set(PLATFORM_LIST
    linux
    macos
    windows
    android
    ios
    web
)

# List of known architectures
set(ARCH_LIST
    x86_32
    x86_64
    arm32
    arm64
    rv64
    ppc32
    ppc64
    wasm32
)

#[=============================[ godot_arch_name ]=============================]
#[[ Function to map CMAKE_SYSTEM_PROCESSOR names to godot arch equivalents ]]
function(godot_arch_name OUTVAR)
    # Special case for macos universal builds that target both x86_64 and arm64
    if(DEFINED CMAKE_OSX_ARCHITECTURES)
        if("x86_64" IN_LIST CMAKE_OSX_ARCHITECTURES AND "arm64" IN_LIST CMAKE_OSX_ARCHITECTURES)
            set(${OUTVAR} "universal" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Direct match early out.
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH)
    if(ARCH IN_LIST ARCH_LIST)
        set(${OUTVAR} "${ARCH}" PARENT_SCOPE)
        return()
    endif()

    # Known aliases
    set(x86_64 "w64;amd64;x86-64")
    set(arm32 "armv7;armv7-a")
    set(arm64 "armv8;arm64v8;aarch64;armv8-a")
    set(rv64 "rv;riscv;riscv64")
    set(ppc32 "ppcle;ppc")
    set(ppc64 "ppc64le")

    if(ARCH IN_LIST x86_64)
        set(${OUTVAR} "x86_64" PARENT_SCOPE)
    elseif(ARCH IN_LIST arm32)
        set(${OUTVAR} "arm32" PARENT_SCOPE)
    elseif(ARCH IN_LIST arm64)
        set(${OUTVAR} "arm64" PARENT_SCOPE)
    elseif(ARCH IN_LIST rv64)
        set(${OUTVAR} "rv64" PARENT_SCOPE)
    elseif(ARCH IN_LIST ppc32)
        set(${OUTVAR} "ppc32" PARENT_SCOPE)
    elseif(ARCH IN_LIST ppc64)
        set(${OUTVAR} "ppc64" PARENT_SCOPE)
    elseif(ARCH MATCHES "86")
        # Catches x86, i386, i486, i586, i686, etc.
        set(${OUTVAR} "x86_32" PARENT_SCOPE)
    else()
        # Default value is whatever the processor is.
        set(${OUTVAR} ${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
    endif()
endfunction()

# Function to define all the options.
function(godotcpp_options)
    #NOTE: platform is managed using toolchain files.
    #NOTE: arch is managed by using toolchain files.
    # To create a universal build for macos, set CMAKE_OSX_ARCHITECTURES

    set(GODOTCPP_TARGET
        "template_debug"
        CACHE STRING
        "Which target to generate. valid values are: template_debug, template_release, and editor"
    )
    set_property(CACHE GODOTCPP_TARGET PROPERTY STRINGS "template_debug;template_release;editor")

    # Input from user for GDExtension interface header and the API JSON file
    set(GODOTCPP_GDEXTENSION_DIR
        "gdextension"
        CACHE PATH
        "Path to a custom directory containing GDExtension interface header and API JSON file ( /path/to/gdextension_dir )"
    )
    set(GODOTCPP_CUSTOM_API_FILE
        ""
        CACHE FILEPATH
        "Path to a custom GDExtension API JSON file (takes precedence over `GODOTCPP_GDEXTENSION_DIR`) ( /path/to/custom_api_file )"
    )

    #TODO generate_bindings

    option(GODOTCPP_GENERATE_TEMPLATE_GET_NODE "Generate a template version of the Node class's get_node. (ON|OFF)" ON)

    #TODO build_library

    set(GODOTCPP_PRECISION "single" CACHE STRING "Set the floating-point precision level (single|double)")

    set(GODOTCPP_THREADS ON CACHE BOOL "Enable threading support")

    #TODO compiledb
    #TODO compiledb_file

    set(GODOTCPP_BUILD_PROFILE "" CACHE PATH "Path to a file containing a feature build profile")

    set(GODOTCPP_USE_HOT_RELOAD "" CACHE BOOL "Enable the extra accounting required to support hot reload. (ON|OFF)")

    # Disable exception handling. Godot doesn't use exceptions anywhere, and this
    # saves around 20% of binary size and very significant build time (GH-80513).
    option(GODOTCPP_DISABLE_EXCEPTIONS "Force disabling exception handling code (ON|OFF)" ON)

    set(GODOTCPP_SYMBOL_VISIBILITY
        "hidden"
        CACHE STRING
        "Symbols visibility on GNU platforms. Use 'auto' to apply the default value. (auto|visible|hidden)"
    )
    set_property(CACHE GODOTCPP_SYMBOL_VISIBILITY PROPERTY STRINGS "auto;visible;hidden")

    #TODO optimize

    option(GODOTCPP_DEV_BUILD "Developer build with dev-only debugging code (DEV_ENABLED)" OFF)

    #[[ debug_symbols
    Debug symbols are enabled by using the Debug or RelWithDebInfo build configurations.
    Single Config Generator is set at configure time

        cmake ../ -DCMAKE_BUILD_TYPE=Debug

    Multi-Config Generator is set at build time

        cmake --build . --config Debug

    ]]

    # FIXME These options are not present in SCons, and perhaps should be added there.
    option(GODOTCPP_SYSTEM_HEADERS "Expose headers as SYSTEM." OFF)
    option(GODOTCPP_WARNING_AS_ERROR "Treat warnings as errors" OFF)

    # Enable Testing
    option(GODOTCPP_ENABLE_TESTING "Enable the godot-cpp.test.<target> integration testing targets" OFF)

    #[[ Target Platform Options ]]
    android_options()
    ios_options()
    linux_options()
    macos_options()
    web_options()
    windows_options()
endfunction()

#[===========================[ Target Generation ]===========================]
function(godotcpp_generate)
    #[[ Multi-Threaded MSVC Compilation
    When using the MSVC compiler the build command -j <n> only specifies
    parallel jobs or targets, and not multi-threaded compilation To speed up
    compile times on msvc, the /MP <n> flag can be set. But we need to set it
    at configure time.

    MSVC is true when the compiler is some version of Microsoft Visual C++ or
    another compiler simulating the Visual C++ cl command-line syntax. ]]
    if(MSVC)
        math(EXPR PROC_N "(${PROC_MAX}-1) | (${X}-2)>>31 & 1")
        message(STATUS "Using ${PROC_N} cores for multi-threaded compilation.")
        # TODO You can override it at configure time with ...." )
    else()
        if(CMAKE_BUILD_PARALLEL_LEVEL)
            set(_cores "${CMAKE_BUILD_PARALLEL_LEVEL}")
        else()
            set(_cores "all")
        endif()
        message(
            STATUS
            "Using ${_cores} cores. You can override"
            " this at configure time by using -j <n> or --parallel <n> in the build"
            " command."
        )
        message(STATUS "  eg. cmake --build . -j 7  ...")
    endif()

    #[[ GODOTCPP_SYMBOL_VISIBLITY
    To match the SCons options, the allowed values are "auto", "visible", and "hidden"
    This effects the compiler flag_ -fvisibility=[default|internal|hidden|protected]
    The corresponding target option CXX_VISIBILITY_PRESET accepts the compiler values.

    TODO: It is probably worth a pull request which changes both to use the compiler values
    .. _flag:https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#index-fvisibility
    ]]
    if(${GODOTCPP_SYMBOL_VISIBILITY} STREQUAL "auto" OR ${GODOTCPP_SYMBOL_VISIBILITY} STREQUAL "visible")
        set(GODOTCPP_SYMBOL_VISIBILITY "default")
    endif()

    # Setup variable to optionally mark headers as SYSTEM
    set(GODOTCPP_SYSTEM_HEADERS_ATTRIBUTE "")
    if(GODOTCPP_SYSTEM_HEADERS)
        set(GODOTCPP_SYSTEM_HEADERS_ATTRIBUTE SYSTEM)
    endif()

    #[[ Configure Binding Variables ]]
    # Generate Binding Parameters (True|False)
    set(USE_TEMPLATE_GET_NODE "False")
    if(GODOTCPP_GENERATE_TEMPLATE_GET_NODE)
        set(USE_TEMPLATE_GET_NODE "True")
    endif()

    # Bits (32|64)
    math(EXPR BITS "${CMAKE_SIZEOF_VOID_P} * 8") # CMAKE_SIZEOF_VOID_P refers to target architecture.

    # API json File
    set(GODOTCPP_GDEXTENSION_API_FILE "${GODOTCPP_GDEXTENSION_DIR}/extension_api.json")
    if(GODOTCPP_CUSTOM_API_FILE) # User-defined override.
        set(GODOTCPP_GDEXTENSION_API_FILE "${GODOTCPP_CUSTOM_API_FILE}")
    endif()

    # Build Profile
    if(GODOTCPP_BUILD_PROFILE)
        message(STATUS "Using build profile to trim api file")
        message(STATUS "\tBUILD_PROFILE = '${GODOTCPP_BUILD_PROFILE}'")
        message(STATUS "\tAPI_SOURCE = '${GODOTCPP_GDEXTENSION_API_FILE}'")
        build_profile_generate_trimmed_api(
                "${GODOTCPP_BUILD_PROFILE}"
                "${GODOTCPP_GDEXTENSION_API_FILE}"
                "${CMAKE_CURRENT_BINARY_DIR}/extension_api.json"
        )
        set(GODOTCPP_GDEXTENSION_API_FILE "${CMAKE_CURRENT_BINARY_DIR}/extension_api.json")
    endif()

    message(STATUS "GODOTCPP_GDEXTENSION_API_FILE = '${GODOTCPP_GDEXTENSION_API_FILE}'")

    # generate the file list to use
    binding_generator_get_file_list( GENERATED_FILES_LIST
            "${GODOTCPP_GDEXTENSION_API_FILE}"
            "${CMAKE_CURRENT_BINARY_DIR}"
    )

    binding_generator_generate_bindings(
            "${GODOTCPP_GDEXTENSION_API_FILE}"
            "${USE_TEMPLATE_GET_NODE}"
            "${BITS}"
            "${GODOTCPP_PRECISION}"
            "${CMAKE_CURRENT_BINARY_DIR}"
    )

    ### Platform is derived from the toolchain target
    # See GeneratorExpressions PLATFORM_ID and CMAKE_SYSTEM_NAME
    string(
        CONCAT
        SYSTEM_NAME
        "$<$<PLATFORM_ID:Android>:android>"
        "$<$<PLATFORM_ID:iOS>:ios>"
        "$<$<PLATFORM_ID:Linux>:linux>"
        "$<$<PLATFORM_ID:Darwin>:macos>"
        "$<$<PLATFORM_ID:Emscripten>:web>"
        "$<$<PLATFORM_ID:Windows>:windows>"
        "$<$<PLATFORM_ID:Msys>:windows>"
    )

    # Process CPU architecture argument.
    godot_arch_name( ARCH_NAME )

    # Transform options into generator expressions
    set(HOT_RELOAD-UNSET "$<STREQUAL:${GODOTCPP_USE_HOT_RELOAD},>")

    set(DISABLE_EXCEPTIONS "$<BOOL:${GODOTCPP_DISABLE_EXCEPTIONS}>")

    set(THREADS_ENABLED "$<BOOL:${GODOTCPP_THREADS}>")

    # GODOTCPP_DEV_BUILD
    set(RELEASE_TYPES "Release;MinSizeRel")
    get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(IS_MULTI_CONFIG)
        message(NOTICE "=> Default build type is Debug. For other build types add --config <type> to build command")
    elseif(GODOTCPP_DEV_BUILD AND CMAKE_BUILD_TYPE IN_LIST RELEASE_TYPES)
        message(
            WARNING
            "=> GODOTCPP_DEV_BUILD implies a Debug-like build but CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}'"
        )
    endif()
    set(IS_DEV_BUILD "$<BOOL:${GODOTCPP_DEV_BUILD}>")

    ### Define our godot-cpp library targets
    # Generator Expressions that rely on the target
    set(DEBUG_FEATURES "$<NOT:$<STREQUAL:${GODOTCPP_TARGET},template_release>>")
    set(HOT_RELOAD "$<IF:${HOT_RELOAD-UNSET},${DEBUG_FEATURES},$<BOOL:${GODOTCPP_USE_HOT_RELOAD}>>")

    # Suffix
    string(
        CONCAT
        GODOTCPP_SUFFIX
        "$<1:.${SYSTEM_NAME}>"
        "$<1:.${GODOTCPP_TARGET}>"
        "$<${IS_DEV_BUILD}:.dev>"
        "$<$<STREQUAL:${GODOTCPP_PRECISION},double>:.double>"
        "$<1:.${ARCH_NAME}>"
        # TODO IOS_SIMULATOR
        "$<$<NOT:${THREADS_ENABLED}>:.nothreads>"
    )

    # the godot-cpp.* library targets
    add_library(godot-cpp STATIC)

    # Without adding this dependency to the binding generator, XCode will complain.
    add_dependencies(godot-cpp generate_bindings)

    # Added for backwards compatibility with prior cmake solution so that builds dont immediately break
    # from a missing target.
    add_library(godot::cpp ALIAS godot-cpp)

    file(GLOB_RECURSE GODOTCPP_SOURCES LIST_DIRECTORIES NO CONFIGURE_DEPENDS src/*.cpp)

    target_sources(godot-cpp PRIVATE ${GODOTCPP_SOURCES} ${GENERATED_FILES_LIST})

    target_include_directories(
        godot-cpp
        ${GODOTCPP_SYSTEM_HEADERS_ATTRIBUTE}
        PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/gen/include ${GODOTCPP_GDEXTENSION_DIR}
    )

    # gersemi: off
    set_target_properties(
        godot-cpp
        PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            CXX_VISIBILITY_PRESET ${GODOTCPP_SYMBOL_VISIBILITY}

            COMPILE_WARNING_AS_ERROR ${GODOTCPP_WARNING_AS_ERROR}
            POSITION_INDEPENDENT_CODE ON
            BUILD_RPATH_USE_ORIGIN ON

            PREFIX      "lib"
            OUTPUT_NAME "${PROJECT_NAME}${GODOTCPP_SUFFIX}"

            ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"

            # Things that are handy to know for dependent targets
            GODOTCPP_PLATFORM  "${SYSTEM_NAME}"
            GODOTCPP_TARGET    "${GODOTCPP_TARGET}"
            GODOTCPP_ARCH      "${ARCH_NAME}"
            GODOTCPP_PRECISION "${GODOTCPP_PRECISION}"
            GODOTCPP_SUFFIX    "${GODOTCPP_SUFFIX}"

            # Some IDE's respect this property to logically group targets
            FOLDER "godot-cpp"
    )
    # gersemi: on
    if(CMAKE_SYSTEM_NAME STREQUAL Android)
        android_generate()
    elseif(CMAKE_SYSTEM_NAME STREQUAL iOS)
        ios_generate()
    elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
        linux_generate()
    elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        macos_generate()
    elseif(CMAKE_SYSTEM_NAME STREQUAL Emscripten)
        web_generate()
    elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
        windows_generate()
    endif()
endfunction()
