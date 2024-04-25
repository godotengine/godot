cmake_minimum_required(VERSION 3.1)

project(GLFW VERSION 3.4.0 LANGUAGES C)

set(CMAKE_LEGACY_CYGWIN_WIN32 OFF)

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(GLFW_STANDALONE TRUE)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" ${GLFW_STANDALONE})
option(GLFW_BUILD_TESTS "Build the GLFW test programs" ${GLFW_STANDALONE})
option(GLFW_BUILD_DOCS "Build the GLFW documentation" ON)
option(GLFW_INSTALL "Generate installation target" ON)
option(GLFW_VULKAN_STATIC "Assume the Vulkan loader is linked with the application" OFF)

include(GNUInstallDirs)
include(CMakeDependentOption)

cmake_dependent_option(GLFW_USE_OSMESA "Use OSMesa for offscreen context creation" OFF
                       "UNIX" OFF)
cmake_dependent_option(GLFW_USE_HYBRID_HPG "Force use of high-performance GPU on hybrid systems" OFF
                       "WIN32" OFF)
cmake_dependent_option(GLFW_USE_WAYLAND "Use Wayland for window creation" OFF
                       "UNIX;NOT APPLE" OFF)
cmake_dependent_option(USE_MSVC_RUNTIME_LIBRARY_DLL "Use MSVC runtime library DLL" ON
                       "MSVC" OFF)

if (BUILD_SHARED_LIBS AND UNIX)
    # On Unix-like systems, shared libraries can use the soname system.
    set(GLFW_LIB_NAME glfw)
else()
    set(GLFW_LIB_NAME glfw3_minimal)
endif()

if (GLFW_VULKAN_STATIC)
    if (BUILD_SHARED_LIBS)
        # If you absolutely must do this, remove this line and add the Vulkan
        # loader static library via the CMAKE_SHARED_LINKER_FLAGS
        message(FATAL_ERROR "You are trying to link the Vulkan loader static library into the GLFW shared library")
    endif()
    set(_GLFW_VULKAN_STATIC 1)
endif()

list(APPEND CMAKE_MODULE_PATH "${GLFW_SOURCE_DIR}/CMake/modules")

find_package(Threads REQUIRED)

if (GLFW_BUILD_DOCS)
    set(DOXYGEN_SKIP_DOT TRUE)
    find_package(Doxygen)
endif()

#--------------------------------------------------------------------
# Set compiler specific flags
#--------------------------------------------------------------------
if (MSVC)
    if (MSVC90)
        # Workaround for VS 2008 not shipping with the DirectX 9 SDK
        include(CheckIncludeFile)
        check_include_file(dinput.h DINPUT_H_FOUND)
        if (NOT DINPUT_H_FOUND)
            message(FATAL_ERROR "DirectX 9 headers not found; install DirectX 9 SDK")
        endif()
        # Workaround for VS 2008 not shipping with stdint.h
        list(APPEND glfw_INCLUDE_DIRS "${GLFW_SOURCE_DIR}/deps/vs2008")
    endif()

    if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
        foreach (flag CMAKE_C_FLAGS
                      CMAKE_C_FLAGS_DEBUG
                      CMAKE_C_FLAGS_RELEASE
                      CMAKE_C_FLAGS_MINSIZEREL
                      CMAKE_C_FLAGS_RELWITHDEBINFO)

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()

        endforeach()
    endif()
endif()

if (MINGW)
    # Workaround for legacy MinGW not providing XInput and DirectInput
    include(CheckIncludeFile)

    check_include_file(dinput.h DINPUT_H_FOUND)
    check_include_file(xinput.h XINPUT_H_FOUND)
    if (NOT DINPUT_H_FOUND OR NOT XINPUT_H_FOUND)
        list(APPEND glfw_INCLUDE_DIRS "${GLFW_SOURCE_DIR}/deps/mingw")
    endif()

    # Enable link-time exploit mitigation features enabled by default on MSVC
    include(CheckCCompilerFlag)

    # Compatibility with data execution prevention (DEP)
    set(CMAKE_REQUIRED_FLAGS "-Wl,--nxcompat")
    check_c_compiler_flag("" _GLFW_HAS_DEP)
    if (_GLFW_HAS_DEP)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--nxcompat ${CMAKE_SHARED_LINKER_FLAGS}")
    endif()

    # Compatibility with address space layout randomization (ASLR)
    set(CMAKE_REQUIRED_FLAGS "-Wl,--dynamicbase")
    check_c_compiler_flag("" _GLFW_HAS_ASLR)
    if (_GLFW_HAS_ASLR)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--dynamicbase ${CMAKE_SHARED_LINKER_FLAGS}")
    endif()

    # Compatibility with 64-bit address space layout randomization (ASLR)
    set(CMAKE_REQUIRED_FLAGS "-Wl,--high-entropy-va")
    check_c_compiler_flag("" _GLFW_HAS_64ASLR)
    if (_GLFW_HAS_64ASLR)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--high-entropy-va ${CMAKE_SHARED_LINKER_FLAGS}")
    endif()
endif()

#--------------------------------------------------------------------
# Detect and select backend APIs
#--------------------------------------------------------------------
if (GLFW_USE_WAYLAND)
    set(_GLFW_WAYLAND 1)
    message(STATUS "Using Wayland for window creation")
elseif (GLFW_USE_OSMESA)
    set(_GLFW_OSMESA 1)
    message(STATUS "Using OSMesa for headless context creation")
elseif (WIN32)
    set(_GLFW_WIN32 1)
    message(STATUS "Using Win32 for window creation")
elseif (APPLE)
    set(_GLFW_COCOA 1)
    message(STATUS "Using Cocoa for window creation")
elseif (UNIX)
    set(_GLFW_X11 1)
    message(STATUS "Using X11 for window creation")
else()
    message(FATAL_ERROR "No supported platform was detected")
endif()

#--------------------------------------------------------------------
# Find and add Unix math and time libraries
#--------------------------------------------------------------------
if (UNIX AND NOT APPLE)
    find_library(RT_LIBRARY rt)
    mark_as_advanced(RT_LIBRARY)
    if (RT_LIBRARY)
        list(APPEND glfw_LIBRARIES "${RT_LIBRARY}")
        list(APPEND glfw_PKG_LIBS "-lrt")
    endif()

    find_library(MATH_LIBRARY m)
    mark_as_advanced(MATH_LIBRARY)
    if (MATH_LIBRARY)
        list(APPEND glfw_LIBRARIES "${MATH_LIBRARY}")
        list(APPEND glfw_PKG_LIBS "-lm")
    endif()

    if (CMAKE_DL_LIBS)
        list(APPEND glfw_LIBRARIES "${CMAKE_DL_LIBS}")
        list(APPEND glfw_PKG_LIBS "-l${CMAKE_DL_LIBS}")
    endif()
endif()

#--------------------------------------------------------------------
# Use Win32 for window creation
#--------------------------------------------------------------------
if (_GLFW_WIN32)

    list(APPEND glfw_PKG_LIBS "-lgdi32")

    if (GLFW_USE_HYBRID_HPG)
        set(_GLFW_USE_HYBRID_HPG 1)
    endif()
endif()

#--------------------------------------------------------------------
# Use X11 for window creation
#--------------------------------------------------------------------
if (_GLFW_X11)

    find_package(X11 REQUIRED)

    list(APPEND glfw_PKG_DEPS "x11")

    # Set up library and include paths
    list(APPEND glfw_INCLUDE_DIRS "${X11_X11_INCLUDE_PATH}")
    list(APPEND glfw_LIBRARIES "${X11_X11_LIB}")

    # Check for XRandR (modern resolution switching and gamma control)
    if (NOT X11_Xrandr_INCLUDE_PATH)
        message(FATAL_ERROR "RandR headers not found; install libxrandr development package")
    endif()

    # Check for Xinerama (legacy multi-monitor support)
    if (NOT X11_Xinerama_INCLUDE_PATH)
        message(FATAL_ERROR "Xinerama headers not found; install libxinerama development package")
    endif()

    # Check for Xkb (X keyboard extension)
    if (NOT X11_Xkb_INCLUDE_PATH)
        message(FATAL_ERROR "XKB headers not found; install X11 development package")
    endif()

    # Check for Xcursor (cursor creation from RGBA images)
    if (NOT X11_Xcursor_INCLUDE_PATH)
        message(FATAL_ERROR "Xcursor headers not found; install libxcursor development package")
    endif()

    # Check for XInput (modern HID input)
    if (NOT X11_Xi_INCLUDE_PATH)
        message(FATAL_ERROR "XInput headers not found; install libxi development package")
    endif()
endif()

#--------------------------------------------------------------------
# Use Wayland for window creation
#--------------------------------------------------------------------
if (_GLFW_WAYLAND)
    find_package(ECM REQUIRED NO_MODULE)
    list(APPEND CMAKE_MODULE_PATH "${ECM_MODULE_PATH}")

    find_package(Wayland REQUIRED Client Cursor Egl)
    find_package(WaylandScanner REQUIRED)
    find_package(WaylandProtocols 1.15 REQUIRED)

    list(APPEND glfw_PKG_DEPS "wayland-egl")

    list(APPEND glfw_INCLUDE_DIRS "${Wayland_INCLUDE_DIRS}")
    list(APPEND glfw_LIBRARIES "${Wayland_LIBRARIES}")

    find_package(XKBCommon REQUIRED)
    list(APPEND glfw_INCLUDE_DIRS "${XKBCOMMON_INCLUDE_DIRS}")

    include(CheckIncludeFiles)
    include(CheckFunctionExists)
    check_include_files(xkbcommon/xkbcommon-compose.h HAVE_XKBCOMMON_COMPOSE_H)
    check_function_exists(memfd_create HAVE_MEMFD_CREATE)

    if (NOT ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux"))
        find_package(EpollShim)
        if (EPOLLSHIM_FOUND)
            list(APPEND glfw_INCLUDE_DIRS "${EPOLLSHIM_INCLUDE_DIRS}")
            list(APPEND glfw_LIBRARIES "${EPOLLSHIM_LIBRARIES}")
        endif()
    endif()
endif()

#--------------------------------------------------------------------
# Use OSMesa for offscreen context creation
#--------------------------------------------------------------------
if (_GLFW_OSMESA)
    find_package(OSMesa REQUIRED)
endif()

#--------------------------------------------------------------------
# Use Cocoa for window creation and NSOpenGL for context creation
#--------------------------------------------------------------------
if (_GLFW_COCOA)

    list(APPEND glfw_LIBRARIES
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation")

    set(glfw_PKG_DEPS "")
    set(glfw_PKG_LIBS "-framework Cocoa -framework IOKit -framework CoreFoundation")
endif()

#--------------------------------------------------------------------
# Add the Vulkan loader as a dependency if necessary
#--------------------------------------------------------------------
if (GLFW_VULKAN_STATIC)
    list(APPEND glfw_PKG_DEPS "vulkan")
endif()

#--------------------------------------------------------------------
# Export GLFW library dependencies
#--------------------------------------------------------------------
foreach(arg ${glfw_PKG_DEPS})
    set(GLFW_PKG_DEPS "${GLFW_PKG_DEPS} ${arg}")
endforeach()
foreach(arg ${glfw_PKG_LIBS})
    set(GLFW_PKG_LIBS "${GLFW_PKG_LIBS} ${arg}")
endforeach()

#--------------------------------------------------------------------
# Create generated files
#--------------------------------------------------------------------
include(CMakePackageConfigHelpers)

set(GLFW_CONFIG_PATH "${CMAKE_INSTALL_LIBDIR}/cmake/glfw3")

configure_package_config_file(src/glfw3Config.cmake.in
                              src/glfw3Config.cmake
                              INSTALL_DESTINATION "${GLFW_CONFIG_PATH}"
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(src/glfw3ConfigVersion.cmake
                                 VERSION ${GLFW_VERSION}
                                 COMPATIBILITY SameMajorVersion)

configure_file(src/glfw3.pc.in src/glfw3.pc @ONLY)

#--------------------------------------------------------------------
# Add subdirectories
#--------------------------------------------------------------------
add_subdirectory(src)

if (GLFW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (GLFW_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (DOXYGEN_FOUND AND GLFW_BUILD_DOCS)
    add_subdirectory(docs)
endif()

#--------------------------------------------------------------------
# Install files other than the library
# The library is installed by src/CMakeLists.txt
#--------------------------------------------------------------------
if (GLFW_INSTALL)
    install(DIRECTORY include/GLFW DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN glfw3.h PATTERN glfw3native.h)

    install(FILES "${GLFW_BINARY_DIR}/src/glfw3Config.cmake"
                  "${GLFW_BINARY_DIR}/src/glfw3ConfigVersion.cmake"
            DESTINATION "${GLFW_CONFIG_PATH}")

    install(EXPORT glfwTargets FILE glfw3Targets.cmake
            EXPORT_LINK_INTERFACE_LIBRARIES
            DESTINATION "${GLFW_CONFIG_PATH}")
    install(FILES "${GLFW_BINARY_DIR}/src/glfw3.pc"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

    if (DOXYGEN_FOUND AND GLFW_BUILD_DOCS)
        install(DIRECTORY "${GLFW_BINARY_DIR}/docs/html"
                DESTINATION "${CMAKE_INSTALL_DOCDIR}")
    endif()

    # Only generate this target if no higher-level project already has
    if (NOT TARGET uninstall)
        configure_file(CMake/cmake_uninstall.cmake.in
                       cmake_uninstall.cmake IMMEDIATE @ONLY)

        add_custom_target(uninstall
                          "${CMAKE_COMMAND}" -P
                          "${GLFW_BINARY_DIR}/cmake_uninstall.cmake")
        set_target_properties(uninstall PROPERTIES FOLDER "GLFW3")
    endif()
endif()

