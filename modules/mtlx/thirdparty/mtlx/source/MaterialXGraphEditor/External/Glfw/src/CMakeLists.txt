
add_library(glfw_minimal "${GLFW_SOURCE_DIR}/include/GLFW/glfw3.h"
                 "${GLFW_SOURCE_DIR}/include/GLFW/glfw3native.h"
                 internal.h mappings.h context.c init.c input.c monitor.c
                 vulkan.c window.c)

if (_GLFW_COCOA)
    target_sources(glfw_minimal PRIVATE cocoa_platform.h cocoa_joystick.h posix_thread.h
                                nsgl_context.h egl_context.h osmesa_context.h
                                cocoa_init.m cocoa_joystick.m cocoa_monitor.m
                                cocoa_window.m cocoa_time.c posix_thread.c
                                nsgl_context.m egl_context.c osmesa_context.c)
elseif (_GLFW_WIN32)
    target_sources(glfw_minimal PRIVATE win32_platform.h win32_joystick.h wgl_context.h
                                egl_context.h osmesa_context.h win32_init.c
                                win32_joystick.c win32_monitor.c win32_time.c
                                win32_thread.c win32_window.c wgl_context.c
                                egl_context.c osmesa_context.c)
elseif (_GLFW_X11)
    target_sources(glfw_minimal PRIVATE x11_platform.h xkb_unicode.h posix_time.h
                                posix_thread.h glx_context.h egl_context.h
                                osmesa_context.h x11_init.c x11_monitor.c
                                x11_window.c xkb_unicode.c posix_time.c
                                posix_thread.c glx_context.c egl_context.c
                                osmesa_context.c)
elseif (_GLFW_WAYLAND)
    target_sources(glfw_minimal PRIVATE wl_platform.h posix_time.h posix_thread.h
                                xkb_unicode.h egl_context.h osmesa_context.h
                                wl_init.c wl_monitor.c wl_window.c posix_time.c
                                posix_thread.c xkb_unicode.c egl_context.c
                                osmesa_context.c)
elseif (_GLFW_OSMESA)
    target_sources(glfw_minimal PRIVATE null_platform.h null_joystick.h posix_time.h
                                posix_thread.h osmesa_context.h null_init.c
                                null_monitor.c null_window.c null_joystick.c
                                posix_time.c posix_thread.c osmesa_context.c)
endif()

if (_GLFW_X11 OR _GLFW_WAYLAND)
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        target_sources(glfw_minimal PRIVATE linux_joystick.h linux_joystick.c)
    else()
        target_sources(glfw_minimal PRIVATE null_joystick.h null_joystick.c)
    endif()
endif()

if (_GLFW_WAYLAND)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/stable/xdg-shell/xdg-shell.xml"
        BASENAME xdg-shell)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/unstable/xdg-decoration/xdg-decoration-unstable-v1.xml"
        BASENAME xdg-decoration)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/stable/viewporter/viewporter.xml"
        BASENAME viewporter)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/unstable/relative-pointer/relative-pointer-unstable-v1.xml"
        BASENAME relative-pointer-unstable-v1)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/unstable/pointer-constraints/pointer-constraints-unstable-v1.xml"
        BASENAME pointer-constraints-unstable-v1)
    ecm_add_wayland_client_protocol(GLFW_WAYLAND_PROTOCOL_SOURCES
        PROTOCOL
        "${WAYLAND_PROTOCOLS_PKGDATADIR}/unstable/idle-inhibit/idle-inhibit-unstable-v1.xml"
        BASENAME idle-inhibit-unstable-v1)
    target_sources(glfw_minimal PRIVATE ${GLFW_WAYLAND_PROTOCOL_SOURCES})
endif()

if (WIN32 AND BUILD_SHARED_LIBS)
    configure_file(glfw.rc.in glfw.rc @ONLY)
    target_sources(glfw_minimal PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/glfw.rc")
endif()

configure_file(glfw_config.h.in glfw_config.h @ONLY)
target_compile_definitions(glfw_minimal PRIVATE _GLFW_USE_CONFIG_H)
target_sources(glfw_minimal PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/glfw_config.h")

set_target_properties(glfw_minimal PROPERTIES
                      OUTPUT_NAME ${GLFW_LIB_NAME}
                      VERSION ${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}
                      SOVERSION ${GLFW_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
                      C_STANDARD 99
                      C_EXTENSIONS OFF
                      DEFINE_SYMBOL _GLFW_BUILD_DLL
                      FOLDER "GLFW3")

target_include_directories(glfw_minimal PUBLIC
                           "$<BUILD_INTERFACE:${GLFW_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_include_directories(glfw_minimal PRIVATE
                           "${GLFW_SOURCE_DIR}/src"
                           "${GLFW_BINARY_DIR}/src"
                           ${glfw_INCLUDE_DIRS})
target_link_libraries(glfw_minimal PRIVATE Threads::Threads ${glfw_LIBRARIES})

if (APPLE)
    # For some reason CMake didn't know about .m until version 3.16
    set_source_files_properties(cocoa_init.m cocoa_joystick.m cocoa_monitor.m
                                cocoa_window.m nsgl_context.m PROPERTIES
                                LANGUAGE C)
endif()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")

    # Make GCC and Clang warn about declarations that VS 2010 and 2012 won't
    # accept for all source files that VS will build
    set_source_files_properties(context.c init.c input.c monitor.c vulkan.c
                                window.c win32_init.c win32_joystick.c
                                win32_monitor.c win32_time.c win32_thread.c
                                win32_window.c wgl_context.c egl_context.c
                                osmesa_context.c PROPERTIES
                                COMPILE_FLAGS -Wdeclaration-after-statement)

    # Enable a reasonable set of warnings (no, -Wextra is not reasonable)
    target_compile_options(glfw_minimal PRIVATE "-Wall")
endif()

if (WIN32)
    target_compile_definitions(glfw_minimal PRIVATE _UNICODE)
endif()

# HACK: When building on MinGW, WINVER and UNICODE need to be defined before
# the inclusion of stddef.h (by glfw3.h), which is itself included before
# win32_platform.h.  We define them here until a saner solution can be found
# NOTE: MinGW-w64 and Visual C++ do /not/ need this hack.
if (MINGW)
    target_compile_definitions(glfw_minimal PRIVATE UNICODE WINVER=0x0501)
endif()

if (BUILD_SHARED_LIBS)
    if (WIN32)
        if (MINGW)
            # Remove the dependency on the shared version of libgcc
            # NOTE: MinGW-w64 has the correct default but MinGW needs this
            target_link_libraries(glfw_minimal PRIVATE "-static-libgcc")

            # Remove the lib prefix on the DLL (but not the import library)
            set_target_properties(glfw_minimal PROPERTIES PREFIX "")

            # Add a suffix to the import library to avoid naming conflicts
            set_target_properties(glfw_minimal PROPERTIES IMPORT_SUFFIX "dll.a")
        else()
            # Add a suffix to the import library to avoid naming conflicts
            set_target_properties(glfw_minimal PROPERTIES IMPORT_SUFFIX "dll.lib")
        endif()

        target_compile_definitions(glfw_minimal INTERFACE GLFW_DLL)
    elseif (APPLE)
        set_target_properties(glfw_minimal PROPERTIES
                              INSTALL_NAME_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    if (UNIX)
        # Hide symbols not explicitly tagged for export from the shared library
        target_compile_options(glfw_minimal PRIVATE "-fvisibility=hidden")
    endif()
endif()

if (MSVC)
    target_compile_definitions(glfw_minimal PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if (GLFW_INSTALL)
    install(TARGETS glfw_minimal
            EXPORT glfwTargets
            RUNTIME DESTINATION "bin"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

