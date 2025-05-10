get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Read version from file
file(READ "${PROJECT_ROOT}/VERSION" RAW_VERSION_STR)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)(.*)" VERSION_STR "${RAW_VERSION_STR}")

if(NOT VERSION_STR)
    message(FATAL_ERROR "Broken VERSION file, couldn't parse '${PROJECT_ROOT}/VERSION' with content: '${RAW_VERSION_STR}'")
endif()

set(VERSION "${CMAKE_MATCH_1}")
string(STRIP "${CMAKE_MATCH_2}" VERSION_SUFFIX)
# compatibility with find_package() vs add_subdirectory
set(hidapi_VERSION "${VERSION}" PARENT_SCOPE)
#

if(DEFINED HIDAPI_PRINT_VERSION AND HIDAPI_PRINT_VERSION)
    set(HIDAPI_PRINT_VERSION "hidapi: v${VERSION}")
    if(VERSION_SUFFIX)
        set(HIDAPI_PRINT_VERSION "${HIDAPI_PRINT_VERSION} (${VERSION_SUFFIX})")
    endif()
    message(STATUS "${HIDAPI_PRINT_VERSION}")
endif()

project(hidapi VERSION "${VERSION}" LANGUAGES C)

# Defaults and required options

if(NOT DEFINED HIDAPI_WITH_TESTS)
    set(HIDAPI_WITH_TESTS OFF)
endif()
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()
if(NOT DEFINED HIDAPI_INSTALL_TARGETS)
    set(HIDAPI_INSTALL_TARGETS OFF)
endif()
if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

get_directory_property(IS_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
if(IS_EXCLUDE_FROM_ALL)
    if(HIDAPI_INSTALL_TARGETS)
        message(WARNING "Installing EXCLUDE_FROM_ALL targets in an undefined behavior in CMake.\nDon't add 'hidapi' sundirectory with 'EXCLUDE_FROM_ALL' property, or don't set 'HIDAPI_INSTALL_TARGETS' to TRUE.")
    endif()
endif()

# Helper(s)

function(hidapi_configure_pc PC_IN_FILE)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pc")

    set(VERSION "${VERSION}${VERSION_SUFFIX}")
    set(prefix "${CMAKE_INSTALL_PREFIX}")
    set(exec_prefix "\${prefix}")
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(libdir "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(libdir "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(includedir "${CMAKE_INSTALL_INCLUDEDIR}")
    else()
        set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif()

    get_filename_component(PC_IN_FILENAME "${PC_IN_FILE}" NAME_WE)
    set(PC_FILE "${CMAKE_CURRENT_BINARY_DIR}/pc/${PC_IN_FILENAME}.pc")
    configure_file("${PC_IN_FILE}" "${PC_FILE}" @ONLY)
    if(HIDAPI_INSTALL_TARGETS)
        install(FILES "${PC_FILE}" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
    endif()
endfunction()

# The library

if(HIDAPI_INSTALL_TARGETS)
    include(GNUInstallDirs)
endif()

add_library(hidapi_include INTERFACE)
target_include_directories(hidapi_include INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_ROOT}/hidapi>"
)
if(APPLE AND CMAKE_FRAMEWORK)
    # FIXME: https://github.com/libusb/hidapi/issues/492: it is untrivial to set the include path for Framework correctly
else()
    target_include_directories(hidapi_include INTERFACE
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hidapi>"
    )
endif()
set_target_properties(hidapi_include PROPERTIES EXPORT_NAME "include")
set(HIDAPI_PUBLIC_HEADERS "${PROJECT_ROOT}/hidapi/hidapi.h")

add_library(hidapi::include ALIAS hidapi_include)

if(HIDAPI_INSTALL_TARGETS)
    install(TARGETS hidapi_include EXPORT hidapi)
endif()

set(EXPORT_ALIAS)
set(EXPORT_COMPONENTS)

set(HIDAPI_NEED_EXPORT_THREADS FALSE)
set(HIDAPI_NEED_EXPORT_LIBUSB FALSE)
set(HIDAPI_NEED_EXPORT_LIBUDEV FALSE)
set(HIDAPI_NEED_EXPORT_ICONV FALSE)

if(WIN32)
    target_include_directories(hidapi_include INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/windows>"
    )
    add_subdirectory("${PROJECT_ROOT}/windows" windows)
    set(EXPORT_ALIAS winapi)
    list(APPEND EXPORT_COMPONENTS winapi)
elseif(APPLE)
    target_include_directories(hidapi_include INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/mac>"
    )
    add_subdirectory("${PROJECT_ROOT}/mac" mac)
    set(EXPORT_ALIAS darwin)
    list(APPEND EXPORT_COMPONENTS darwin)
    if(NOT BUILD_SHARED_LIBS)
        set(HIDAPI_NEED_EXPORT_THREADS TRUE)
    endif()
else()
    if(NOT DEFINED HIDAPI_WITH_LIBUSB)
        set(HIDAPI_WITH_LIBUSB ON)
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        if(NOT DEFINED HIDAPI_WITH_HIDRAW)
            set(HIDAPI_WITH_HIDRAW ON)
        endif()
        if(HIDAPI_WITH_HIDRAW)
            add_subdirectory("${PROJECT_ROOT}/linux" linux)
            list(APPEND EXPORT_COMPONENTS hidraw)
            set(EXPORT_ALIAS hidraw)
            if(NOT BUILD_SHARED_LIBS)
                set(HIDAPI_NEED_EXPORT_THREADS TRUE)
                set(HIDAPI_NEED_EXPORT_LIBUDEV TRUE)
            endif()
        endif()
    elseif(CMAKE_SYSTEM_NAME MATCHES "NetBSD")
        if(NOT DEFINED HIDAPI_WITH_NETBSD)
            set(HIDAPI_WITH_NETBSD ON)
        endif()
        if(HIDAPI_WITH_NETBSD)
            add_subdirectory("${PROJECT_ROOT}/netbsd" netbsd)
            list(APPEND EXPORT_COMPONENTS netbsd)
            set(EXPORT_ALIAS netbsd)
            if(NOT BUILD_SHARED_LIBS)
                set(HIDAPI_NEED_EXPORT_THREADS TRUE)
            endif()
        endif()
    else()
        set(HIDAPI_WITH_LIBUSB ON)
    endif()
    if(HIDAPI_WITH_LIBUSB)
        target_include_directories(hidapi_include INTERFACE
            "$<BUILD_INTERFACE:${PROJECT_ROOT}/libusb>"
        )
        if(NOT DEFINED HIDAPI_NO_ICONV)
            set(HIDAPI_NO_ICONV OFF)
        endif()
        add_subdirectory("${PROJECT_ROOT}/libusb" libusb)
        list(APPEND EXPORT_COMPONENTS libusb)
        if(NOT EXPORT_ALIAS)
            set(EXPORT_ALIAS libusb)
        endif()
        if(NOT BUILD_SHARED_LIBS)
            set(HIDAPI_NEED_EXPORT_THREADS TRUE)
            if(NOT TARGET usb-1.0)
                set(HIDAPI_NEED_EXPORT_LIBUSB TRUE)
            endif()
        endif()
    elseif(NOT TARGET hidapi_hidraw)
        message(FATAL_ERROR "Select at least one option to build: HIDAPI_WITH_LIBUSB or HIDAPI_WITH_HIDRAW")
    endif()
endif()

add_library(hidapi::hidapi ALIAS hidapi_${EXPORT_ALIAS})

if(HIDAPI_INSTALL_TARGETS)
    include(CMakePackageConfigHelpers)
    set(EXPORT_DENERATED_LOCATION "${CMAKE_BINARY_DIR}/export_generated")
    set(EXPORT_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hidapi")
    write_basic_package_version_file("${EXPORT_DENERATED_LOCATION}/hidapi-config-version.cmake"
        COMPATIBILITY SameMajorVersion
    )
    configure_package_config_file("cmake/hidapi-config.cmake.in" "${EXPORT_DENERATED_LOCATION}/hidapi-config.cmake"
        INSTALL_DESTINATION "${EXPORT_DESTINATION}"
        NO_SET_AND_CHECK_MACRO
    )

    install(EXPORT hidapi
        DESTINATION "${EXPORT_DESTINATION}"
        NAMESPACE hidapi::
        FILE "libhidapi.cmake"
    )
    install(FILES
            "${EXPORT_DENERATED_LOCATION}/hidapi-config-version.cmake"
            "${EXPORT_DENERATED_LOCATION}/hidapi-config.cmake"
        DESTINATION "${EXPORT_DESTINATION}"
    )
endif()
