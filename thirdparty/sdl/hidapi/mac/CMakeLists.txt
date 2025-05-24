cmake_minimum_required(VERSION 3.4.3...3.25 FATAL_ERROR)

list(APPEND HIDAPI_PUBLIC_HEADERS "hidapi_darwin.h")

add_library(hidapi_darwin
    ${HIDAPI_PUBLIC_HEADERS}
    hid.c
)

find_package(Threads REQUIRED)

target_link_libraries(hidapi_darwin
    PUBLIC hidapi_include
    PRIVATE Threads::Threads
    PRIVATE "-framework IOKit" "-framework CoreFoundation"
)

set_target_properties(hidapi_darwin
    PROPERTIES
        EXPORT_NAME "darwin"
        OUTPUT_NAME "hidapi"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        MACHO_COMPATIBILITY_VERSION ${PROJECT_VERSION_MAJOR}
        FRAMEWORK_VERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${HIDAPI_PUBLIC_HEADERS}"
)

# compatibility with find_package()
add_library(hidapi::darwin ALIAS hidapi_darwin)
# compatibility with raw library link
add_library(hidapi ALIAS hidapi_darwin)

set(PUBLIC_HEADER_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
if(NOT CMAKE_FRAMEWORK)
    set(PUBLIC_HEADER_DESTINATION "${PUBLIC_HEADER_DESTINATION}/hidapi")
endif()

if(HIDAPI_INSTALL_TARGETS)
    install(TARGETS hidapi_darwin EXPORT hidapi
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        FRAMEWORK DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${PUBLIC_HEADER_DESTINATION}"
    )
endif()

hidapi_configure_pc("${PROJECT_ROOT}/pc/hidapi.pc.in")
