cmake_minimum_required(VERSION 3.6.3...3.25 FATAL_ERROR)

add_library(hidapi_hidraw
    ${HIDAPI_PUBLIC_HEADERS}
    hid.c
)
target_link_libraries(hidapi_hidraw PUBLIC hidapi_include)

find_package(Threads REQUIRED)

include(FindPkgConfig)
pkg_check_modules(libudev REQUIRED IMPORTED_TARGET libudev)

target_link_libraries(hidapi_hidraw PRIVATE PkgConfig::libudev Threads::Threads)

set_target_properties(hidapi_hidraw
    PROPERTIES
        EXPORT_NAME "hidraw"
        OUTPUT_NAME "hidapi-hidraw"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${HIDAPI_PUBLIC_HEADERS}"
)

# compatibility with find_package()
add_library(hidapi::hidraw ALIAS hidapi_hidraw)
# compatibility with raw library link
add_library(hidapi-hidraw ALIAS hidapi_hidraw)

if(HIDAPI_INSTALL_TARGETS)
    install(TARGETS hidapi_hidraw EXPORT hidapi
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hidapi"
    )
endif()

hidapi_configure_pc("${PROJECT_ROOT}/pc/hidapi-hidraw.pc.in")
