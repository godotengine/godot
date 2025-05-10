cmake_minimum_required(VERSION 3.6.3...3.25 FATAL_ERROR)

list(APPEND HIDAPI_PUBLIC_HEADERS "hidapi_libusb.h")

add_library(hidapi_libusb
    ${HIDAPI_PUBLIC_HEADERS}
    hid.c
)
target_link_libraries(hidapi_libusb PUBLIC hidapi_include)

if(TARGET usb-1.0)
    target_link_libraries(hidapi_libusb PRIVATE usb-1.0)
else()
    include(FindPkgConfig)
    pkg_check_modules(libusb REQUIRED IMPORTED_TARGET libusb-1.0>=1.0.9)
    target_link_libraries(hidapi_libusb PRIVATE PkgConfig::libusb)
endif()

find_package(Threads REQUIRED)
target_link_libraries(hidapi_libusb PRIVATE Threads::Threads)

if(HIDAPI_NO_ICONV)
    target_compile_definitions(hidapi_libusb PRIVATE NO_ICONV)
else()
    if(NOT ANDROID)
        include(CheckCSourceCompiles)

        if(NOT CMAKE_VERSION VERSION_LESS 3.11)
            message(STATUS "Check for Iconv")
            find_package(Iconv)
            if(Iconv_FOUND)
                if(NOT Iconv_IS_BUILT_IN)
                    target_link_libraries(hidapi_libusb PRIVATE Iconv::Iconv)
                    set(CMAKE_REQUIRED_LIBRARIES "Iconv::Iconv")
                    if(NOT BUILD_SHARED_LIBS)
                        set(HIDAPI_NEED_EXPORT_ICONV TRUE PARENT_SCOPE)
                    endif()
                endif()
            else()
                message(STATUS "Iconv Explicitly check '-liconv'")
                # Sometime the build environment is not setup
                # in a way CMake can find Iconv on its own by default.
                # But if we simply link against iconv (-liconv), the build may succeed
                # due to other compiler/link flags.
                set(CMAKE_REQUIRED_LIBRARIES "iconv")
                check_c_source_compiles("
                    #include <stddef.h>
                    #include <iconv.h>
                    int main() {
                        char *a, *b;
                        size_t i, j;
                        iconv_t ic;
                        ic = iconv_open(\"to\", \"from\");
                        iconv(ic, &a, &i, &b, &j);
                        iconv_close(ic);
                    }
                    "
                    Iconv_EXPLICITLY_AT_ENV)
                if(Iconv_EXPLICITLY_AT_ENV)
                    message(STATUS "Iconv Explicitly check '-liconv' - Available")
                    target_link_libraries(hidapi_libusb PRIVATE iconv)
                else()
                    message(FATAL_ERROR "Iconv is not found, make sure to provide it in the build environment")
                endif()
            endif()
        else()
            # otherwise there is 2 options:
            # 1) iconv is provided by Standard C library and the build will be just fine
            # 2) The _user_ has to provide additional compilation options for this project/target
        endif()

        # check for error: "conflicting types for 'iconv'"
        check_c_source_compiles("#include<iconv.h>
            extern size_t iconv (iconv_t cd, const char **inbuf, size_t *inbytesleft, char **outbuf, size_t *outbytesleft);
            int main() {}"
        HIDAPI_ICONV_CONST)
        if(HIDAPI_ICONV_CONST)
            target_compile_definitions(hidapi_libusb PRIVATE "ICONV_CONST=const")
        endif()
    else()
        # On Android Iconv is disabled on the code level anyway, so no issue;
    endif()
endif()

set_target_properties(hidapi_libusb
    PROPERTIES
        EXPORT_NAME "libusb"
        OUTPUT_NAME "hidapi-libusb"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${HIDAPI_PUBLIC_HEADERS}"
)

# compatibility with find_package()
add_library(hidapi::libusb ALIAS hidapi_libusb)
# compatibility with raw library link
add_library(hidapi-libusb ALIAS hidapi_libusb)

if(HIDAPI_INSTALL_TARGETS)
    install(TARGETS hidapi_libusb EXPORT hidapi
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hidapi"
    )
endif()

hidapi_configure_pc("${PROJECT_ROOT}/pc/hidapi-libusb.pc.in")
