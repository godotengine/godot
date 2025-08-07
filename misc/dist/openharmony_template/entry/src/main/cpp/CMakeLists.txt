# the minimum version of CMake.
cmake_minimum_required(VERSION 3.5.0)
project(template)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(entry SHARED napi_init.cpp)

add_library(godot SHARED IMPORTED)
set_target_properties(godot
	PROPERTIES
	IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${OHOS_ARCH}/libgodot.so
)

target_link_libraries(entry PUBLIC libhilog_ndk.z.so libace_ndk.z.so libnative_window.so libace_napi.z.so librawfile.z.so libnative_vsync.so libohaudio.so libohinputmethod.so libnative_display_manager.so libnative_window_manager.so libnative_drawing.so libudmf.so libpasteboard.so godot)
