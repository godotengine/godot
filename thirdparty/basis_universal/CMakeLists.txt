project(basisu)

cmake_minimum_required(VERSION 3.0)
option(BUILD_X64 "build 64-bit" TRUE)
option(STATIC "static linking" FALSE)

message("Initial BUILD_X64=${BUILD_X64}")
message("Initial CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif( NOT CMAKE_BUILD_TYPE )

message( ${PROJECT_NAME} " build type: " ${CMAKE_BUILD_TYPE} )

if (BUILD_X64)
	message("Building 64-bit")
else()
	message("Building 32-bit")
endif(BUILD_X64)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS -std=c++11)
set(GCC_COMPILE_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden -fPIC -fno-strict-aliasing -D_LARGEFILE64_SOURCE=1 -D_FILE_OFFSET_BITS=64 -Wall -Wextra -Wno-unused-local-typedefs -Wno-unused-value -Wno-unused-parameter -Wno-unused-variable -Wno-reorder")

if (NOT BUILD_X64)
	set(GCC_COMPILE_FLAGS "${GCC_COMPILE_FLAGS} -m32")
endif()

if (STATIC)
	set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_LINK_FLAGS} -static-libgcc -static-libstdc++ -static")
else()
	set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_LINK_FLAGS} -Wl,-rpath .")
endif(STATIC)

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} ${GCC_COMPILE_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${GCC_COMPILE_FLAGS} -D_DEBUG")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} ${GCC_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_COMPILE_FLAGS} -D_DEBUG")

set(BASISU_SRC_LIST ${COMMON_SRC_LIST} 
	basisu_backend.cpp
	basisu_basis_file.cpp
	basisu_comp.cpp
	basisu_enc.cpp
	basisu_etc.cpp
	basisu_frontend.cpp
	basisu_global_selector_palette_helpers.cpp
	basisu_gpu_texture.cpp
	basisu_pvrtc1_4.cpp
	basisu_resampler.cpp
	basisu_resample_filters.cpp
	basisu_ssim.cpp
	basisu_tool.cpp
	lodepng.cpp
	transcoder/basisu_transcoder.cpp
	)

if (APPLE)
	set(BIN_DIRECTORY "bin_osx")
else()
    set(BIN_DIRECTORY "bin")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_DIRECTORY})

add_executable(basisu ${BASISU_SRC_LIST})
target_link_libraries(basisu m pthread)

install(TARGETS basisu DESTINATION bin)

if (CMAKE_BUILD_TYPE STREQUAL Release)
    if (APPLE)
	    add_custom_command(TARGET basisu POST_BUILD COMMAND strip -X -x ${CMAKE_SOURCE_DIR}/${BIN_DIRECTORY}/basisu)
    else()
        add_custom_command(TARGET basisu POST_BUILD COMMAND strip -g -X -x ${CMAKE_SOURCE_DIR}/${BIN_DIRECTORY}/basisu)
    endif()
endif()

