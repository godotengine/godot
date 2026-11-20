cmake_minimum_required(VERSION 4.0)

project(simplewebm VERSION 0.0.2 LANGUAGES CXX)

option(LIBSIMPLEWEBM_BUILD_EXAMPLES "Build the simplewebm example" OFF)
option(LIBSIMPLEWEBM_BUILD_PLAYER "Build the SDL3-based simplewebm player" OFF)

set(WEBM_PARSER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libwebm)

set(WEBM_PARSER_SOURCES
    "${WEBM_PARSER_DIR}/webm_parser/src/ancestory.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/bit_utils.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_header_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/buffer_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/callback.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/date_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/file_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/float_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_element_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/istream_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/master_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/parser_utils.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/segment_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/size_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/skip_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/unknown_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/var_int_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/virtual_block_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/void_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/webm_parser.cc"
)

add_library(webm_parser STATIC ${WEBM_PARSER_SOURCES})

target_include_directories(webm_parser PUBLIC ${WEBM_PARSER_DIR}/webm_parser/include PRIVATE ${WEBM_PARSER_DIR}/webm_parser)

target_compile_features(webm_parser PUBLIC cxx_std_11)

add_library(simplewebm
    demuxer/WebMDemuxer.cpp
    decoder/VPXDecoder.cpp
    decoder/OpusDecoder.cpp
    decoder/VorbisDecoder.cpp
)

target_include_directories(simplewebm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(simplewebm PUBLIC cxx_std_11)

find_package(PkgConfig REQUIRED)

pkg_check_modules(VPX REQUIRED IMPORTED_TARGET vpx)

pkg_check_modules(OPUS REQUIRED IMPORTED_TARGET opus)

pkg_check_modules(VORBIS REQUIRED IMPORTED_TARGET vorbis)

target_link_libraries(simplewebm PUBLIC webm_parser PRIVATE PkgConfig::VPX PkgConfig::OPUS PkgConfig::VORBIS)

if(LIBSIMPLEWEBM_BUILD_EXAMPLES)
    add_executable(simplewebm_example example/example.cpp)

    target_link_libraries(simplewebm_example PRIVATE simplewebm)

    add_custom_command(TARGET simplewebm_example POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/example/example.webm $<TARGET_FILE_DIR:simplewebm_example>/example.webm)

endif()

if(LIBSIMPLEWEBM_BUILD_PLAYER)

    find_package(SDL3 REQUIRED)

    add_executable(simplewebm_player example/player.cpp)

    target_link_libraries(simplewebm_player PRIVATE simplewebm SDL3::SDL3)

    add_custom_command(TARGET simplewebm_player POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/example/example.webm $<TARGET_FILE_DIR:simplewebm_example>/example.webm)

endif()