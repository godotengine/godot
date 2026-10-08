cmake_minimum_required(VERSION 4.0)

project(simplewebm VERSION 0.0.2 LANGUAGES CXX)

option(LIBSIMPLEWEBM_BUILD_PLAYER "Build the SDL3-based simplewebm player" OFF)

set(WEBM_PARSER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libwebm)

set(WEBM_PARSER_SOURCES
    "${WEBM_PARSER_DIR}/webm_parser/src/ancestory.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/ancestory.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/audio_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/bit_utils.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/bit_utils.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_additions_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_group_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_header_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_header_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_more_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/block_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/bool_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/buffer_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/byte_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/callback.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/chapter_atom_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/chapter_display_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/chapters_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/cluster_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/colour_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/content_enc_aes_settings_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/content_encoding_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/content_encodings_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/content_encryption_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/cue_point_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/cue_track_positions_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/cues_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/date_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/date_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/ebml_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/edition_entry_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/element_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/file_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/float_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/float_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_element_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_element_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/id_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/info_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/int_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/istream_reader.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/master_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/master_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/master_value_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/mastering_metadata_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/parser_utils.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/parser_utils.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/projection_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/recursive_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/seek_head_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/seek_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/segment_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/segment_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/simple_tag_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/size_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/size_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/skip_callback.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/skip_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/skip_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/slices_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/tag_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/tags_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/targets_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/time_slice_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/track_entry_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/tracks_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/unknown_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/unknown_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/var_int_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/var_int_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/video_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/virtual_block_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/virtual_block_parser.h"
    "${WEBM_PARSER_DIR}/webm_parser/src/void_parser.cc"
    "${WEBM_PARSER_DIR}/webm_parser/src/void_parser.h"
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

add_executable(simplewebm_example example/example.cpp)

target_link_libraries(simplewebm_example PRIVATE simplewebm)

if(LIBSIMPLEWEBM_BUILD_PLAYER)

    find_package(SDL3 REQUIRED)

    add_executable(simplewebm_player example/player.cpp)

    target_link_libraries(simplewebm_player PRIVATE simplewebm SDL3::SDL3)

endif()