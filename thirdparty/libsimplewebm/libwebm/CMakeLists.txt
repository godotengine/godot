#  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
#
#  Use of this source code is governed by a BSD-style license
#  that can be found in the LICENSE file in the root of the source
#  tree. An additional intellectual property rights grant can be found
#  in the file PATENTS.  All contributing project authors may
#  be found in the AUTHORS file in the root of the source tree.
cmake_minimum_required(VERSION 3.16)
project(LIBWEBM CXX)

include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/build/cxx_flags.cmake")

if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
  include("${CMAKE_CURRENT_SOURCE_DIR}/build/msvc_runtime.cmake")
endif()

set(LIBWEBM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Build/test configuration flags.
option(ENABLE_WEBMTS "Enables WebM PES/TS support." ON)
option(ENABLE_WEBMINFO "Enables building webm_info." ON)
option(ENABLE_TESTS "Enables tests." OFF)
option(ENABLE_IWYU "Enables include-what-you-use support." OFF)
option(ENABLE_WERROR "Enable warnings as errors." OFF)
option(ENABLE_WEBM_PARSER "Enables new parser API." OFF)
option(ENABLE_SAMPLE_PROGRAMS "Enables building sample programs." ON)

if(WIN32
   OR CYGWIN
   OR MSYS)
  # Allow use of rand_r() / fdopen() and other POSIX functions.
  require_cxx_flag_nomsvc("-std=gnu++11")
else()
  require_cxx_flag_nomsvc("-std=c++11")
endif()

add_cxx_preproc_definition("__STDC_CONSTANT_MACROS")
add_cxx_preproc_definition("__STDC_FORMAT_MACROS")
add_cxx_preproc_definition("__STDC_LIMIT_MACROS")

# Set up compiler flags and build properties.
include_directories("${LIBWEBM_SRC_DIR}")

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_cxx_flag_if_supported("/W4")
  # Disable MSVC warnings that suggest making code non-portable.
  add_cxx_flag_if_supported("/wd4996")
  if(ENABLE_WERROR)
    add_cxx_flag_if_supported("/WX")
  endif()
else()
  add_cxx_flag_if_supported("-Wall")
  add_cxx_flag_if_supported("-Wc++14-compat")
  add_cxx_flag_if_supported("-Wc++17-compat")
  add_cxx_flag_if_supported("-Wc++20-compat")
  add_cxx_flag_if_supported("-Wextra")
  add_cxx_flag_if_supported("-Wnarrowing")
  add_cxx_flag_if_supported("-Wno-deprecated")
  add_cxx_flag_if_supported("-Wshorten-64-to-32")
  if(ENABLE_WERROR)
    add_cxx_flag_if_supported("-Werror")
  endif()
endif()

# Source list variables.
set(dumpvtt_sources "${LIBWEBM_SRC_DIR}/dumpvtt.cc")

set(libwebm_common_public_headers "${LIBWEBM_SRC_DIR}/common/webmids.h")

set(libwebm_common_sources
    ${libwebm_common_public_headers}
    "${LIBWEBM_SRC_DIR}/common/file_util.cc"
    "${LIBWEBM_SRC_DIR}/common/file_util.h"
    "${LIBWEBM_SRC_DIR}/common/hdr_util.cc"
    "${LIBWEBM_SRC_DIR}/common/hdr_util.h")

set(mkvmuxer_public_headers
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvmuxer.h"
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvmuxertypes.h"
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvmuxerutil.h"
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvwriter.h")

set(mkvmuxer_sources
    ${mkvmuxer_public_headers}
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvmuxer.cc"
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvmuxerutil.cc"
    "${LIBWEBM_SRC_DIR}/mkvmuxer/mkvwriter.cc"
    "${LIBWEBM_SRC_DIR}/common/webmids.h")

set(mkvmuxer_sample_sources
    "${LIBWEBM_SRC_DIR}/mkvmuxer_sample.cc"
    "${LIBWEBM_SRC_DIR}/sample_muxer_metadata.cc"
    "${LIBWEBM_SRC_DIR}/sample_muxer_metadata.h")

set(mkvmuxer_tests_sources
    "${LIBWEBM_SRC_DIR}/testing/mkvmuxer_test.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.h")

set(mkvparser_public_headers "${LIBWEBM_SRC_DIR}/mkvparser/mkvparser.h"
                             "${LIBWEBM_SRC_DIR}/mkvparser/mkvreader.h")

set(mkvparser_sources
    ${mkvparser_public_headers} "${LIBWEBM_SRC_DIR}/mkvparser/mkvparser.cc"
    "${LIBWEBM_SRC_DIR}/mkvparser/mkvreader.cc"
    "${LIBWEBM_SRC_DIR}/common/webmids.h")

set(mkvparser_sample_sources "${LIBWEBM_SRC_DIR}/mkvparser_sample.cc")
set(mkvparser_tests_sources
    "${LIBWEBM_SRC_DIR}/testing/mkvparser_test.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.h")

set(vp9_header_parser_tests_sources
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.h"
    "${LIBWEBM_SRC_DIR}/testing/test_util.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.h")

set(vp9_level_stats_tests_sources
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.h"
    "${LIBWEBM_SRC_DIR}/common/vp9_level_stats_test.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_level_stats.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_level_stats.h"
    "${LIBWEBM_SRC_DIR}/testing/test_util.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.h")

set(webm_parser_public_headers
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/buffer_reader.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/callback.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/dom_types.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/element.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/file_reader.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/id.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/istream_reader.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/reader.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/status.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/include/webm/webm_parser.h")

set(webm_parser_sources
    ${webm_parser_public_headers}
    "${LIBWEBM_SRC_DIR}/webm_parser/src/ancestory.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/ancestory.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/audio_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/bit_utils.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/bit_utils.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_additions_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_group_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_header_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_header_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_more_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/block_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/bool_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/buffer_reader.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/byte_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/callback.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/chapter_atom_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/chapter_display_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/chapters_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/cluster_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/colour_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/content_enc_aes_settings_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/content_encoding_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/content_encodings_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/content_encryption_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/cue_point_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/cue_track_positions_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/cues_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/date_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/date_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/ebml_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/edition_entry_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/element_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/file_reader.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/float_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/float_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/id_element_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/id_element_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/id_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/id_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/info_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/int_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/istream_reader.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/master_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/master_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/master_value_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/mastering_metadata_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/parser_utils.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/parser_utils.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/projection_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/recursive_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/seek_head_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/seek_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/segment_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/segment_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/simple_tag_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/size_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/size_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/skip_callback.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/skip_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/skip_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/slices_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/tag_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/tags_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/targets_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/time_slice_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/track_entry_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/tracks_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/unknown_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/unknown_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/var_int_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/var_int_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/video_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/virtual_block_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/virtual_block_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/void_parser.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/void_parser.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/src/webm_parser.cc")

set(webm_parser_demo_sources "${LIBWEBM_SRC_DIR}/webm_parser/demo/demo.cc")
set(webm_parser_tests_sources
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/audio_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/bit_utils_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/block_additions_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/block_group_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/block_header_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/block_more_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/block_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/bool_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/buffer_reader_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/byte_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/callback_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/chapter_atom_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/chapter_display_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/chapters_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/cluster_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/colour_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/content_enc_aes_settings_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/content_encoding_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/content_encodings_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/content_encryption_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/cue_point_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/cue_track_positions_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/cues_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/date_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/ebml_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/edition_entry_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/element_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/float_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/id_element_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/id_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/info_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/int_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/istream_reader_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/limited_reader_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/master_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/master_value_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/mastering_metadata_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/parser_utils_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/projection_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/recursive_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/seek_head_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/seek_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/segment_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/simple_tag_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/size_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/skip_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/slices_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/tag_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/tags_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/targets_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/time_slice_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/track_entry_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/tracks_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/unknown_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/var_int_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/video_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/virtual_block_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/void_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/tests/webm_parser_test.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/test_utils/element_parser_test.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/test_utils/limited_reader.cc"
    "${LIBWEBM_SRC_DIR}/webm_parser/test_utils/limited_reader.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/test_utils/mock_callback.h"
    "${LIBWEBM_SRC_DIR}/webm_parser/test_utils/parser_test.h")

set(webm_info_sources
    "${LIBWEBM_SRC_DIR}/common/indent.cc"
    "${LIBWEBM_SRC_DIR}/common/indent.h"
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_header_parser.h"
    "${LIBWEBM_SRC_DIR}/common/vp9_level_stats.cc"
    "${LIBWEBM_SRC_DIR}/common/vp9_level_stats.h"
    "${LIBWEBM_SRC_DIR}/common/webm_constants.h"
    "${LIBWEBM_SRC_DIR}/common/webm_endian.cc"
    "${LIBWEBM_SRC_DIR}/common/webm_endian.h"
    "${LIBWEBM_SRC_DIR}/webm_info.cc")

set(webmts_sources
    "${LIBWEBM_SRC_DIR}/common/libwebm_util.cc"
    "${LIBWEBM_SRC_DIR}/common/libwebm_util.h"
    "${LIBWEBM_SRC_DIR}/common/video_frame.cc"
    "${LIBWEBM_SRC_DIR}/common/video_frame.h"
    "${LIBWEBM_SRC_DIR}/m2ts/vpxpes2ts.cc"
    "${LIBWEBM_SRC_DIR}/m2ts/vpxpes2ts.h"
    "${LIBWEBM_SRC_DIR}/m2ts/vpxpes_parser.cc"
    "${LIBWEBM_SRC_DIR}/m2ts/vpxpes_parser.h"
    "${LIBWEBM_SRC_DIR}/m2ts/webm2pes.cc"
    "${LIBWEBM_SRC_DIR}/m2ts/webm2pes.h")

set(webm2pes_sources "${LIBWEBM_SRC_DIR}/m2ts/webm2pes_main.cc")
set(webm2pes_tests_sources
    "${LIBWEBM_SRC_DIR}/testing/test_util.cc"
    "${LIBWEBM_SRC_DIR}/testing/test_util.h"
    "${LIBWEBM_SRC_DIR}/testing/video_frame_test.cc"
    "${LIBWEBM_SRC_DIR}/m2ts/tests/webm2pes_test.cc")
set(webm2ts_sources "${LIBWEBM_SRC_DIR}/m2ts/vpxpes2ts_main.cc")

set(webvtt_common_headers "${LIBWEBM_SRC_DIR}/webvtt/vttreader.h"
                          "${LIBWEBM_SRC_DIR}/webvtt/webvttparser.h")

set(webvtt_common_sources
    ${webvtt_common_headers} "${LIBWEBM_SRC_DIR}/webvtt/vttreader.cc"
    "${LIBWEBM_SRC_DIR}/webvtt/webvttparser.cc")

set(vttdemux_sources ${webvtt_common_sources} "${LIBWEBM_SRC_DIR}/vttdemux.cc")

# Targets.
add_library(mkvmuxer OBJECT ${mkvmuxer_sources})
add_library(mkvparser OBJECT ${mkvparser_sources})
add_library(webvtt_common OBJECT ${webvtt_common_sources})

add_library(webm ${libwebm_common_sources} $<TARGET_OBJECTS:mkvmuxer>
                 $<TARGET_OBJECTS:mkvparser>)

if(ENABLE_SAMPLE_PROGRAMS)
  add_executable(mkvparser_sample ${mkvparser_sample_sources})
  target_link_libraries(mkvparser_sample LINK_PUBLIC webm)

  add_executable(mkvmuxer_sample ${mkvmuxer_sample_sources}
                                 $<TARGET_OBJECTS:webvtt_common>)
  target_link_libraries(mkvmuxer_sample LINK_PUBLIC webm)

  add_executable(dumpvtt ${dumpvtt_sources} $<TARGET_OBJECTS:webvtt_common>)
  target_link_libraries(dumpvtt LINK_PUBLIC webm)

  add_executable(vttdemux ${vttdemux_sources})
  target_link_libraries(vttdemux LINK_PUBLIC webm)
endif()

if(ENABLE_WEBMINFO)
  add_executable(webm_info ${webm_info_sources})
  target_link_libraries(webm_info LINK_PUBLIC webm)
endif()

if(ENABLE_WEBM_PARSER)
  include_directories(webm_parser webm_parser/include)
  add_library(webm_parser OBJECT ${webm_parser_sources})
  target_sources(webm PUBLIC $<TARGET_OBJECTS:webm_parser>)

  add_executable(webm_parser_demo ${webm_parser_demo_sources})
  target_link_libraries(webm_parser_demo LINK_PUBLIC webm)
endif()

if(ENABLE_WEBMTS)
  add_library(webmts OBJECT ${webmts_sources})

  add_executable(webm2pes ${webm2pes_sources} $<TARGET_OBJECTS:webmts>)
  target_link_libraries(webm2pes LINK_PUBLIC webm)

  add_executable(webm2ts ${webm2ts_sources} $<TARGET_OBJECTS:webmts>)
  target_link_libraries(webm2ts LINK_PUBLIC webm)
endif()

if(ENABLE_TESTS)
  set(GTEST_SRC_DIR
      "${LIBWEBM_SRC_DIR}/../googletest"
      CACHE PATH "Path to Googletest git repository.")
  # This directory is where libwebm will build googletest dependencies.
  set(GTEST_BUILD_DIR "${CMAKE_BINARY_DIR}/googletest_build")

  if(LIBWEBM_DISABLE_GTEST_CMAKE)
    add_library(gtest STATIC "${GTEST_SRC_DIR}/googletest/src/gtest-all.cc")
    add_library(gtest_main STATIC
                "${GTEST_SRC_DIR}/googletest/src/gtest_main.cc")
    target_link_libraries(gtest_main PUBLIC gtest)
    include_directories("${GTEST_SRC_DIR}/googletest")
  else()
    add_subdirectory("${GTEST_SRC_DIR}" "${GTEST_BUILD_DIR}")
  endif()
  include_directories("${GTEST_SRC_DIR}/googletest/include")

  add_executable(mkvmuxer_tests ${mkvmuxer_tests_sources})
  target_link_libraries(mkvmuxer_tests LINK_PUBLIC gtest_main webm)

  add_executable(mkvparser_tests ${mkvparser_tests_sources})
  target_link_libraries(mkvparser_tests LINK_PUBLIC gtest_main webm)

  add_executable(vp9_header_parser_tests ${vp9_header_parser_tests_sources})
  target_link_libraries(vp9_header_parser_tests LINK_PUBLIC gtest_main webm)

  add_executable(vp9_level_stats_tests ${vp9_level_stats_tests_sources})
  target_link_libraries(vp9_level_stats_tests LINK_PUBLIC gtest_main webm)

  if(ENABLE_WEBMTS)
    add_executable(webm2pes_tests ${webm2pes_tests_sources}
                                  $<TARGET_OBJECTS:webmts>)
    target_link_libraries(webm2pes_tests LINK_PUBLIC gtest_main webm)
  endif()

  if(ENABLE_WEBM_PARSER)
    include_directories("${GTEST_SRC_DIR}/googlemock/include")
    add_executable(webm_parser_tests ${webm_parser_tests_sources})
    target_link_libraries(webm_parser_tests LINK_PUBLIC gmock gtest_main webm)
  endif()
endif()

# Include-what-you-use.
if(ENABLE_IWYU)
  # Make sure all the tools necessary for IWYU are present.
  find_program(iwyu_path NAMES include-what-you-use)
  find_program(iwyu_tool_path NAMES iwyu_tool.py)

  # Some odd behavior on cmake's part: PYTHON_EXECUTABLE and PYTHON_VERSION_*
  # are set by cmake when it does its internal python check, but
  # PYTHONINTERP_FOUND is empty without explicitly looking for it.
  find_package(PythonInterp)

  if(iwyu_path
     AND iwyu_tool_path
     AND PYTHONINTERP_FOUND)
    # Enable compilation command export (needed for iwyu_tool.py)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Add a custom target to run iwyu across all targets.
    add_custom_target(
      iwyu ALL
      COMMAND "${PYTHON_EXECUTABLE}" "${iwyu_tool_path}" -p
              "${CMAKE_BINARY_DIR}"
      COMMENT "Running include-what-you-use..."
      VERBATIM)
  else()
    message(STATUS "Ignoring ENABLE_IWYU because reasons:")
    message(STATUS "  iwyu_path=" ${iwyu_path})
    message(STATUS "  iwyu_tool_path=" ${iwyu_tool_path})
    message(STATUS "  PYTHONINTERP_FOUND=" ${PYTHONINTERP_FOUND})
    message(STATUS "  See README.libwebm for more information.")
  endif()
endif()

# webm_parser headers are rooted at webm/.
set_target_properties(webm PROPERTIES PUBLIC_HEADER
                                      "${webm_parser_public_headers}")
install(
  TARGETS webm
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webm)

# Install common headers into a subdirectory to avoid breaking nested includes.
install(FILES ${libwebm_common_public_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webm/common)
# mkvmuxer and mkvparser headers are referenced with their prefix.
install(FILES ${mkvmuxer_public_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webm/mkvmuxer)
install(FILES ${mkvparser_public_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/webm/mkvparser)
