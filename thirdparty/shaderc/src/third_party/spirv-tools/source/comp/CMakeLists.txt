# Copyright (c) 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(SPIRV_BUILD_COMPRESSION)
  add_library(SPIRV-Tools-comp
      bit_stream.cpp
      bit_stream.h
      huffman_codec.h
      markv_codec.cpp
      markv_codec.h
      markv.cpp
      markv.h
      markv_decoder.cpp
      markv_decoder.h
      markv_encoder.cpp
      markv_encoder.h
      markv_logger.h
      move_to_front.h
      move_to_front.cpp)

  spvtools_default_compile_options(SPIRV-Tools-comp)
  target_include_directories(SPIRV-Tools-comp
    PUBLIC ${spirv-tools_SOURCE_DIR}/include
    PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
    PRIVATE ${spirv-tools_BINARY_DIR}
  )

  target_link_libraries(SPIRV-Tools-comp
    PUBLIC ${SPIRV_TOOLS})

  set_property(TARGET SPIRV-Tools-comp PROPERTY FOLDER "SPIRV-Tools libraries")
  spvtools_check_symbol_exports(SPIRV-Tools-comp)

  if(ENABLE_SPIRV_TOOLS_INSTALL)
    install(TARGETS SPIRV-Tools-comp
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif(ENABLE_SPIRV_TOOLS_INSTALL)

endif(SPIRV_BUILD_COMPRESSION)
