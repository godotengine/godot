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

set(VAL_TEST_COMMON_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../test_fixture.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_spirv.h
)

if(SPIRV_BUILD_COMPRESSION)
  add_spvtools_unittest(TARGET markv_codec
    SRCS
      markv_codec_test.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/comp/markv_model_factory.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/comp/markv_model_shader.cpp
      ${VAL_TEST_COMMON_SRCS}
    LIBS SPIRV-Tools-comp ${SPIRV_TOOLS}
  )
endif(SPIRV_BUILD_COMPRESSION)
