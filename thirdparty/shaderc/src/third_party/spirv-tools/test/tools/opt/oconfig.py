# Copyright (c) 2018 Google LLC
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

import placeholder
import expect
import re

from spirv_test_framework import inside_spirv_testsuite


def empty_main_assembly():
  return """
         OpCapability Shader
         OpMemoryModel Logical GLSL450
         OpEntryPoint Vertex %4 "main"
         OpName %4 "main"
    %2 = OpTypeVoid
    %3 = OpTypeFunction %2
    %4 = OpFunction %2 None %3
    %5 = OpLabel
         OpReturn
         OpFunctionEnd"""


@inside_spirv_testsuite('SpirvOptConfigFile')
class TestOconfigEmpty(expect.SuccessfulReturn):
  """Tests empty config files are accepted."""

  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  config = placeholder.ConfigFlagsFile('', '.cfg')
  spirv_args = [shader, '-o', placeholder.TempFileName('output.spv'), config]


@inside_spirv_testsuite('SpirvOptConfigFile')
class TestOconfigComments(expect.SuccessfulReturn):
  """Tests empty config files are accepted.

  https://github.com/KhronosGroup/SPIRV-Tools/issues/1778
  """

  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  config = placeholder.ConfigFlagsFile("""
# This is a comment.
-O
--loop-unroll
""", '.cfg')
  spirv_args = [shader, '-o', placeholder.TempFileName('output.spv'), config]

@inside_spirv_testsuite('SpirvOptConfigFile')
class TestOconfigComments(expect.SuccessfulReturn):
  """Tests empty config files are accepted.

  https://github.com/KhronosGroup/SPIRV-Tools/issues/1778
  """

  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  config = placeholder.ConfigFlagsFile("""
# This is a comment.
-O
--relax-struct-store
""", '.cfg')
  spirv_args = [shader, '-o', placeholder.TempFileName('output.spv'), config]
