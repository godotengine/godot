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


@inside_spirv_testsuite('SpirvOptBase')
class TestAssemblyFileAsOnlyParameter(expect.ValidObjectFile1_3):
  """Tests that spirv-opt accepts a SPIR-V object file."""

  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  output = placeholder.TempFileName('output.spv')
  spirv_args = [shader, '-o', output]
  expected_object_filenames = (output)


@inside_spirv_testsuite('SpirvOptFlags')
class TestHelpFlag(expect.ReturnCodeIsZero, expect.StdoutMatch):
  """Test the --help flag."""

  spirv_args = ['--help']
  expected_stdout = re.compile(r'.*The SPIR-V binary is read from <input>')


@inside_spirv_testsuite('SpirvOptFlags')
class TestValidPassFlags(expect.ValidObjectFile1_3,
                         expect.ExecutedListOfPasses):
  """Tests that spirv-opt accepts all valid optimization flags."""

  flags = [
      '--ccp', '--cfg-cleanup', '--combine-access-chains', '--compact-ids',
      '--convert-local-access-chains', '--copy-propagate-arrays',
      '--eliminate-common-uniform', '--eliminate-dead-branches',
      '--eliminate-dead-code-aggressive', '--eliminate-dead-const',
      '--eliminate-dead-functions', '--eliminate-dead-inserts',
      '--eliminate-dead-variables', '--eliminate-insert-extract',
      '--eliminate-local-multi-store', '--eliminate-local-single-block',
      '--eliminate-local-single-store', '--flatten-decorations',
      '--fold-spec-const-op-composite', '--freeze-spec-const',
      '--if-conversion', '--inline-entry-points-exhaustive', '--loop-fission',
      '20', '--loop-fusion', '5', '--loop-unroll', '--loop-unroll-partial', '3',
      '--loop-peeling', '--merge-blocks', '--merge-return', '--loop-unswitch',
      '--private-to-local', '--reduce-load-size', '--redundancy-elimination',
      '--remove-duplicates', '--replace-invalid-opcode', '--ssa-rewrite',
      '--scalar-replacement', '--scalar-replacement=42', '--strength-reduction',
      '--strip-debug', '--strip-reflect', '--vector-dce', '--workaround-1209',
      '--unify-const'
  ]
  expected_passes = [
      'ccp',
      'cfg-cleanup',
      'combine-access-chains',
      'compact-ids',
      'convert-local-access-chains',
      'copy-propagate-arrays',
      'eliminate-common-uniform',
      'eliminate-dead-branches',
      'eliminate-dead-code-aggressive',
      'eliminate-dead-const',
      'eliminate-dead-functions',
      'eliminate-dead-inserts',
      'eliminate-dead-variables',
      # --eliminate-insert-extract runs the simplify-instructions pass.
      'simplify-instructions',
      'eliminate-local-multi-store',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'flatten-decorations',
      'fold-spec-const-op-composite',
      'freeze-spec-const',
      'if-conversion',
      'inline-entry-points-exhaustive',
      'loop-fission',
      'loop-fusion',
      'loop-unroll',
      'loop-unroll',
      'loop-peeling',
      'merge-blocks',
      'merge-return',
      'loop-unswitch',
      'private-to-local',
      'reduce-load-size',
      'redundancy-elimination',
      'remove-duplicates',
      'replace-invalid-opcode',
      'ssa-rewrite',
      'scalar-replacement=100',
      'scalar-replacement=42',
      'strength-reduction',
      'strip-debug',
      'strip-reflect',
      'vector-dce',
      'workaround-1209',
      'unify-const'
  ]
  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  output = placeholder.TempFileName('output.spv')
  spirv_args = [shader, '-o', output, '--print-all'] + flags
  expected_object_filenames = (output)


@inside_spirv_testsuite('SpirvOptFlags')
class TestPerformanceOptimizationPasses(expect.ValidObjectFile1_3,
                                        expect.ExecutedListOfPasses):
  """Tests that spirv-opt schedules all the passes triggered by -O."""

  flags = ['-O']
  expected_passes = [
      'eliminate-dead-branches',
      'merge-return',
      'inline-entry-points-exhaustive',
      'eliminate-dead-code-aggressive',
      'private-to-local',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'eliminate-dead-code-aggressive',
      'scalar-replacement=100',
      'convert-local-access-chains',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'eliminate-dead-code-aggressive',
      'eliminate-local-multi-store',
      'eliminate-dead-code-aggressive',
      'ccp',
      'eliminate-dead-code-aggressive',
      'redundancy-elimination',
      'combine-access-chains',
      'simplify-instructions',
      'vector-dce',
      'eliminate-dead-inserts',
      'eliminate-dead-branches',
      'simplify-instructions',
      'if-conversion',
      'copy-propagate-arrays',
      'reduce-load-size',
      'eliminate-dead-code-aggressive',
      'merge-blocks',
      'redundancy-elimination',
      'eliminate-dead-branches',
      'merge-blocks',
      'simplify-instructions',
  ]
  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  output = placeholder.TempFileName('output.spv')
  spirv_args = [shader, '-o', output, '--print-all'] + flags
  expected_object_filenames = (output)


@inside_spirv_testsuite('SpirvOptFlags')
class TestSizeOptimizationPasses(expect.ValidObjectFile1_3,
                                 expect.ExecutedListOfPasses):
  """Tests that spirv-opt schedules all the passes triggered by -Os."""

  flags = ['-Os']
  expected_passes = [
      'eliminate-dead-branches',
      'merge-return',
      'inline-entry-points-exhaustive',
      'eliminate-dead-code-aggressive',
      'private-to-local',
      'scalar-replacement=100',
      'convert-local-access-chains',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'eliminate-dead-code-aggressive',
      'simplify-instructions',
      'eliminate-dead-inserts',
      'eliminate-local-multi-store',
      'eliminate-dead-code-aggressive',
      'ccp',
      'eliminate-dead-code-aggressive',
      'eliminate-dead-branches',
      'if-conversion',
      'eliminate-dead-code-aggressive',
      'merge-blocks',
      'simplify-instructions',
      'eliminate-dead-inserts',
      'redundancy-elimination',
      'cfg-cleanup',
      'eliminate-dead-code-aggressive',
  ]
  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  output = placeholder.TempFileName('output.spv')
  spirv_args = [shader, '-o', output, '--print-all'] + flags
  expected_object_filenames = (output)


@inside_spirv_testsuite('SpirvOptFlags')
class TestLegalizationPasses(expect.ValidObjectFile1_3,
                             expect.ExecutedListOfPasses):
  """Tests that spirv-opt schedules all the passes triggered by --legalize-hlsl.
  """

  flags = ['--legalize-hlsl']
  expected_passes = [
      'eliminate-dead-branches',
      'merge-return',
      'inline-entry-points-exhaustive',
      'eliminate-dead-functions',
      'private-to-local',
      'fix-storage-class',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'eliminate-dead-code-aggressive',
      'scalar-replacement=0',
      'eliminate-local-single-block',
      'eliminate-local-single-store',
      'eliminate-dead-code-aggressive',
      'eliminate-local-multi-store',
      'eliminate-dead-code-aggressive',
      'ccp',
      'loop-unroll',
      'eliminate-dead-branches',
      'simplify-instructions',
      'eliminate-dead-code-aggressive',
      'copy-propagate-arrays',
      'vector-dce',
      'eliminate-dead-inserts',
      'reduce-load-size',
      'eliminate-dead-code-aggressive',
  ]
  shader = placeholder.FileSPIRVShader(empty_main_assembly(), '.spvasm')
  output = placeholder.TempFileName('output.spv')
  spirv_args = [shader, '-o', output, '--print-all'] + flags
  expected_object_filenames = (output)


@inside_spirv_testsuite('SpirvOptFlags')
class TestScalarReplacementArgsNegative(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --scalar-replacement."""

  spirv_args = ['--scalar-replacement=-10']
  expected_error_substr = 'must have no arguments or a non-negative integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestScalarReplacementArgsInvalidNumber(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --scalar-replacement."""

  spirv_args = ['--scalar-replacement=a10f']
  expected_error_substr = 'must have no arguments or a non-negative integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopFissionArgsNegative(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-fission."""

  spirv_args = ['--loop-fission=-10']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopFissionArgsInvalidNumber(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-fission."""

  spirv_args = ['--loop-fission=a10f']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopFusionArgsNegative(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-fusion."""

  spirv_args = ['--loop-fusion=-10']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopFusionArgsInvalidNumber(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-fusion."""

  spirv_args = ['--loop-fusion=a10f']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopUnrollPartialArgsNegative(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-unroll-partial."""

  spirv_args = ['--loop-unroll-partial=-10']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopUnrollPartialArgsInvalidNumber(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-unroll-partial."""

  spirv_args = ['--loop-unroll-partial=a10f']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopPeelingThresholdArgsNegative(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-peeling-threshold."""

  spirv_args = ['--loop-peeling-threshold=-10']
  expected_error_substr = 'must have a positive integer argument'


@inside_spirv_testsuite('SpirvOptFlags')
class TestLoopPeelingThresholdArgsInvalidNumber(expect.ErrorMessageSubstr):
  """Tests invalid arguments to --loop-peeling-threshold."""

  spirv_args = ['--loop-peeling-threshold=a10f']
  expected_error_substr = 'must have a positive integer argument'
