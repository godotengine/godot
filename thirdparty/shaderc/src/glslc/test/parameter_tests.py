# Copyright 2015 The Shaderc Authors. All rights reserved.
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

import expect
import os.path
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader, StdinShader, TempFileName


@inside_glslc_testsuite('File')
class SimpleFileCompiled(expect.ValidObjectFile):
    """Tests whether or not a simple glsl file compiles."""

    shader = FileShader('#version 310 es\nvoid main() {}', '.frag')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('File')
class NotSpecifyingOutputName(expect.SuccessfulReturn,
                              expect.CorrectObjectFilePreamble):
    """Tests that when there is no -o and -E/-S/-c specified, output as a.spv."""

    shader = FileShader('#version 140\nvoid main() {}', '.frag')
    glslc_args = [shader]

    def check_output_a_spv(self, status):
        output_name = os.path.join(status.directory, 'a.spv')
        return self.verify_object_file_preamble(output_name)


@inside_glslc_testsuite('Parameters')
class HelpParameters(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests the --help flag outputs correctly and does not produce and error."""

    glslc_args = ['--help']

    expected_stdout = '''glslc - Compile shaders into SPIR-V

Usage: glslc [options] file...

An input file of - represents standard input.

Options:
  -c                Only run preprocess, compile, and assemble steps.
  -Dmacro[=defn]    Add an implicit macro definition.
  -E                Outputs only the results of the preprocessing step.
                    Output defaults to standard output.
  -fauto-bind-uniforms
                    Automatically assign bindings to uniform variables that
                    don't have an explicit 'binding' layout in the shader
                    source.
  -fauto-map-locations
                    Automatically assign locations to uniform variables that
                    don't have an explicit 'location' layout in the shader
                    source.
  -fentry-point=<name>
                    Specify the entry point name for HLSL compilation, for
                    all subsequent source files.  Default is "main".
  -fhlsl_functionality1, -fhlsl-functionality1
                    Enable extension SPV_GOOGLE_hlsl_functionality1 for HLSL
                    compilation.
  -fhlsl-iomap      Use HLSL IO mappings for bindings.
  -fhlsl-offsets    Use HLSL offset rules for packing members of blocks.
                    Affects only GLSL.  HLSL rules are always used for HLSL.
  -flimit=<settings>
                    Specify resource limits. Each limit is specified by a limit
                    name followed by an integer value.  Tokens should be
                    separated by whitespace.  If the same limit is specified
                    several times, only the last setting takes effect.
  -flimit-file <file>
                    Set limits as specified in the given file.
  -fresource-set-binding [stage] <reg0> <set0> <binding0>
                        [<reg1> <set1> <binding1>...]
                    Explicitly sets the descriptor set and binding for
                    HLSL resources, by register name.  Optionally restrict
                    it to a single stage.
  -fcbuffer-binding-base [stage] <value>
                    Same as -fubo-binding-base.
  -fimage-binding-base [stage] <value>
                    Sets the lowest automatically assigned binding number for
                    images.  Optionally only set it for a single shader stage.
                    For HLSL, the resource register number is added to this
                    base.
  -fsampler-binding-base [stage] <value>
                    Sets the lowest automatically assigned binding number for
                    samplers  Optionally only set it for a single shader stage.
                    For HLSL, the resource register number is added to this
                    base.
  -fssbo-binding-base [stage] <value>
                    Sets the lowest automatically assigned binding number for
                    shader storage buffer objects (SSBO).  Optionally only set
                    it for a single shader stage.  Only affects GLSL.
  -ftexture-binding-base [stage] <value>
                    Sets the lowest automatically assigned binding number for
                    textures.  Optionally only set it for a single shader stage.
                    For HLSL, the resource register number is added to this
                    base.
  -fuav-binding-base [stage] <value>
                    For automatically assigned bindings for unordered access
                    views (UAV), the register number is added to this base to
                    determine the binding number.  Optionally only set it for
                    a single shader stage.  Only affects HLSL.
  -fubo-binding-base [stage] <value>
                    Sets the lowest automatically assigned binding number for
                    uniform buffer objects (UBO).  Optionally only set it for
                    a single shader stage.
                    For HLSL, the resource register number is added to this
                    base.
  -fshader-stage=<stage>
                    Treat subsequent input files as having stage <stage>.
                    Valid stages are vertex, vert, fragment, frag, tesscontrol,
                    tesc, tesseval, tese, geometry, geom, compute, and comp.
  -g                Generate source-level debug information.
                    Currently this option has no effect.
  --help            Display available options.
  -I <value>        Add directory to include search path.
  -mfmt=<format>    Output SPIR-V binary code using the selected format. This
                    option may be specified only when the compilation output is
                    in SPIR-V binary code form. Available options include bin, c
                    and num. By default the binary output format is bin.
  -M                Generate make dependencies. Implies -E and -w.
  -MM               An alias for -M.
  -MD               Generate make dependencies and compile.
  -MF <file>        Write dependency output to the given file.
  -MT <target>      Specify the target of the rule emitted by dependency
                    generation.
  -O                Optimize the generated SPIR-V code for better performance.
  -Os               Optimize the generated SPIR-V code for smaller size.
  -O0               Disable optimization.
  -o <file>         Write output to <file>.
                    A file name of '-' represents standard output.
  -std=<value>      Version and profile for GLSL input files. Possible values
                    are concatenations of version and profile, e.g. 310es,
                    450core, etc.  Ignored for HLSL files.
  -S                Only run preprocess and compilation steps.
  --show-limits     Display available limit names and their default values.
  --target-env=<environment>
                    Set the target client environment, and the semantics
                    of warnings and errors.  An optional suffix can specify
                    the client version.  Values are:
                        vulkan1.0       # The default
                        vulkan1.1
                        vulkan          # Same as vulkan1.0
                        opengl4.5
                        opengl          # Same as opengl4.5
  --version         Display compiler version information.
  -w                Suppresses all warning messages.
  -Werror           Treat all warnings as errors.
  -x <language>     Treat subsequent input files as having type <language>.
                    Valid languages are: glsl, hlsl.
                    For files ending in .hlsl the default is hlsl.
                    Otherwise the default is glsl.
'''

    expected_stderr = ''


@inside_glslc_testsuite('Parameters')
class HelpIsNotTooWide(expect.StdoutNoWiderThan80Columns):
    """Tests that --help output is not too wide."""

    glslc_args = ['--help']


@inside_glslc_testsuite('Parameters')
class UnknownSingleLetterArgument(expect.ErrorMessage):
    """Tests that an unknown argument triggers an error message."""

    glslc_args = ['-a']
    expected_error = ["glslc: error: unknown argument: '-a'\n"]


@inside_glslc_testsuite('Parameters')
class UnknownMultiLetterArgument(expect.ErrorMessage):
    """Tests that an unknown argument triggers an error message."""

    glslc_args = ['-zzz']
    expected_error = ["glslc: error: unknown argument: '-zzz'\n"]


@inside_glslc_testsuite('Parameters')
class UnsupportedOption(expect.ErrorMessage):
    """Tests that an unsupported option triggers an error message."""

    glslc_args = ['--unsupported-option']
    expected_error = [
        "glslc: error: unsupported option: '--unsupported-option'\n"]


@inside_glslc_testsuite('File')
class FileNotFound(expect.ErrorMessage):
    """Tests the error message if a file cannot be found."""

    blabla_file = TempFileName('blabla.frag')
    glslc_args = [blabla_file]
    expected_error = [
        "glslc: error: cannot open input file: '", blabla_file,
        "': No such file or directory\n"]


@inside_glslc_testsuite('Unsupported')
class LinkingNotSupported(expect.ErrorMessage):
    """Tests the error message generated by linking not supported yet."""

    shader1 = FileShader('#version 140\nvoid main() {}', '.vert')
    shader2 = FileShader('#version 140\nvoid main() {}', '.frag')
    glslc_args = [shader1, shader2]
    expected_error = [
        'glslc: error: linking multiple files is not supported yet. ',
        'Use -c to compile files individually.\n']


@inside_glslc_testsuite('Unsupported')
class MultipleStdinUnsupported(expect.ErrorMessage):
    """Tests the error message generated by having more than one - input."""

    glslc_args = ['-c', '-fshader-stage=vertex', '-', '-']
    expected_error = [
        'glslc: error: specifying standard input "-" as input more'
        ' than once is not allowed.\n']


@inside_glslc_testsuite('Parameters')
class StdinWithoutShaderStage(expect.StdoutMatch, expect.StderrMatch):
    """Tests that you must use -fshader-stage when specifying - as input."""
    shader = StdinShader(
        """#version 140
    int a() {
    }
    void main() {
      int x = a();
    }
    """)
    glslc_args = [shader]

    expected_stdout = ''
    expected_stderr = [
        "glslc: error: '-': -fshader-stage required when input is from "
        'standard input "-"\n']


@inside_glslc_testsuite('Parameters')
class LimitsHelp(expect.StdoutMatch, expect.StderrMatch):
    """Tests --show-limits shows correct output."""

    glslc_args = ['--show-limits']

    expected_stderr = ''
    expected_stdout = """MaxLights 8
MaxClipPlanes 6
MaxTextureUnits 2
MaxTextureCoords 8
MaxVertexAttribs 16
MaxVertexUniformComponents 4096
MaxVaryingFloats 60
MaxVertexTextureImageUnits 16
MaxCombinedTextureImageUnits 80
MaxTextureImageUnits 16
MaxFragmentUniformComponents 1024
MaxDrawBuffers 8
MaxVertexUniformVectors 256
MaxVaryingVectors 15
MaxFragmentUniformVectors 256
MaxVertexOutputVectors 16
MaxFragmentInputVectors 15
MinProgramTexelOffset -8
MaxProgramTexelOffset 7
MaxClipDistances 8
MaxComputeWorkGroupCountX 65535
MaxComputeWorkGroupCountY 65535
MaxComputeWorkGroupCountZ 65535
MaxComputeWorkGroupSizeX 1024
MaxComputeWorkGroupSizeY 1024
MaxComputeWorkGroupSizeZ 64
MaxComputeUniformComponents 512
MaxComputeTextureImageUnits 16
MaxComputeImageUniforms 8
MaxComputeAtomicCounters 8
MaxComputeAtomicCounterBuffers 1
MaxVaryingComponents 60
MaxVertexOutputComponents 64
MaxGeometryInputComponents 64
MaxGeometryOutputComponents 128
MaxFragmentInputComponents 128
MaxImageUnits 8
MaxCombinedImageUnitsAndFragmentOutputs 8
MaxCombinedShaderOutputResources 8
MaxImageSamples 0
MaxVertexImageUniforms 0
MaxTessControlImageUniforms 0
MaxTessEvaluationImageUniforms 0
MaxGeometryImageUniforms 0
MaxFragmentImageUniforms 8
MaxCombinedImageUniforms 8
MaxGeometryTextureImageUnits 16
MaxGeometryOutputVertices 256
MaxGeometryTotalOutputComponents 1024
MaxGeometryUniformComponents 512
MaxGeometryVaryingComponents 60
MaxTessControlInputComponents 128
MaxTessControlOutputComponents 128
MaxTessControlTextureImageUnits 16
MaxTessControlUniformComponents 1024
MaxTessControlTotalOutputComponents 4096
MaxTessEvaluationInputComponents 128
MaxTessEvaluationOutputComponents 128
MaxTessEvaluationTextureImageUnits 16
MaxTessEvaluationUniformComponents 1024
MaxTessPatchComponents 120
MaxPatchVertices 32
MaxTessGenLevel 64
MaxViewports 16
MaxVertexAtomicCounters 0
MaxTessControlAtomicCounters 0
MaxTessEvaluationAtomicCounters 0
MaxGeometryAtomicCounters 0
MaxFragmentAtomicCounters 8
MaxCombinedAtomicCounters 8
MaxAtomicCounterBindings 1
MaxVertexAtomicCounterBuffers 0
MaxTessControlAtomicCounterBuffers 0
MaxTessEvaluationAtomicCounterBuffers 0
MaxGeometryAtomicCounterBuffers 0
MaxFragmentAtomicCounterBuffers 0
MaxCombinedAtomicCounterBuffers 1
MaxAtomicCounterBufferSize 32
MaxTransformFeedbackBuffers 4
MaxTransformFeedbackInterleavedComponents 64
MaxCullDistances 8
MaxCombinedClipAndCullDistances 8
MaxSamples 4
"""
