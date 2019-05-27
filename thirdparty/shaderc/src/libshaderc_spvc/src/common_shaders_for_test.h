// Copyright 2018 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMON_SHADERS_FOR_TESTS_H_
#define COMMON_SHADERS_FOR_TESTS_H_

#ifdef __cplusplus
extern "C" {
#endif

const uint32_t kShader1[] = {
    /*
                   OpCapability Shader
              %1 = OpExtInstImport "GLSL.std.450"
                   OpMemoryModel Logical GLSL450
                   OpEntryPoint Vertex %main "main" %outColor %vtxColor
                   OpSource ESSL 310
                   OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
                   OpSourceExtension "GL_GOOGLE_include_directive"
                   OpName %main "main"
                   OpName %outColor "outColor"
                   OpName %vtxColor "vtxColor"
                   OpDecorate %outColor Location 0
                   OpDecorate %vtxColor Location 0
           %void = OpTypeVoid
              %3 = OpTypeFunction %void
          %float = OpTypeFloat 32
        %v4float = OpTypeVector %float 4
    %_ptr_Output_v4float = OpTypePointer Output %v4float
       %outColor = OpVariable %_ptr_Output_v4float Output
    %_ptr_Input_v4float = OpTypePointer Input %v4float
       %vtxColor = OpVariable %_ptr_Input_v4float Input
           %main = OpFunction %void None %3
              %5 = OpLabel
             %12 = OpLoad %v4float %vtxColor
                   OpStore %outColor %12
                   OpReturn
                   OpFunctionEnd
    */
    0x07230203, 0x00010000, 0x000d0007, 0x0000000d, 0x00000000, 0x00020011,
    0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
    0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000b, 0x00030003,
    0x00000001, 0x00000136, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45,
    0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365, 0x00006576,
    0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65,
    0x74636572, 0x00657669, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00050005, 0x00000009, 0x4374756f, 0x726f6c6f, 0x00000000, 0x00050005,
    0x0000000b, 0x43787476, 0x726f6c6f, 0x00000000, 0x00040047, 0x00000009,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000b, 0x0000001e, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
    0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004,
    0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000007,
    0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000007, 0x0000000c, 0x0000000b, 0x0003003e, 0x00000009, 0x0000000c,
    0x000100fd, 0x00010038,
};

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // COMMON_SHADERS_FOR_TESTS_H_
