// Copyright (c) 2020 The Khronos Group Inc.
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

declare interface SpirvTools {
  as(input: string, env: number, options: number): Uint8Array;
  dis(input: Uint8Array, env: number, options: number): string;

  SPV_ENV_UNIVERSAL_1_0: number;
  SPV_ENV_VULKAN_1_0: number;
  SPV_ENV_UNIVERSAL_1_1: number;
  SPV_ENV_OPENCL_2_1: number;
  SPV_ENV_OPENCL_2_2: number;
  SPV_ENV_OPENGL_4_0: number;
  SPV_ENV_OPENGL_4_1: number;
  SPV_ENV_OPENGL_4_2: number;
  SPV_ENV_OPENGL_4_3: number;
  SPV_ENV_OPENGL_4_5: number;
  SPV_ENV_UNIVERSAL_1_2: number;
  SPV_ENV_OPENCL_1_2: number;
  SPV_ENV_OPENCL_EMBEDDED_1_2: number;
  SPV_ENV_OPENCL_2_0: number;
  SPV_ENV_OPENCL_EMBEDDED_2_0: number;
  SPV_ENV_OPENCL_EMBEDDED_2_1: number;
  SPV_ENV_OPENCL_EMBEDDED_2_2: number;
  SPV_ENV_UNIVERSAL_1_3: number;
  SPV_ENV_VULKAN_1_1: number;
  SPV_ENV_WEBGPU_0: number;
  SPV_ENV_UNIVERSAL_1_4: number;
  SPV_ENV_VULKAN_1_1_SPIRV_1_4: number;
  SPV_ENV_UNIVERSAL_1_5: number;
  SPV_ENV_VULKAN_1_2: number;
  SPV_ENV_UNIVERSAL_1_6: number;

  SPV_TEXT_TO_BINARY_OPTION_NONE: number;
  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS: number;

  SPV_BINARY_TO_TEXT_OPTION_NONE: number;
  SPV_BINARY_TO_TEXT_OPTION_PRINT: number;
  SPV_BINARY_TO_TEXT_OPTION_COLOR: number;
  SPV_BINARY_TO_TEXT_OPTION_INDENT: number;
  SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET: number;
  SPV_BINARY_TO_TEXT_OPTION_NO_HEADER: number;
  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES: number;
}

export default function (): Promise<SpirvTools>;
