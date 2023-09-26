// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/ext_inst.h"

#include <cstring>

// DebugInfo extended instruction set.
// See https://www.khronos.org/registry/spir-v/specs/1.0/DebugInfo.html
// TODO(dneto): DebugInfo.h should probably move to SPIRV-Headers.
#include "DebugInfo.h"

#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_opencl_std_header.h"
#include "source/macro.h"
#include "source/spirv_definition.h"

#include "debuginfo.insts.inc"
#include "glsl.std.450.insts.inc"
#include "nonsemantic.clspvreflection.insts.inc"
#include "nonsemantic.shader.debuginfo.100.insts.inc"
#include "opencl.debuginfo.100.insts.inc"
#include "opencl.std.insts.inc"

#include "spirv-tools/libspirv.h"
#include "spv-amd-gcn-shader.insts.inc"
#include "spv-amd-shader-ballot.insts.inc"
#include "spv-amd-shader-explicit-vertex-parameter.insts.inc"
#include "spv-amd-shader-trinary-minmax.insts.inc"

static const spv_ext_inst_group_t kGroups_1_0[] = {
    {SPV_EXT_INST_TYPE_GLSL_STD_450, ARRAY_SIZE(glsl_entries), glsl_entries},
    {SPV_EXT_INST_TYPE_OPENCL_STD, ARRAY_SIZE(opencl_entries), opencl_entries},
    {SPV_EXT_INST_TYPE_SPV_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER,
     ARRAY_SIZE(spv_amd_shader_explicit_vertex_parameter_entries),
     spv_amd_shader_explicit_vertex_parameter_entries},
    {SPV_EXT_INST_TYPE_SPV_AMD_SHADER_TRINARY_MINMAX,
     ARRAY_SIZE(spv_amd_shader_trinary_minmax_entries),
     spv_amd_shader_trinary_minmax_entries},
    {SPV_EXT_INST_TYPE_SPV_AMD_GCN_SHADER,
     ARRAY_SIZE(spv_amd_gcn_shader_entries), spv_amd_gcn_shader_entries},
    {SPV_EXT_INST_TYPE_SPV_AMD_SHADER_BALLOT,
     ARRAY_SIZE(spv_amd_shader_ballot_entries), spv_amd_shader_ballot_entries},
    {SPV_EXT_INST_TYPE_DEBUGINFO, ARRAY_SIZE(debuginfo_entries),
     debuginfo_entries},
    {SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100,
     ARRAY_SIZE(opencl_debuginfo_100_entries), opencl_debuginfo_100_entries},
    {SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100,
     ARRAY_SIZE(nonsemantic_shader_debuginfo_100_entries),
     nonsemantic_shader_debuginfo_100_entries},
    {SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION,
     ARRAY_SIZE(nonsemantic_clspvreflection_entries),
     nonsemantic_clspvreflection_entries},
};

static const spv_ext_inst_table_t kTable_1_0 = {ARRAY_SIZE(kGroups_1_0),
                                                kGroups_1_0};

spv_result_t spvExtInstTableGet(spv_ext_inst_table* pExtInstTable,
                                spv_target_env env) {
  if (!pExtInstTable) return SPV_ERROR_INVALID_POINTER;

  switch (env) {
    // The extended instruction sets are all version 1.0 so far.
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
      *pExtInstTable = &kTable_1_0;
      return SPV_SUCCESS;
    default:
      return SPV_ERROR_INVALID_TABLE;
  }
}

spv_ext_inst_type_t spvExtInstImportTypeGet(const char* name) {
  // The names are specified by the respective extension instruction
  // specifications.
  if (!strcmp("GLSL.std.450", name)) {
    return SPV_EXT_INST_TYPE_GLSL_STD_450;
  }
  if (!strcmp("OpenCL.std", name)) {
    return SPV_EXT_INST_TYPE_OPENCL_STD;
  }
  if (!strcmp("SPV_AMD_shader_explicit_vertex_parameter", name)) {
    return SPV_EXT_INST_TYPE_SPV_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER;
  }
  if (!strcmp("SPV_AMD_shader_trinary_minmax", name)) {
    return SPV_EXT_INST_TYPE_SPV_AMD_SHADER_TRINARY_MINMAX;
  }
  if (!strcmp("SPV_AMD_gcn_shader", name)) {
    return SPV_EXT_INST_TYPE_SPV_AMD_GCN_SHADER;
  }
  if (!strcmp("SPV_AMD_shader_ballot", name)) {
    return SPV_EXT_INST_TYPE_SPV_AMD_SHADER_BALLOT;
  }
  if (!strcmp("DebugInfo", name)) {
    return SPV_EXT_INST_TYPE_DEBUGINFO;
  }
  if (!strcmp("OpenCL.DebugInfo.100", name)) {
    return SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100;
  }
  if (!strcmp("NonSemantic.Shader.DebugInfo.100", name)) {
    return SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100;
  }
  if (!strncmp("NonSemantic.ClspvReflection.", name, 28)) {
    return SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION;
  }
  // ensure to add any known non-semantic extended instruction sets
  // above this point, and update spvExtInstIsNonSemantic()
  if (!strncmp("NonSemantic.", name, 12)) {
    return SPV_EXT_INST_TYPE_NONSEMANTIC_UNKNOWN;
  }
  return SPV_EXT_INST_TYPE_NONE;
}

bool spvExtInstIsNonSemantic(const spv_ext_inst_type_t type) {
  if (type == SPV_EXT_INST_TYPE_NONSEMANTIC_UNKNOWN ||
      type == SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100 ||
      type == SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION) {
    return true;
  }
  return false;
}

bool spvExtInstIsDebugInfo(const spv_ext_inst_type_t type) {
  if (type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 ||
      type == SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100 ||
      type == SPV_EXT_INST_TYPE_DEBUGINFO) {
    return true;
  }
  return false;
}

spv_result_t spvExtInstTableNameLookup(const spv_ext_inst_table table,
                                       const spv_ext_inst_type_t type,
                                       const char* name,
                                       spv_ext_inst_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  for (uint32_t groupIndex = 0; groupIndex < table->count; groupIndex++) {
    const auto& group = table->groups[groupIndex];
    if (type != group.type) continue;
    for (uint32_t index = 0; index < group.count; index++) {
      const auto& entry = group.entries[index];
      if (!strcmp(name, entry.name)) {
        *pEntry = &entry;
        return SPV_SUCCESS;
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

spv_result_t spvExtInstTableValueLookup(const spv_ext_inst_table table,
                                        const spv_ext_inst_type_t type,
                                        const uint32_t value,
                                        spv_ext_inst_desc* pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  for (uint32_t groupIndex = 0; groupIndex < table->count; groupIndex++) {
    const auto& group = table->groups[groupIndex];
    if (type != group.type) continue;
    for (uint32_t index = 0; index < group.count; index++) {
      const auto& entry = group.entries[index];
      if (value == entry.ext_inst) {
        *pEntry = &entry;
        return SPV_SUCCESS;
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}
