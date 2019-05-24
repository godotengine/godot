// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/local_ssa_elim_pass.h"

#include "source/cfa.h"
#include "source/opt/iterator.h"
#include "source/opt/ssa_rewrite_pass.h"

namespace spvtools {
namespace opt {

bool LocalMultiStoreElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status LocalMultiStoreElimPass::ProcessImpl() {
  // Assumes relaxed logical addressing only (see instruction.h)
  // TODO(greg-lunarg): Add support for physical addressing
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;
  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == SpvOpGroupDecorate) return Status::SuccessWithoutChange;
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process functions
  ProcessFunction pfn = [this](Function* fp) {
    return SSARewriter(this).RewriteFunctionIntoSSA(fp);
  };
  bool modified = context()->ProcessEntryPointCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalMultiStoreElimPass::LocalMultiStoreElimPass() = default;

Pass::Status LocalMultiStoreElimPass::Process() {
  // Initialize extension whitelist
  InitExtensions();
  return ProcessImpl();
}

void LocalMultiStoreElimPass::InitExtensions() {
  extensions_whitelist_.clear();
  extensions_whitelist_.insert({
      "SPV_AMD_shader_explicit_vertex_parameter",
      "SPV_AMD_shader_trinary_minmax",
      "SPV_AMD_gcn_shader",
      "SPV_KHR_shader_ballot",
      "SPV_AMD_shader_ballot",
      "SPV_AMD_gpu_shader_half_float",
      "SPV_KHR_shader_draw_parameters",
      "SPV_KHR_subgroup_vote",
      "SPV_KHR_16bit_storage",
      "SPV_KHR_device_group",
      "SPV_KHR_multiview",
      "SPV_NVX_multiview_per_view_attributes",
      "SPV_NV_viewport_array2",
      "SPV_NV_stereo_view_rendering",
      "SPV_NV_sample_mask_override_coverage",
      "SPV_NV_geometry_shader_passthrough",
      "SPV_AMD_texture_gather_bias_lod",
      "SPV_KHR_storage_buffer_storage_class",
      "SPV_KHR_variable_pointers",
      "SPV_AMD_gpu_shader_int16",
      "SPV_KHR_post_depth_coverage",
      "SPV_KHR_shader_atomic_counter_ops",
      "SPV_EXT_shader_stencil_export",
      "SPV_EXT_shader_viewport_index_layer",
      "SPV_AMD_shader_image_load_store_lod",
      "SPV_AMD_shader_fragment_mask",
      "SPV_EXT_fragment_fully_covered",
      "SPV_AMD_gpu_shader_half_float_fetch",
      "SPV_GOOGLE_decorate_string",
      "SPV_GOOGLE_hlsl_functionality1",
      "SPV_NV_shader_subgroup_partitioned",
      "SPV_EXT_descriptor_indexing",
      "SPV_NV_fragment_shader_barycentric",
      "SPV_NV_compute_shader_derivatives",
      "SPV_NV_shader_image_footprint",
      "SPV_NV_shading_rate",
      "SPV_NV_mesh_shader",
      "SPV_NV_ray_tracing",
      "SPV_EXT_fragment_invocation_density",
  });
}

}  // namespace opt
}  // namespace spvtools
