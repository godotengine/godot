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

#include "source/opt/local_single_block_elim_pass.h"

#include <vector>

#include "source/opt/iterator.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kStoreValIdInIdx = 1;
}  // namespace

bool LocalSingleBlockLoadStoreElimPass::HasOnlySupportedRefs(uint32_t ptrId) {
  if (supported_ref_ptrs_.find(ptrId) != supported_ref_ptrs_.end()) return true;
  if (get_def_use_mgr()->WhileEachUser(ptrId, [this](Instruction* user) {
        auto dbg_op = user->GetCommonDebugOpcode();
        if (dbg_op == CommonDebugInfoDebugDeclare ||
            dbg_op == CommonDebugInfoDebugValue) {
          return true;
        }
        spv::Op op = user->opcode();
        if (IsNonPtrAccessChain(op) || op == spv::Op::OpCopyObject) {
          if (!HasOnlySupportedRefs(user->result_id())) {
            return false;
          }
        } else if (op != spv::Op::OpStore && op != spv::Op::OpLoad &&
                   op != spv::Op::OpName && !IsNonTypeDecorate(op)) {
          return false;
        }
        return true;
      })) {
    supported_ref_ptrs_.insert(ptrId);
    return true;
  }
  return false;
}

bool LocalSingleBlockLoadStoreElimPass::LocalSingleBlockLoadStoreElim(
    Function* func) {
  // Perform local store/load, load/load and store/store elimination
  // on each block
  bool modified = false;
  std::vector<Instruction*> instructions_to_kill;
  std::unordered_set<Instruction*> instructions_to_save;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    var2store_.clear();
    var2load_.clear();
    auto next = bi->begin();
    for (auto ii = next; ii != bi->end(); ii = next) {
      ++next;
      switch (ii->opcode()) {
        case spv::Op::OpStore: {
          // Verify store variable is target type
          uint32_t varId;
          Instruction* ptrInst = GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) continue;
          if (!HasOnlySupportedRefs(varId)) continue;
          // If a store to the whole variable, remember it for succeeding
          // loads and stores. Otherwise forget any previous store to that
          // variable.
          if (ptrInst->opcode() == spv::Op::OpVariable) {
            // If a previous store to same variable, mark the store
            // for deletion if not still used. Don't delete store
            // if debugging; let ssa-rewrite and DCE handle it
            auto prev_store = var2store_.find(varId);
            if (prev_store != var2store_.end() &&
                instructions_to_save.count(prev_store->second) == 0 &&
                !context()->get_debug_info_mgr()->IsVariableDebugDeclared(
                    varId)) {
              instructions_to_kill.push_back(prev_store->second);
              modified = true;
            }

            bool kill_store = false;
            auto li = var2load_.find(varId);
            if (li != var2load_.end()) {
              if (ii->GetSingleWordInOperand(kStoreValIdInIdx) ==
                  li->second->result_id()) {
                // We are storing the same value that already exists in the
                // memory location.  The store does nothing.
                kill_store = true;
              }
            }

            if (!kill_store) {
              var2store_[varId] = &*ii;
              var2load_.erase(varId);
            } else {
              instructions_to_kill.push_back(&*ii);
              modified = true;
            }
          } else {
            assert(IsNonPtrAccessChain(ptrInst->opcode()));
            var2store_.erase(varId);
            var2load_.erase(varId);
          }
        } break;
        case spv::Op::OpLoad: {
          // Verify store variable is target type
          uint32_t varId;
          Instruction* ptrInst = GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) continue;
          if (!HasOnlySupportedRefs(varId)) continue;
          uint32_t replId = 0;
          if (ptrInst->opcode() == spv::Op::OpVariable) {
            // If a load from a variable, look for a previous store or
            // load from that variable and use its value.
            auto si = var2store_.find(varId);
            if (si != var2store_.end()) {
              replId = si->second->GetSingleWordInOperand(kStoreValIdInIdx);
            } else {
              auto li = var2load_.find(varId);
              if (li != var2load_.end()) {
                replId = li->second->result_id();
              }
            }
          } else {
            // If a partial load of a previously seen store, remember
            // not to delete the store.
            auto si = var2store_.find(varId);
            if (si != var2store_.end()) instructions_to_save.insert(si->second);
          }
          if (replId != 0) {
            // replace load's result id and delete load
            context()->KillNamesAndDecorates(&*ii);
            context()->ReplaceAllUsesWith(ii->result_id(), replId);
            instructions_to_kill.push_back(&*ii);
            modified = true;
          } else {
            if (ptrInst->opcode() == spv::Op::OpVariable)
              var2load_[varId] = &*ii;  // register load
          }
        } break;
        case spv::Op::OpFunctionCall: {
          // Conservatively assume all locals are redefined for now.
          // TODO(): Handle more optimally
          var2store_.clear();
          var2load_.clear();
        } break;
        default:
          break;
      }
    }
  }

  for (Instruction* inst : instructions_to_kill) {
    context()->KillInst(inst);
  }

  return modified;
}

void LocalSingleBlockLoadStoreElimPass::Initialize() {
  // Initialize Target Type Caches
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();

  // Clear collections
  supported_ref_ptrs_.clear();

  // Initialize extensions allowlist
  InitExtensions();
}

bool LocalSingleBlockLoadStoreElimPass::AllExtensionsSupported() const {
  // If any extension not in allowlist, return false
  for (auto& ei : get_module()->extensions()) {
    const std::string extName = ei.GetInOperand(0).AsString();
    if (extensions_allowlist_.find(extName) == extensions_allowlist_.end())
      return false;
  }
  // only allow NonSemantic.Shader.DebugInfo.100, we cannot safely optimise
  // around unknown extended
  // instruction sets even if they are non-semantic
  for (auto& inst : context()->module()->ext_inst_imports()) {
    assert(inst.opcode() == spv::Op::OpExtInstImport &&
           "Expecting an import of an extension's instruction set.");
    const std::string extension_name = inst.GetInOperand(0).AsString();
    if (spvtools::utils::starts_with(extension_name, "NonSemantic.") &&
        extension_name != "NonSemantic.Shader.DebugInfo.100") {
      return false;
    }
  }
  return true;
}

Pass::Status LocalSingleBlockLoadStoreElimPass::ProcessImpl() {
  // Assumes relaxed logical addressing only (see instruction.h).
  if (context()->get_feature_mgr()->HasCapability(spv::Capability::Addresses))
    return Status::SuccessWithoutChange;

  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == spv::Op::OpGroupDecorate)
      return Status::SuccessWithoutChange;
  // If any extensions in the module are not explicitly supported,
  // return unmodified.
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process all entry point functions
  ProcessFunction pfn = [this](Function* fp) {
    return LocalSingleBlockLoadStoreElim(fp);
  };

  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalSingleBlockLoadStoreElimPass::LocalSingleBlockLoadStoreElimPass() =
    default;

Pass::Status LocalSingleBlockLoadStoreElimPass::Process() {
  Initialize();
  return ProcessImpl();
}

void LocalSingleBlockLoadStoreElimPass::InitExtensions() {
  extensions_allowlist_.clear();
  extensions_allowlist_.insert({
      "SPV_AMD_shader_explicit_vertex_parameter",
      "SPV_AMD_shader_trinary_minmax",
      "SPV_AMD_gcn_shader",
      "SPV_KHR_shader_ballot",
      "SPV_AMD_shader_ballot",
      "SPV_AMD_gpu_shader_half_float",
      "SPV_KHR_shader_draw_parameters",
      "SPV_KHR_subgroup_vote",
      "SPV_KHR_8bit_storage",
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
      "SPV_GOOGLE_user_type",
      "SPV_NV_shader_subgroup_partitioned",
      "SPV_EXT_demote_to_helper_invocation",
      "SPV_EXT_descriptor_indexing",
      "SPV_NV_fragment_shader_barycentric",
      "SPV_NV_compute_shader_derivatives",
      "SPV_NV_shader_image_footprint",
      "SPV_NV_shading_rate",
      "SPV_NV_mesh_shader",
      "SPV_NV_ray_tracing",
      "SPV_KHR_ray_tracing",
      "SPV_KHR_ray_query",
      "SPV_EXT_fragment_invocation_density",
      "SPV_EXT_physical_storage_buffer",
      "SPV_KHR_terminate_invocation",
      "SPV_KHR_subgroup_uniform_control_flow",
      "SPV_KHR_integer_dot_product",
      "SPV_EXT_shader_image_int64",
      "SPV_KHR_non_semantic_info",
      "SPV_KHR_uniform_group_instructions",
      "SPV_KHR_fragment_shader_barycentric",
  });
}

}  // namespace opt
}  // namespace spvtools
