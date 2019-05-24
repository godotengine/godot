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

namespace spvtools {
namespace opt {
namespace {

const uint32_t kStoreValIdInIdx = 1;

}  // anonymous namespace

bool LocalSingleBlockLoadStoreElimPass::HasOnlySupportedRefs(uint32_t ptrId) {
  if (supported_ref_ptrs_.find(ptrId) != supported_ref_ptrs_.end()) return true;
  if (get_def_use_mgr()->WhileEachUser(ptrId, [this](Instruction* user) {
        SpvOp op = user->opcode();
        if (IsNonPtrAccessChain(op) || op == SpvOpCopyObject) {
          if (!HasOnlySupportedRefs(user->result_id())) {
            return false;
          }
        } else if (op != SpvOpStore && op != SpvOpLoad && op != SpvOpName &&
                   !IsNonTypeDecorate(op)) {
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
        case SpvOpStore: {
          // Verify store variable is target type
          uint32_t varId;
          Instruction* ptrInst = GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) continue;
          if (!HasOnlySupportedRefs(varId)) continue;
          // If a store to the whole variable, remember it for succeeding
          // loads and stores. Otherwise forget any previous store to that
          // variable.
          if (ptrInst->opcode() == SpvOpVariable) {
            // If a previous store to same variable, mark the store
            // for deletion if not still used.
            auto prev_store = var2store_.find(varId);
            if (prev_store != var2store_.end() &&
                instructions_to_save.count(prev_store->second) == 0) {
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
        case SpvOpLoad: {
          // Verify store variable is target type
          uint32_t varId;
          Instruction* ptrInst = GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) continue;
          if (!HasOnlySupportedRefs(varId)) continue;
          uint32_t replId = 0;
          if (ptrInst->opcode() == SpvOpVariable) {
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
            if (ptrInst->opcode() == SpvOpVariable)
              var2load_[varId] = &*ii;  // register load
          }
        } break;
        case SpvOpFunctionCall: {
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

  // Initialize extensions whitelist
  InitExtensions();
}

bool LocalSingleBlockLoadStoreElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status LocalSingleBlockLoadStoreElimPass::ProcessImpl() {
  // Assumes relaxed logical addressing only (see instruction.h).
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;

  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == SpvOpGroupDecorate) return Status::SuccessWithoutChange;
  // If any extensions in the module are not explicitly supported,
  // return unmodified.
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process all entry point functions
  ProcessFunction pfn = [this](Function* fp) {
    return LocalSingleBlockLoadStoreElim(fp);
  };

  bool modified = context()->ProcessEntryPointCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalSingleBlockLoadStoreElimPass::LocalSingleBlockLoadStoreElimPass() =
    default;

Pass::Status LocalSingleBlockLoadStoreElimPass::Process() {
  Initialize();
  return ProcessImpl();
}

void LocalSingleBlockLoadStoreElimPass::InitExtensions() {
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
