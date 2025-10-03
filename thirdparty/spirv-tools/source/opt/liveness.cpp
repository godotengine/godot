// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#include "source/opt/liveness.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {
constexpr uint32_t kDecorationLocationInIdx = 2;
constexpr uint32_t kOpDecorateMemberMemberInIdx = 1;
constexpr uint32_t kOpDecorateMemberLocationInIdx = 3;
constexpr uint32_t kOpDecorateBuiltInLiteralInIdx = 2;
constexpr uint32_t kOpDecorateMemberBuiltInLiteralInIdx = 3;
}  // namespace

LivenessManager::LivenessManager(IRContext* ctx) : ctx_(ctx), computed_(false) {
  // Liveness sets computed when queried
}

void LivenessManager::InitializeAnalysis() {
  live_locs_.clear();
  live_builtins_.clear();
  // Mark all builtins live for frag shader.
  if (context()->GetStage() == spv::ExecutionModel::Fragment) {
    live_builtins_.insert(uint32_t(spv::BuiltIn::PointSize));
    live_builtins_.insert(uint32_t(spv::BuiltIn::ClipDistance));
    live_builtins_.insert(uint32_t(spv::BuiltIn::CullDistance));
  }
}

bool LivenessManager::IsAnalyzedBuiltin(uint32_t bi) {
  // There are only three builtins that can be analyzed and removed between
  // two stages: PointSize, ClipDistance and CullDistance. All others are
  // always consumed implicitly by the downstream stage.
  const auto builtin = spv::BuiltIn(bi);
  return builtin == spv::BuiltIn::PointSize ||
         builtin == spv::BuiltIn::ClipDistance ||
         builtin == spv::BuiltIn::CullDistance;
}

bool LivenessManager::AnalyzeBuiltIn(uint32_t id) {
  auto deco_mgr = context()->get_decoration_mgr();
  bool saw_builtin = false;
  // Analyze all builtin decorations of |id|.
  (void)deco_mgr->ForEachDecoration(
      id, uint32_t(spv::Decoration::BuiltIn),
      [this, &saw_builtin](const Instruction& deco_inst) {
        saw_builtin = true;
        // No need to process builtins in frag shader. All assumed used.
        if (context()->GetStage() == spv::ExecutionModel::Fragment) return;
        uint32_t builtin = uint32_t(spv::BuiltIn::Max);
        if (deco_inst.opcode() == spv::Op::OpDecorate)
          builtin =
              deco_inst.GetSingleWordInOperand(kOpDecorateBuiltInLiteralInIdx);
        else if (deco_inst.opcode() == spv::Op::OpMemberDecorate)
          builtin = deco_inst.GetSingleWordInOperand(
              kOpDecorateMemberBuiltInLiteralInIdx);
        else
          assert(false && "unexpected decoration");
        if (IsAnalyzedBuiltin(builtin)) live_builtins_.insert(builtin);
      });
  return saw_builtin;
}

void LivenessManager::MarkLocsLive(uint32_t start, uint32_t count) {
  auto finish = start + count;
  for (uint32_t u = start; u < finish; ++u) {
    live_locs_.insert(u);
  }
}

uint32_t LivenessManager::GetLocSize(const analysis::Type* type) const {
  auto arr_type = type->AsArray();
  if (arr_type) {
    auto comp_type = arr_type->element_type();
    auto len_info = arr_type->length_info();
    assert(len_info.words[0] == analysis::Array::LengthInfo::kConstant &&
           "unexpected array length");
    auto comp_len = len_info.words[1];
    return comp_len * GetLocSize(comp_type);
  }
  auto struct_type = type->AsStruct();
  if (struct_type) {
    uint32_t size = 0u;
    for (auto& el_type : struct_type->element_types())
      size += GetLocSize(el_type);
    return size;
  }
  auto mat_type = type->AsMatrix();
  if (mat_type) {
    auto cnt = mat_type->element_count();
    auto comp_type = mat_type->element_type();
    return cnt * GetLocSize(comp_type);
  }
  auto vec_type = type->AsVector();
  if (vec_type) {
    auto comp_type = vec_type->element_type();
    if (comp_type->AsInteger()) return 1;
    auto float_type = comp_type->AsFloat();
    assert(float_type && "unexpected vector component type");
    auto width = float_type->width();
    if (width == 32 || width == 16) return 1;
    assert(width == 64 && "unexpected float type width");
    auto comp_cnt = vec_type->element_count();
    return (comp_cnt > 2) ? 2 : 1;
  }
  assert((type->AsInteger() || type->AsFloat()) && "unexpected input type");
  return 1;
}

const analysis::Type* LivenessManager::GetComponentType(
    uint32_t index, const analysis::Type* agg_type) const {
  auto arr_type = agg_type->AsArray();
  if (arr_type) return arr_type->element_type();
  auto struct_type = agg_type->AsStruct();
  if (struct_type) return struct_type->element_types()[index];
  auto mat_type = agg_type->AsMatrix();
  if (mat_type) return mat_type->element_type();
  auto vec_type = agg_type->AsVector();
  assert(vec_type && "unexpected non-aggregate type");
  return vec_type->element_type();
}

uint32_t LivenessManager::GetLocOffset(uint32_t index,
                                       const analysis::Type* agg_type) const {
  auto arr_type = agg_type->AsArray();
  if (arr_type) return index * GetLocSize(arr_type->element_type());
  auto struct_type = agg_type->AsStruct();
  if (struct_type) {
    uint32_t offset = 0u;
    uint32_t cnt = 0u;
    for (auto& el_type : struct_type->element_types()) {
      if (cnt == index) break;
      offset += GetLocSize(el_type);
      ++cnt;
    }
    return offset;
  }
  auto mat_type = agg_type->AsMatrix();
  if (mat_type) return index * GetLocSize(mat_type->element_type());
  auto vec_type = agg_type->AsVector();
  assert(vec_type && "unexpected non-aggregate type");
  auto comp_type = vec_type->element_type();
  auto flt_type = comp_type->AsFloat();
  if (flt_type && flt_type->width() == 64u && index >= 2u) return 1;
  return 0;
}

void LivenessManager::AnalyzeAccessChainLoc(const Instruction* ac,
                                            const analysis::Type** curr_type,
                                            uint32_t* offset, bool* no_loc,
                                            bool is_patch, bool input) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // For tesc, tese and geom input variables, and tesc output variables,
  // first array index does not contribute to offset.
  auto stage = context()->GetStage();
  bool skip_first_index = false;
  if ((input && (stage == spv::ExecutionModel::TessellationControl ||
                 stage == spv::ExecutionModel::TessellationEvaluation ||
                 stage == spv::ExecutionModel::Geometry)) ||
      (!input && stage == spv::ExecutionModel::TessellationControl))
    skip_first_index = !is_patch;
  uint32_t ocnt = 0;
  ac->WhileEachInOperand([this, &ocnt, def_use_mgr, type_mgr, deco_mgr,
                          curr_type, offset, no_loc,
                          skip_first_index](const uint32_t* opnd) {
    if (ocnt >= 1) {
      // Skip first index's contribution to offset if indicated
      if (ocnt == 1 && skip_first_index) {
        auto arr_type = (*curr_type)->AsArray();
        assert(arr_type && "unexpected wrapper type");
        *curr_type = arr_type->element_type();
        ocnt++;
        return true;
      }
      // If any non-constant index, mark the entire current object and return.
      auto idx_inst = def_use_mgr->GetDef(*opnd);
      if (idx_inst->opcode() != spv::Op::OpConstant) return false;
      // If current type is struct, look for location decoration on member and
      // reset offset if found.
      auto index = idx_inst->GetSingleWordInOperand(0);
      auto str_type = (*curr_type)->AsStruct();
      if (str_type) {
        uint32_t loc = 0;
        auto str_type_id = type_mgr->GetId(str_type);
        bool no_mem_loc = deco_mgr->WhileEachDecoration(
            str_type_id, uint32_t(spv::Decoration::Location),
            [&loc, index, no_loc](const Instruction& deco) {
              assert(deco.opcode() == spv::Op::OpMemberDecorate &&
                     "unexpected decoration");
              if (deco.GetSingleWordInOperand(kOpDecorateMemberMemberInIdx) ==
                  index) {
                loc =
                    deco.GetSingleWordInOperand(kOpDecorateMemberLocationInIdx);
                *no_loc = false;
                return false;
              }
              return true;
            });
        if (!no_mem_loc) {
          *offset = loc;
          *curr_type = GetComponentType(index, *curr_type);
          ocnt++;
          return true;
        }
      }

      // Update offset and current type based on constant index.
      *offset += GetLocOffset(index, *curr_type);
      *curr_type = GetComponentType(index, *curr_type);
    }
    ocnt++;
    return true;
  });
}

void LivenessManager::MarkRefLive(const Instruction* ref, Instruction* var) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Find variable location if present.
  uint32_t loc = 0;
  auto var_id = var->result_id();
  bool no_loc = deco_mgr->WhileEachDecoration(
      var_id, uint32_t(spv::Decoration::Location),
      [&loc](const Instruction& deco) {
        assert(deco.opcode() == spv::Op::OpDecorate && "unexpected decoration");
        loc = deco.GetSingleWordInOperand(kDecorationLocationInIdx);
        return false;
      });
  // Find patch decoration if present
  bool is_patch = !deco_mgr->WhileEachDecoration(
      var_id, uint32_t(spv::Decoration::Patch), [](const Instruction& deco) {
        if (deco.opcode() != spv::Op::OpDecorate)
          assert(false && "unexpected decoration");
        return false;
      });
  // If use is a load, mark all locations of var
  auto ptr_type = type_mgr->GetType(var->type_id())->AsPointer();
  assert(ptr_type && "unexpected var type");
  auto var_type = ptr_type->pointee_type();
  if (ref->opcode() == spv::Op::OpLoad) {
    assert(!no_loc && "missing input variable location");
    MarkLocsLive(loc, GetLocSize(var_type));
    return;
  }
  // Mark just those locations indicated by access chain
  assert((ref->opcode() == spv::Op::OpAccessChain ||
          ref->opcode() == spv::Op::OpInBoundsAccessChain) &&
         "unexpected use of input variable");
  // Traverse access chain, compute location offset and type of reference
  // through constant indices and mark those locs live. Assert if no location
  // found.
  uint32_t offset = loc;
  auto curr_type = var_type;
  AnalyzeAccessChainLoc(ref, &curr_type, &offset, &no_loc, is_patch);
  assert(!no_loc && "missing input variable location");
  MarkLocsLive(offset, GetLocSize(curr_type));
}

void LivenessManager::ComputeLiveness() {
  InitializeAnalysis();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  // Process all input variables
  for (auto& var : context()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable) {
      continue;
    }
    analysis::Type* var_type = type_mgr->GetType(var.type_id());
    analysis::Pointer* ptr_type = var_type->AsPointer();
    if (ptr_type->storage_class() != spv::StorageClass::Input) {
      continue;
    }
    // If var is builtin, mark live if analyzed and continue to next variable
    auto var_id = var.result_id();
    if (AnalyzeBuiltIn(var_id)) continue;
    // If interface block with builtin members, mark live if analyzed and
    // continue to next variable. Input interface blocks will only appear
    // in tesc, tese and geom shaders. Will need to strip off one level of
    // arrayness to get to block type.
    auto pte_type = ptr_type->pointee_type();
    auto arr_type = pte_type->AsArray();
    if (arr_type) {
      auto elt_type = arr_type->element_type();
      auto str_type = elt_type->AsStruct();
      if (str_type) {
        auto str_type_id = type_mgr->GetId(str_type);
        if (AnalyzeBuiltIn(str_type_id)) continue;
      }
    }
    // Mark all used locations of var live
    def_use_mgr->ForEachUser(var_id, [this, &var](Instruction* user) {
      auto op = user->opcode();
      if (op == spv::Op::OpEntryPoint || op == spv::Op::OpName ||
          op == spv::Op::OpDecorate) {
        return;
      }
      MarkRefLive(user, &var);
    });
  }
}

void LivenessManager::GetLiveness(std::unordered_set<uint32_t>* live_locs,
                                  std::unordered_set<uint32_t>* live_builtins) {
  if (!computed_) {
    ComputeLiveness();
    computed_ = true;
  }
  *live_locs = live_locs_;
  *live_builtins = live_builtins_;
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
