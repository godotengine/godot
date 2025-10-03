// Copyright (c) 2017 Google Inc.
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

#include "source/opt/ir_context.h"

#include <cstring>

#include "OpenCLDebugInfo100.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/opt/log.h"
#include "source/opt/mem_pass.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {
namespace {
constexpr int kSpvDecorateTargetIdInIdx = 0;
constexpr int kSpvDecorateDecorationInIdx = 1;
constexpr int kSpvDecorateBuiltinInIdx = 2;
constexpr int kEntryPointInterfaceInIdx = 3;
constexpr int kEntryPointFunctionIdInIdx = 1;
constexpr int kEntryPointExecutionModelInIdx = 0;

// Constants for OpenCL.DebugInfo.100 / NonSemantic.Shader.DebugInfo.100
// extension instructions.
constexpr uint32_t kDebugFunctionOperandFunctionIndex = 13;
constexpr uint32_t kDebugGlobalVariableOperandVariableIndex = 11;
}  // namespace

void IRContext::BuildInvalidAnalyses(IRContext::Analysis set) {
  set = Analysis(set & ~valid_analyses_);

  if (set & kAnalysisDefUse) {
    BuildDefUseManager();
  }
  if (set & kAnalysisInstrToBlockMapping) {
    BuildInstrToBlockMapping();
  }
  if (set & kAnalysisDecorations) {
    BuildDecorationManager();
  }
  if (set & kAnalysisCFG) {
    BuildCFG();
  }
  if (set & kAnalysisDominatorAnalysis) {
    ResetDominatorAnalysis();
  }
  if (set & kAnalysisLoopAnalysis) {
    ResetLoopAnalysis();
  }
  if (set & kAnalysisBuiltinVarId) {
    ResetBuiltinAnalysis();
  }
  if (set & kAnalysisNameMap) {
    BuildIdToNameMap();
  }
  if (set & kAnalysisScalarEvolution) {
    BuildScalarEvolutionAnalysis();
  }
  if (set & kAnalysisRegisterPressure) {
    BuildRegPressureAnalysis();
  }
  if (set & kAnalysisValueNumberTable) {
    BuildValueNumberTable();
  }
  if (set & kAnalysisStructuredCFG) {
    BuildStructuredCFGAnalysis();
  }
  if (set & kAnalysisIdToFuncMapping) {
    BuildIdToFuncMapping();
  }
  if (set & kAnalysisConstants) {
    BuildConstantManager();
  }
  if (set & kAnalysisTypes) {
    BuildTypeManager();
  }
  if (set & kAnalysisDebugInfo) {
    BuildDebugInfoManager();
  }
}

void IRContext::InvalidateAnalysesExceptFor(
    IRContext::Analysis preserved_analyses) {
  uint32_t analyses_to_invalidate = valid_analyses_ & (~preserved_analyses);
  InvalidateAnalyses(static_cast<IRContext::Analysis>(analyses_to_invalidate));
}

void IRContext::InvalidateAnalyses(IRContext::Analysis analyses_to_invalidate) {
  // The ConstantManager and DebugInfoManager contain Type pointers. If the
  // TypeManager goes away, the ConstantManager and DebugInfoManager have to
  // go away.
  if (analyses_to_invalidate & kAnalysisTypes) {
    analyses_to_invalidate |= kAnalysisConstants;
    analyses_to_invalidate |= kAnalysisDebugInfo;
  }

  // The dominator analysis hold the pseudo entry and exit nodes from the CFG.
  // Also if the CFG change the dominators many changed as well, so the
  // dominator analysis should be invalidated as well.
  if (analyses_to_invalidate & kAnalysisCFG) {
    analyses_to_invalidate |= kAnalysisDominatorAnalysis;
  }

  if (analyses_to_invalidate & kAnalysisDefUse) {
    def_use_mgr_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisInstrToBlockMapping) {
    instr_to_block_.clear();
  }
  if (analyses_to_invalidate & kAnalysisDecorations) {
    decoration_mgr_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisCombinators) {
    combinator_ops_.clear();
  }
  if (analyses_to_invalidate & kAnalysisBuiltinVarId) {
    builtin_var_id_map_.clear();
  }
  if (analyses_to_invalidate & kAnalysisCFG) {
    cfg_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisDominatorAnalysis) {
    dominator_trees_.clear();
    post_dominator_trees_.clear();
  }
  if (analyses_to_invalidate & kAnalysisNameMap) {
    id_to_name_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisValueNumberTable) {
    vn_table_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisStructuredCFG) {
    struct_cfg_analysis_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisIdToFuncMapping) {
    id_to_func_.clear();
  }
  if (analyses_to_invalidate & kAnalysisConstants) {
    constant_mgr_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisLiveness) {
    liveness_mgr_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisTypes) {
    type_mgr_.reset(nullptr);
  }

  if (analyses_to_invalidate & kAnalysisDebugInfo) {
    debug_info_mgr_.reset(nullptr);
  }

  valid_analyses_ = Analysis(valid_analyses_ & ~analyses_to_invalidate);
}

Instruction* IRContext::KillInst(Instruction* inst) {
  if (!inst) {
    return nullptr;
  }

  KillNamesAndDecorates(inst);

  KillOperandFromDebugInstructions(inst);

  if (AreAnalysesValid(kAnalysisDefUse)) {
    analysis::DefUseManager* def_use_mgr = get_def_use_mgr();
    def_use_mgr->ClearInst(inst);
    for (auto& l_inst : inst->dbg_line_insts()) def_use_mgr->ClearInst(&l_inst);
  }
  if (AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
    instr_to_block_.erase(inst);
  }
  if (AreAnalysesValid(kAnalysisDecorations)) {
    if (inst->IsDecoration()) {
      decoration_mgr_->RemoveDecoration(inst);
    }
  }
  if (AreAnalysesValid(kAnalysisDebugInfo)) {
    get_debug_info_mgr()->ClearDebugScopeAndInlinedAtUses(inst);
    get_debug_info_mgr()->ClearDebugInfo(inst);
  }
  if (type_mgr_ && IsTypeInst(inst->opcode())) {
    type_mgr_->RemoveId(inst->result_id());
  }
  if (constant_mgr_ && IsConstantInst(inst->opcode())) {
    constant_mgr_->RemoveId(inst->result_id());
  }
  if (inst->opcode() == spv::Op::OpCapability ||
      inst->opcode() == spv::Op::OpExtension) {
    // We reset the feature manager, instead of updating it, because it is just
    // as much work.  We would have to remove all capabilities implied by this
    // capability that are not also implied by the remaining OpCapability
    // instructions. We could update extensions, but we will see if it is
    // needed.
    ResetFeatureManager();
  }

  RemoveFromIdToName(inst);

  Instruction* next_instruction = nullptr;
  if (inst->IsInAList()) {
    next_instruction = inst->NextNode();
    inst->RemoveFromList();
    delete inst;
  } else {
    // Needed for instructions that are not part of a list like OpLabels,
    // OpFunction, OpFunctionEnd, etc..
    inst->ToNop();
  }
  return next_instruction;
}

void IRContext::CollectNonSemanticTree(
    Instruction* inst, std::unordered_set<Instruction*>* to_kill) {
  if (!inst->HasResultId()) return;
  // Debug[No]Line result id is not used, so we are done
  if (inst->IsDebugLineInst()) return;
  std::vector<Instruction*> work_list;
  std::unordered_set<Instruction*> seen;
  work_list.push_back(inst);

  while (!work_list.empty()) {
    auto* i = work_list.back();
    work_list.pop_back();
    get_def_use_mgr()->ForEachUser(
        i, [&work_list, to_kill, &seen](Instruction* user) {
          if (user->IsNonSemanticInstruction() && seen.insert(user).second) {
            work_list.push_back(user);
            to_kill->insert(user);
          }
        });
  }
}

bool IRContext::KillDef(uint32_t id) {
  Instruction* def = get_def_use_mgr()->GetDef(id);
  if (def != nullptr) {
    KillInst(def);
    return true;
  }
  return false;
}

bool IRContext::ReplaceAllUsesWith(uint32_t before, uint32_t after) {
  return ReplaceAllUsesWithPredicate(before, after,
                                     [](Instruction*) { return true; });
}

bool IRContext::ReplaceAllUsesWithPredicate(
    uint32_t before, uint32_t after,
    const std::function<bool(Instruction*)>& predicate) {
  if (before == after) return false;

  if (AreAnalysesValid(kAnalysisDebugInfo)) {
    get_debug_info_mgr()->ReplaceAllUsesInDebugScopeWithPredicate(before, after,
                                                                  predicate);
  }

  // Ensure that |after| has been registered as def.
  assert(get_def_use_mgr()->GetDef(after) &&
         "'after' is not a registered def.");

  std::vector<std::pair<Instruction*, uint32_t>> uses_to_update;
  get_def_use_mgr()->ForEachUse(
      before, [&predicate, &uses_to_update](Instruction* user, uint32_t index) {
        if (predicate(user)) {
          uses_to_update.emplace_back(user, index);
        }
      });

  Instruction* prev = nullptr;
  for (auto p : uses_to_update) {
    Instruction* user = p.first;
    uint32_t index = p.second;
    if (prev == nullptr || prev != user) {
      ForgetUses(user);
      prev = user;
    }
    const uint32_t type_result_id_count =
        (user->result_id() != 0) + (user->type_id() != 0);

    if (index < type_result_id_count) {
      // Update the type_id. Note that result id is immutable so it should
      // never be updated.
      if (user->type_id() != 0 && index == 0) {
        user->SetResultType(after);
      } else if (user->type_id() == 0) {
        SPIRV_ASSERT(consumer_, false,
                     "Result type id considered as use while the instruction "
                     "doesn't have a result type id.");
        (void)consumer_;  // Makes the compiler happy for release build.
      } else {
        SPIRV_ASSERT(consumer_, false,
                     "Trying setting the immutable result id.");
      }
    } else {
      // Update an in-operand.
      uint32_t in_operand_pos = index - type_result_id_count;
      // Make the modification in the instruction.
      user->SetInOperand(in_operand_pos, {after});
    }
    AnalyzeUses(user);
  }
  return true;
}

bool IRContext::IsConsistent() {
#ifndef SPIRV_CHECK_CONTEXT
  return true;
#else
  if (AreAnalysesValid(kAnalysisDefUse)) {
    analysis::DefUseManager new_def_use(module());
    if (!CompareAndPrintDifferences(*get_def_use_mgr(), new_def_use)) {
      return false;
    }
  }

  if (AreAnalysesValid(kAnalysisIdToFuncMapping)) {
    for (auto& fn : *module_) {
      if (id_to_func_[fn.result_id()] != &fn) {
        return false;
      }
    }
  }

  if (AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
    for (auto& func : *module()) {
      for (auto& block : func) {
        if (!block.WhileEachInst([this, &block](Instruction* inst) {
              if (get_instr_block(inst) != &block) {
                return false;
              }
              return true;
            }))
          return false;
      }
    }
  }

  if (!CheckCFG()) {
    return false;
  }

  if (AreAnalysesValid(kAnalysisDecorations)) {
    analysis::DecorationManager* dec_mgr = get_decoration_mgr();
    analysis::DecorationManager current(module());

    if (*dec_mgr != current) {
      return false;
    }
  }

  if (feature_mgr_ != nullptr) {
    FeatureManager current(grammar_);
    current.Analyze(module());

    if (current != *feature_mgr_) {
      return false;
    }
  }
  return true;
#endif
}

void IRContext::ForgetUses(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->EraseUseRecordsOfOperandIds(inst);
  }
  if (AreAnalysesValid(kAnalysisDecorations)) {
    if (inst->IsDecoration()) {
      get_decoration_mgr()->RemoveDecoration(inst);
    }
  }
  if (AreAnalysesValid(kAnalysisDebugInfo)) {
    get_debug_info_mgr()->ClearDebugInfo(inst);
  }
  RemoveFromIdToName(inst);
}

void IRContext::AnalyzeUses(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstUse(inst);
  }
  if (AreAnalysesValid(kAnalysisDecorations)) {
    if (inst->IsDecoration()) {
      get_decoration_mgr()->AddDecoration(inst);
    }
  }
  if (AreAnalysesValid(kAnalysisDebugInfo)) {
    get_debug_info_mgr()->AnalyzeDebugInst(inst);
  }
  if (id_to_name_ && (inst->opcode() == spv::Op::OpName ||
                      inst->opcode() == spv::Op::OpMemberName)) {
    id_to_name_->insert({inst->GetSingleWordInOperand(0), inst});
  }
}

void IRContext::KillNamesAndDecorates(uint32_t id) {
  analysis::DecorationManager* dec_mgr = get_decoration_mgr();
  dec_mgr->RemoveDecorationsFrom(id);

  std::vector<Instruction*> name_to_kill;
  for (auto name : GetNames(id)) {
    name_to_kill.push_back(name.second);
  }
  for (Instruction* name_inst : name_to_kill) {
    KillInst(name_inst);
  }
}

void IRContext::KillNamesAndDecorates(Instruction* inst) {
  const uint32_t rId = inst->result_id();
  if (rId == 0) return;
  KillNamesAndDecorates(rId);
}

void IRContext::KillOperandFromDebugInstructions(Instruction* inst) {
  const auto opcode = inst->opcode();
  const uint32_t id = inst->result_id();
  // Kill id of OpFunction from DebugFunction.
  if (opcode == spv::Op::OpFunction) {
    for (auto it = module()->ext_inst_debuginfo_begin();
         it != module()->ext_inst_debuginfo_end(); ++it) {
      if (it->GetOpenCL100DebugOpcode() != OpenCLDebugInfo100DebugFunction)
        continue;
      auto& operand = it->GetOperand(kDebugFunctionOperandFunctionIndex);
      if (operand.words[0] == id) {
        operand.words[0] =
            get_debug_info_mgr()->GetDebugInfoNone()->result_id();
        get_def_use_mgr()->AnalyzeInstUse(&*it);
      }
    }
  }
  // Kill id of OpVariable for global variable from DebugGlobalVariable.
  if (opcode == spv::Op::OpVariable || IsConstantInst(opcode)) {
    for (auto it = module()->ext_inst_debuginfo_begin();
         it != module()->ext_inst_debuginfo_end(); ++it) {
      if (it->GetCommonDebugOpcode() != CommonDebugInfoDebugGlobalVariable)
        continue;
      auto& operand = it->GetOperand(kDebugGlobalVariableOperandVariableIndex);
      if (operand.words[0] == id) {
        operand.words[0] =
            get_debug_info_mgr()->GetDebugInfoNone()->result_id();
        get_def_use_mgr()->AnalyzeInstUse(&*it);
      }
    }
  }
}

void IRContext::AddCombinatorsForCapability(uint32_t capability) {
  spv::Capability cap = spv::Capability(capability);
  if (cap == spv::Capability::Shader) {
    combinator_ops_[0].insert(
        {(uint32_t)spv::Op::OpNop,
         (uint32_t)spv::Op::OpUndef,
         (uint32_t)spv::Op::OpConstant,
         (uint32_t)spv::Op::OpConstantTrue,
         (uint32_t)spv::Op::OpConstantFalse,
         (uint32_t)spv::Op::OpConstantComposite,
         (uint32_t)spv::Op::OpConstantSampler,
         (uint32_t)spv::Op::OpConstantNull,
         (uint32_t)spv::Op::OpTypeVoid,
         (uint32_t)spv::Op::OpTypeBool,
         (uint32_t)spv::Op::OpTypeInt,
         (uint32_t)spv::Op::OpTypeFloat,
         (uint32_t)spv::Op::OpTypeVector,
         (uint32_t)spv::Op::OpTypeMatrix,
         (uint32_t)spv::Op::OpTypeImage,
         (uint32_t)spv::Op::OpTypeSampler,
         (uint32_t)spv::Op::OpTypeSampledImage,
         (uint32_t)spv::Op::OpTypeAccelerationStructureNV,
         (uint32_t)spv::Op::OpTypeAccelerationStructureKHR,
         (uint32_t)spv::Op::OpTypeRayQueryKHR,
         (uint32_t)spv::Op::OpTypeHitObjectNV,
         (uint32_t)spv::Op::OpTypeArray,
         (uint32_t)spv::Op::OpTypeRuntimeArray,
         (uint32_t)spv::Op::OpTypeStruct,
         (uint32_t)spv::Op::OpTypeOpaque,
         (uint32_t)spv::Op::OpTypePointer,
         (uint32_t)spv::Op::OpTypeFunction,
         (uint32_t)spv::Op::OpTypeEvent,
         (uint32_t)spv::Op::OpTypeDeviceEvent,
         (uint32_t)spv::Op::OpTypeReserveId,
         (uint32_t)spv::Op::OpTypeQueue,
         (uint32_t)spv::Op::OpTypePipe,
         (uint32_t)spv::Op::OpTypeForwardPointer,
         (uint32_t)spv::Op::OpVariable,
         (uint32_t)spv::Op::OpImageTexelPointer,
         (uint32_t)spv::Op::OpLoad,
         (uint32_t)spv::Op::OpAccessChain,
         (uint32_t)spv::Op::OpInBoundsAccessChain,
         (uint32_t)spv::Op::OpArrayLength,
         (uint32_t)spv::Op::OpVectorExtractDynamic,
         (uint32_t)spv::Op::OpVectorInsertDynamic,
         (uint32_t)spv::Op::OpVectorShuffle,
         (uint32_t)spv::Op::OpCompositeConstruct,
         (uint32_t)spv::Op::OpCompositeExtract,
         (uint32_t)spv::Op::OpCompositeInsert,
         (uint32_t)spv::Op::OpCopyObject,
         (uint32_t)spv::Op::OpTranspose,
         (uint32_t)spv::Op::OpSampledImage,
         (uint32_t)spv::Op::OpImageSampleImplicitLod,
         (uint32_t)spv::Op::OpImageSampleExplicitLod,
         (uint32_t)spv::Op::OpImageSampleDrefImplicitLod,
         (uint32_t)spv::Op::OpImageSampleDrefExplicitLod,
         (uint32_t)spv::Op::OpImageSampleProjImplicitLod,
         (uint32_t)spv::Op::OpImageSampleProjExplicitLod,
         (uint32_t)spv::Op::OpImageSampleProjDrefImplicitLod,
         (uint32_t)spv::Op::OpImageSampleProjDrefExplicitLod,
         (uint32_t)spv::Op::OpImageFetch,
         (uint32_t)spv::Op::OpImageGather,
         (uint32_t)spv::Op::OpImageDrefGather,
         (uint32_t)spv::Op::OpImageRead,
         (uint32_t)spv::Op::OpImage,
         (uint32_t)spv::Op::OpImageQueryFormat,
         (uint32_t)spv::Op::OpImageQueryOrder,
         (uint32_t)spv::Op::OpImageQuerySizeLod,
         (uint32_t)spv::Op::OpImageQuerySize,
         (uint32_t)spv::Op::OpImageQueryLevels,
         (uint32_t)spv::Op::OpImageQuerySamples,
         (uint32_t)spv::Op::OpConvertFToU,
         (uint32_t)spv::Op::OpConvertFToS,
         (uint32_t)spv::Op::OpConvertSToF,
         (uint32_t)spv::Op::OpConvertUToF,
         (uint32_t)spv::Op::OpUConvert,
         (uint32_t)spv::Op::OpSConvert,
         (uint32_t)spv::Op::OpFConvert,
         (uint32_t)spv::Op::OpQuantizeToF16,
         (uint32_t)spv::Op::OpBitcast,
         (uint32_t)spv::Op::OpSNegate,
         (uint32_t)spv::Op::OpFNegate,
         (uint32_t)spv::Op::OpIAdd,
         (uint32_t)spv::Op::OpFAdd,
         (uint32_t)spv::Op::OpISub,
         (uint32_t)spv::Op::OpFSub,
         (uint32_t)spv::Op::OpIMul,
         (uint32_t)spv::Op::OpFMul,
         (uint32_t)spv::Op::OpUDiv,
         (uint32_t)spv::Op::OpSDiv,
         (uint32_t)spv::Op::OpFDiv,
         (uint32_t)spv::Op::OpUMod,
         (uint32_t)spv::Op::OpSRem,
         (uint32_t)spv::Op::OpSMod,
         (uint32_t)spv::Op::OpFRem,
         (uint32_t)spv::Op::OpFMod,
         (uint32_t)spv::Op::OpVectorTimesScalar,
         (uint32_t)spv::Op::OpMatrixTimesScalar,
         (uint32_t)spv::Op::OpVectorTimesMatrix,
         (uint32_t)spv::Op::OpMatrixTimesVector,
         (uint32_t)spv::Op::OpMatrixTimesMatrix,
         (uint32_t)spv::Op::OpOuterProduct,
         (uint32_t)spv::Op::OpDot,
         (uint32_t)spv::Op::OpIAddCarry,
         (uint32_t)spv::Op::OpISubBorrow,
         (uint32_t)spv::Op::OpUMulExtended,
         (uint32_t)spv::Op::OpSMulExtended,
         (uint32_t)spv::Op::OpAny,
         (uint32_t)spv::Op::OpAll,
         (uint32_t)spv::Op::OpIsNan,
         (uint32_t)spv::Op::OpIsInf,
         (uint32_t)spv::Op::OpLogicalEqual,
         (uint32_t)spv::Op::OpLogicalNotEqual,
         (uint32_t)spv::Op::OpLogicalOr,
         (uint32_t)spv::Op::OpLogicalAnd,
         (uint32_t)spv::Op::OpLogicalNot,
         (uint32_t)spv::Op::OpSelect,
         (uint32_t)spv::Op::OpIEqual,
         (uint32_t)spv::Op::OpINotEqual,
         (uint32_t)spv::Op::OpUGreaterThan,
         (uint32_t)spv::Op::OpSGreaterThan,
         (uint32_t)spv::Op::OpUGreaterThanEqual,
         (uint32_t)spv::Op::OpSGreaterThanEqual,
         (uint32_t)spv::Op::OpULessThan,
         (uint32_t)spv::Op::OpSLessThan,
         (uint32_t)spv::Op::OpULessThanEqual,
         (uint32_t)spv::Op::OpSLessThanEqual,
         (uint32_t)spv::Op::OpFOrdEqual,
         (uint32_t)spv::Op::OpFUnordEqual,
         (uint32_t)spv::Op::OpFOrdNotEqual,
         (uint32_t)spv::Op::OpFUnordNotEqual,
         (uint32_t)spv::Op::OpFOrdLessThan,
         (uint32_t)spv::Op::OpFUnordLessThan,
         (uint32_t)spv::Op::OpFOrdGreaterThan,
         (uint32_t)spv::Op::OpFUnordGreaterThan,
         (uint32_t)spv::Op::OpFOrdLessThanEqual,
         (uint32_t)spv::Op::OpFUnordLessThanEqual,
         (uint32_t)spv::Op::OpFOrdGreaterThanEqual,
         (uint32_t)spv::Op::OpFUnordGreaterThanEqual,
         (uint32_t)spv::Op::OpShiftRightLogical,
         (uint32_t)spv::Op::OpShiftRightArithmetic,
         (uint32_t)spv::Op::OpShiftLeftLogical,
         (uint32_t)spv::Op::OpBitwiseOr,
         (uint32_t)spv::Op::OpBitwiseXor,
         (uint32_t)spv::Op::OpBitwiseAnd,
         (uint32_t)spv::Op::OpNot,
         (uint32_t)spv::Op::OpBitFieldInsert,
         (uint32_t)spv::Op::OpBitFieldSExtract,
         (uint32_t)spv::Op::OpBitFieldUExtract,
         (uint32_t)spv::Op::OpBitReverse,
         (uint32_t)spv::Op::OpBitCount,
         (uint32_t)spv::Op::OpPhi,
         (uint32_t)spv::Op::OpImageSparseSampleImplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleExplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleDrefImplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleDrefExplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleProjImplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleProjExplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleProjDrefImplicitLod,
         (uint32_t)spv::Op::OpImageSparseSampleProjDrefExplicitLod,
         (uint32_t)spv::Op::OpImageSparseFetch,
         (uint32_t)spv::Op::OpImageSparseGather,
         (uint32_t)spv::Op::OpImageSparseDrefGather,
         (uint32_t)spv::Op::OpImageSparseTexelsResident,
         (uint32_t)spv::Op::OpImageSparseRead,
         (uint32_t)spv::Op::OpSizeOf});
  }
}

void IRContext::AddCombinatorsForExtension(Instruction* extension) {
  assert(extension->opcode() == spv::Op::OpExtInstImport &&
         "Expecting an import of an extension's instruction set.");
  const std::string extension_name = extension->GetInOperand(0).AsString();
  if (extension_name == "GLSL.std.450") {
    combinator_ops_[extension->result_id()] = {
        (uint32_t)GLSLstd450Round,
        (uint32_t)GLSLstd450RoundEven,
        (uint32_t)GLSLstd450Trunc,
        (uint32_t)GLSLstd450FAbs,
        (uint32_t)GLSLstd450SAbs,
        (uint32_t)GLSLstd450FSign,
        (uint32_t)GLSLstd450SSign,
        (uint32_t)GLSLstd450Floor,
        (uint32_t)GLSLstd450Ceil,
        (uint32_t)GLSLstd450Fract,
        (uint32_t)GLSLstd450Radians,
        (uint32_t)GLSLstd450Degrees,
        (uint32_t)GLSLstd450Sin,
        (uint32_t)GLSLstd450Cos,
        (uint32_t)GLSLstd450Tan,
        (uint32_t)GLSLstd450Asin,
        (uint32_t)GLSLstd450Acos,
        (uint32_t)GLSLstd450Atan,
        (uint32_t)GLSLstd450Sinh,
        (uint32_t)GLSLstd450Cosh,
        (uint32_t)GLSLstd450Tanh,
        (uint32_t)GLSLstd450Asinh,
        (uint32_t)GLSLstd450Acosh,
        (uint32_t)GLSLstd450Atanh,
        (uint32_t)GLSLstd450Atan2,
        (uint32_t)GLSLstd450Pow,
        (uint32_t)GLSLstd450Exp,
        (uint32_t)GLSLstd450Log,
        (uint32_t)GLSLstd450Exp2,
        (uint32_t)GLSLstd450Log2,
        (uint32_t)GLSLstd450Sqrt,
        (uint32_t)GLSLstd450InverseSqrt,
        (uint32_t)GLSLstd450Determinant,
        (uint32_t)GLSLstd450MatrixInverse,
        (uint32_t)GLSLstd450ModfStruct,
        (uint32_t)GLSLstd450FMin,
        (uint32_t)GLSLstd450UMin,
        (uint32_t)GLSLstd450SMin,
        (uint32_t)GLSLstd450FMax,
        (uint32_t)GLSLstd450UMax,
        (uint32_t)GLSLstd450SMax,
        (uint32_t)GLSLstd450FClamp,
        (uint32_t)GLSLstd450UClamp,
        (uint32_t)GLSLstd450SClamp,
        (uint32_t)GLSLstd450FMix,
        (uint32_t)GLSLstd450IMix,
        (uint32_t)GLSLstd450Step,
        (uint32_t)GLSLstd450SmoothStep,
        (uint32_t)GLSLstd450Fma,
        (uint32_t)GLSLstd450FrexpStruct,
        (uint32_t)GLSLstd450Ldexp,
        (uint32_t)GLSLstd450PackSnorm4x8,
        (uint32_t)GLSLstd450PackUnorm4x8,
        (uint32_t)GLSLstd450PackSnorm2x16,
        (uint32_t)GLSLstd450PackUnorm2x16,
        (uint32_t)GLSLstd450PackHalf2x16,
        (uint32_t)GLSLstd450PackDouble2x32,
        (uint32_t)GLSLstd450UnpackSnorm2x16,
        (uint32_t)GLSLstd450UnpackUnorm2x16,
        (uint32_t)GLSLstd450UnpackHalf2x16,
        (uint32_t)GLSLstd450UnpackSnorm4x8,
        (uint32_t)GLSLstd450UnpackUnorm4x8,
        (uint32_t)GLSLstd450UnpackDouble2x32,
        (uint32_t)GLSLstd450Length,
        (uint32_t)GLSLstd450Distance,
        (uint32_t)GLSLstd450Cross,
        (uint32_t)GLSLstd450Normalize,
        (uint32_t)GLSLstd450FaceForward,
        (uint32_t)GLSLstd450Reflect,
        (uint32_t)GLSLstd450Refract,
        (uint32_t)GLSLstd450FindILsb,
        (uint32_t)GLSLstd450FindSMsb,
        (uint32_t)GLSLstd450FindUMsb,
        (uint32_t)GLSLstd450InterpolateAtCentroid,
        (uint32_t)GLSLstd450InterpolateAtSample,
        (uint32_t)GLSLstd450InterpolateAtOffset,
        (uint32_t)GLSLstd450NMin,
        (uint32_t)GLSLstd450NMax,
        (uint32_t)GLSLstd450NClamp};
  } else {
    // Map the result id to the empty set.
    combinator_ops_[extension->result_id()];
  }
}

void IRContext::InitializeCombinators() {
  get_feature_mgr()->GetCapabilities()->ForEach([this](spv::Capability cap) {
    AddCombinatorsForCapability(uint32_t(cap));
  });

  for (auto& extension : module()->ext_inst_imports()) {
    AddCombinatorsForExtension(&extension);
  }

  valid_analyses_ |= kAnalysisCombinators;
}

void IRContext::RemoveFromIdToName(const Instruction* inst) {
  if (id_to_name_ && (inst->opcode() == spv::Op::OpName ||
                      inst->opcode() == spv::Op::OpMemberName)) {
    auto range = id_to_name_->equal_range(inst->GetSingleWordInOperand(0));
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == inst) {
        id_to_name_->erase(it);
        break;
      }
    }
  }
}

LoopDescriptor* IRContext::GetLoopDescriptor(const Function* f) {
  if (!AreAnalysesValid(kAnalysisLoopAnalysis)) {
    ResetLoopAnalysis();
  }

  std::unordered_map<const Function*, LoopDescriptor>::iterator it =
      loop_descriptors_.find(f);
  if (it == loop_descriptors_.end()) {
    return &loop_descriptors_
                .emplace(std::make_pair(f, LoopDescriptor(this, f)))
                .first->second;
  }

  return &it->second;
}

uint32_t IRContext::FindBuiltinInputVar(uint32_t builtin) {
  for (auto& a : module_->annotations()) {
    if (spv::Op(a.opcode()) != spv::Op::OpDecorate) continue;
    if (spv::Decoration(a.GetSingleWordInOperand(
            kSpvDecorateDecorationInIdx)) != spv::Decoration::BuiltIn)
      continue;
    if (a.GetSingleWordInOperand(kSpvDecorateBuiltinInIdx) != builtin) continue;
    uint32_t target_id = a.GetSingleWordInOperand(kSpvDecorateTargetIdInIdx);
    Instruction* b_var = get_def_use_mgr()->GetDef(target_id);
    if (b_var->opcode() != spv::Op::OpVariable) continue;
    if (spv::StorageClass(b_var->GetSingleWordInOperand(0)) !=
        spv::StorageClass::Input)
      continue;
    return target_id;
  }
  return 0;
}

void IRContext::AddVarToEntryPoints(uint32_t var_id) {
  uint32_t ocnt = 0;
  for (auto& e : module()->entry_points()) {
    bool found = false;
    e.ForEachInOperand([&ocnt, &found, &var_id](const uint32_t* idp) {
      if (ocnt >= kEntryPointInterfaceInIdx) {
        if (*idp == var_id) found = true;
      }
      ++ocnt;
    });
    if (!found) {
      e.AddOperand({SPV_OPERAND_TYPE_ID, {var_id}});
      get_def_use_mgr()->AnalyzeInstDefUse(&e);
    }
  }
}

uint32_t IRContext::GetBuiltinInputVarId(uint32_t builtin) {
  if (!AreAnalysesValid(kAnalysisBuiltinVarId)) ResetBuiltinAnalysis();
  // If cached, return it.
  std::unordered_map<uint32_t, uint32_t>::iterator it =
      builtin_var_id_map_.find(builtin);
  if (it != builtin_var_id_map_.end()) return it->second;
  // Look for one in shader
  uint32_t var_id = FindBuiltinInputVar(builtin);
  if (var_id == 0) {
    // If not found, create it
    // TODO(greg-lunarg): Add support for all builtins
    analysis::TypeManager* type_mgr = get_type_mgr();
    analysis::Type* reg_type;
    switch (spv::BuiltIn(builtin)) {
      case spv::BuiltIn::FragCoord: {
        analysis::Float float_ty(32);
        analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
        analysis::Vector v4float_ty(reg_float_ty, 4);
        reg_type = type_mgr->GetRegisteredType(&v4float_ty);
        break;
      }
      case spv::BuiltIn::VertexIndex:
      case spv::BuiltIn::InstanceIndex:
      case spv::BuiltIn::PrimitiveId:
      case spv::BuiltIn::InvocationId:
      case spv::BuiltIn::SubgroupLocalInvocationId: {
        analysis::Integer uint_ty(32, false);
        reg_type = type_mgr->GetRegisteredType(&uint_ty);
        break;
      }
      case spv::BuiltIn::GlobalInvocationId:
      case spv::BuiltIn::LaunchIdNV: {
        analysis::Integer uint_ty(32, false);
        analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
        analysis::Vector v3uint_ty(reg_uint_ty, 3);
        reg_type = type_mgr->GetRegisteredType(&v3uint_ty);
        break;
      }
      case spv::BuiltIn::TessCoord: {
        analysis::Float float_ty(32);
        analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
        analysis::Vector v3float_ty(reg_float_ty, 3);
        reg_type = type_mgr->GetRegisteredType(&v3float_ty);
        break;
      }
      case spv::BuiltIn::SubgroupLtMask: {
        analysis::Integer uint_ty(32, false);
        analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
        analysis::Vector v4uint_ty(reg_uint_ty, 4);
        reg_type = type_mgr->GetRegisteredType(&v4uint_ty);
        break;
      }
      default: {
        assert(false && "unhandled builtin");
        return 0;
      }
    }
    uint32_t type_id = type_mgr->GetTypeInstruction(reg_type);
    uint32_t varTyPtrId =
        type_mgr->FindPointerToType(type_id, spv::StorageClass::Input);
    // TODO(1841): Handle id overflow.
    var_id = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(
        new Instruction(this, spv::Op::OpVariable, varTyPtrId, var_id,
                        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {uint32_t(spv::StorageClass::Input)}}}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newVarOp);
    module()->AddGlobalValue(std::move(newVarOp));
    get_decoration_mgr()->AddDecorationVal(
        var_id, uint32_t(spv::Decoration::BuiltIn), builtin);
    AddVarToEntryPoints(var_id);
  }
  builtin_var_id_map_[builtin] = var_id;
  return var_id;
}

void IRContext::AddCalls(const Function* func, std::queue<uint32_t>* todo) {
  for (auto bi = func->begin(); bi != func->end(); ++bi)
    for (auto ii = bi->begin(); ii != bi->end(); ++ii)
      if (ii->opcode() == spv::Op::OpFunctionCall)
        todo->push(ii->GetSingleWordInOperand(0));
}

bool IRContext::ProcessEntryPointCallTree(ProcessFunction& pfn) {
  // Collect all of the entry points as the roots.
  std::queue<uint32_t> roots;
  for (auto& e : module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  return ProcessCallTreeFromRoots(pfn, &roots);
}

bool IRContext::ProcessReachableCallTree(ProcessFunction& pfn) {
  std::queue<uint32_t> roots;

  // Add all entry points since they can be reached from outside the module.
  for (auto& e : module()->entry_points())
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));

  // Add all exported functions since they can be reached from outside the
  // module.
  for (auto& a : annotations()) {
    // TODO: Handle group decorations as well.  Currently not generate by any
    // front-end, but could be coming.
    if (a.opcode() == spv::Op::OpDecorate) {
      if (spv::Decoration(a.GetSingleWordOperand(1)) ==
          spv::Decoration::LinkageAttributes) {
        uint32_t lastOperand = a.NumOperands() - 1;
        if (spv::LinkageType(a.GetSingleWordOperand(lastOperand)) ==
            spv::LinkageType::Export) {
          uint32_t id = a.GetSingleWordOperand(0);
          if (GetFunction(id)) {
            roots.push(id);
          }
        }
      }
    }
  }

  return ProcessCallTreeFromRoots(pfn, &roots);
}

bool IRContext::ProcessCallTreeFromRoots(ProcessFunction& pfn,
                                         std::queue<uint32_t>* roots) {
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> done;

  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = GetFunction(fi);
      assert(fn && "Trying to process a function that does not exist.");
      modified = pfn(fn) || modified;
      AddCalls(fn, roots);
    }
  }
  return modified;
}

void IRContext::CollectCallTreeFromRoots(unsigned entryId,
                                         std::unordered_set<uint32_t>* funcs) {
  std::queue<uint32_t> roots;
  roots.push(entryId);
  while (!roots.empty()) {
    const uint32_t fi = roots.front();
    roots.pop();
    funcs->insert(fi);
    Function* fn = GetFunction(fi);
    AddCalls(fn, &roots);
  }
}

void IRContext::EmitErrorMessage(std::string message, Instruction* inst) {
  if (!consumer()) {
    return;
  }

  Instruction* line_inst = inst;
  while (line_inst != nullptr) {  // Stop at the beginning of the basic block.
    if (!line_inst->dbg_line_insts().empty()) {
      line_inst = &line_inst->dbg_line_insts().back();
      if (line_inst->IsNoLine()) {
        line_inst = nullptr;
      }
      break;
    }
    line_inst = line_inst->PreviousNode();
  }

  uint32_t line_number = 0;
  uint32_t col_number = 0;
  std::string source;
  if (line_inst != nullptr) {
    Instruction* file_name =
        get_def_use_mgr()->GetDef(line_inst->GetSingleWordInOperand(0));
    source = file_name->GetInOperand(0).AsString();

    // Get the line number and column number.
    line_number = line_inst->GetSingleWordInOperand(1);
    col_number = line_inst->GetSingleWordInOperand(2);
  }

  message +=
      "\n  " + inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  consumer()(SPV_MSG_ERROR, source.c_str(), {line_number, col_number, 0},
             message.c_str());
}

// Gets the dominator analysis for function |f|.
DominatorAnalysis* IRContext::GetDominatorAnalysis(const Function* f) {
  if (!AreAnalysesValid(kAnalysisDominatorAnalysis)) {
    ResetDominatorAnalysis();
  }

  if (dominator_trees_.find(f) == dominator_trees_.end()) {
    dominator_trees_[f].InitializeTree(*cfg(), f);
  }

  return &dominator_trees_[f];
}

// Gets the postdominator analysis for function |f|.
PostDominatorAnalysis* IRContext::GetPostDominatorAnalysis(const Function* f) {
  if (!AreAnalysesValid(kAnalysisDominatorAnalysis)) {
    ResetDominatorAnalysis();
  }

  if (post_dominator_trees_.find(f) == post_dominator_trees_.end()) {
    post_dominator_trees_[f].InitializeTree(*cfg(), f);
  }

  return &post_dominator_trees_[f];
}

bool IRContext::CheckCFG() {
  std::unordered_map<uint32_t, std::vector<uint32_t>> real_preds;
  if (!AreAnalysesValid(kAnalysisCFG)) {
    return true;
  }

  for (Function& function : *module()) {
    for (const auto& bb : function) {
      bb.ForEachSuccessorLabel([&bb, &real_preds](const uint32_t lab_id) {
        real_preds[lab_id].push_back(bb.id());
      });
    }

    for (auto& bb : function) {
      std::vector<uint32_t> preds = cfg()->preds(bb.id());
      std::vector<uint32_t> real = real_preds[bb.id()];
      std::sort(preds.begin(), preds.end());
      std::sort(real.begin(), real.end());

      bool same = true;
      if (preds.size() != real.size()) {
        same = false;
      }

      for (size_t i = 0; i < real.size() && same; i++) {
        if (preds[i] != real[i]) {
          same = false;
        }
      }

      if (!same) {
        std::cerr << "Predecessors for " << bb.id() << " are different:\n";

        std::cerr << "Real:";
        for (uint32_t i : real) {
          std::cerr << ' ' << i;
        }
        std::cerr << std::endl;

        std::cerr << "Recorded:";
        for (uint32_t i : preds) {
          std::cerr << ' ' << i;
        }
        std::cerr << std::endl;
      }
      if (!same) return false;
    }
  }

  return true;
}

bool IRContext::IsReachable(const opt::BasicBlock& bb) {
  auto enclosing_function = bb.GetParent();
  return GetDominatorAnalysis(enclosing_function)
      ->Dominates(enclosing_function->entry().get(), &bb);
}

spv::ExecutionModel IRContext::GetStage() {
  const auto& entry_points = module()->entry_points();
  if (entry_points.empty()) {
    return spv::ExecutionModel::Max;
  }

  uint32_t stage = entry_points.begin()->GetSingleWordInOperand(
      kEntryPointExecutionModelInIdx);
  auto it = std::find_if(
      entry_points.begin(), entry_points.end(), [stage](const Instruction& x) {
        return x.GetSingleWordInOperand(kEntryPointExecutionModelInIdx) !=
               stage;
      });
  if (it != entry_points.end()) {
    EmitErrorMessage("Mixed stage shader module not supported", &(*it));
  }

  return static_cast<spv::ExecutionModel>(stage);
}

}  // namespace opt
}  // namespace spvtools
