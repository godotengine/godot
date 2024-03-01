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

#include "source/opt/scalar_replacement_pass.h"

#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>

#include "source/extensions.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kDebugValueOperandValueIndex = 5;
constexpr uint32_t kDebugValueOperandExpressionIndex = 6;
constexpr uint32_t kDebugDeclareOperandVariableIndex = 5;
}  // namespace

Pass::Status ScalarReplacementPass::Process() {
  Status status = Status::SuccessWithoutChange;
  for (auto& f : *get_module()) {
    if (f.IsDeclaration()) {
      continue;
    }

    Status functionStatus = ProcessFunction(&f);
    if (functionStatus == Status::Failure)
      return functionStatus;
    else if (functionStatus == Status::SuccessWithChange)
      status = functionStatus;
  }

  return status;
}

Pass::Status ScalarReplacementPass::ProcessFunction(Function* function) {
  std::queue<Instruction*> worklist;
  BasicBlock& entry = *function->begin();
  for (auto iter = entry.begin(); iter != entry.end(); ++iter) {
    // Function storage class OpVariables must appear as the first instructions
    // of the entry block.
    if (iter->opcode() != spv::Op::OpVariable) break;

    Instruction* varInst = &*iter;
    if (CanReplaceVariable(varInst)) {
      worklist.push(varInst);
    }
  }

  Status status = Status::SuccessWithoutChange;
  while (!worklist.empty()) {
    Instruction* varInst = worklist.front();
    worklist.pop();

    Status var_status = ReplaceVariable(varInst, &worklist);
    if (var_status == Status::Failure)
      return var_status;
    else if (var_status == Status::SuccessWithChange)
      status = var_status;
  }

  return status;
}

Pass::Status ScalarReplacementPass::ReplaceVariable(
    Instruction* inst, std::queue<Instruction*>* worklist) {
  std::vector<Instruction*> replacements;
  if (!CreateReplacementVariables(inst, &replacements)) {
    return Status::Failure;
  }

  std::vector<Instruction*> dead;
  bool replaced_all_uses = get_def_use_mgr()->WhileEachUser(
      inst, [this, &replacements, &dead](Instruction* user) {
        if (user->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare) {
          if (ReplaceWholeDebugDeclare(user, replacements)) {
            dead.push_back(user);
            return true;
          }
          return false;
        }
        if (user->GetCommonDebugOpcode() == CommonDebugInfoDebugValue) {
          if (ReplaceWholeDebugValue(user, replacements)) {
            dead.push_back(user);
            return true;
          }
          return false;
        }
        if (!IsAnnotationInst(user->opcode())) {
          switch (user->opcode()) {
            case spv::Op::OpLoad:
              if (ReplaceWholeLoad(user, replacements)) {
                dead.push_back(user);
              } else {
                return false;
              }
              break;
            case spv::Op::OpStore:
              if (ReplaceWholeStore(user, replacements)) {
                dead.push_back(user);
              } else {
                return false;
              }
              break;
            case spv::Op::OpAccessChain:
            case spv::Op::OpInBoundsAccessChain:
              if (ReplaceAccessChain(user, replacements))
                dead.push_back(user);
              else
                return false;
              break;
            case spv::Op::OpName:
            case spv::Op::OpMemberName:
              break;
            default:
              assert(false && "Unexpected opcode");
              break;
          }
        }
        return true;
      });

  if (replaced_all_uses) {
    dead.push_back(inst);
  } else {
    return Status::Failure;
  }

  // If there are no dead instructions to clean up, return with no changes.
  if (dead.empty()) return Status::SuccessWithoutChange;

  // Clean up some dead code.
  while (!dead.empty()) {
    Instruction* toKill = dead.back();
    dead.pop_back();
    context()->KillInst(toKill);
  }

  // Attempt to further scalarize.
  for (auto var : replacements) {
    if (var->opcode() == spv::Op::OpVariable) {
      if (get_def_use_mgr()->NumUsers(var) == 0) {
        context()->KillInst(var);
      } else if (CanReplaceVariable(var)) {
        worklist->push(var);
      }
    }
  }

  return Status::SuccessWithChange;
}

bool ScalarReplacementPass::ReplaceWholeDebugDeclare(
    Instruction* dbg_decl, const std::vector<Instruction*>& replacements) {
  // Insert Deref operation to the front of the operation list of |dbg_decl|.
  Instruction* dbg_expr = context()->get_def_use_mgr()->GetDef(
      dbg_decl->GetSingleWordOperand(kDebugValueOperandExpressionIndex));
  auto* deref_expr =
      context()->get_debug_info_mgr()->DerefDebugExpression(dbg_expr);

  // Add DebugValue instruction with Indexes operand and Deref operation.
  int32_t idx = 0;
  for (const auto* var : replacements) {
    Instruction* insert_before = var->NextNode();
    while (insert_before->opcode() == spv::Op::OpVariable)
      insert_before = insert_before->NextNode();
    assert(insert_before != nullptr && "unexpected end of list");
    Instruction* added_dbg_value =
        context()->get_debug_info_mgr()->AddDebugValueForDecl(
            dbg_decl, /*value_id=*/var->result_id(),
            /*insert_before=*/insert_before, /*scope_and_line=*/dbg_decl);

    if (added_dbg_value == nullptr) return false;
    added_dbg_value->AddOperand(
        {SPV_OPERAND_TYPE_ID,
         {context()->get_constant_mgr()->GetSIntConstId(idx)}});
    added_dbg_value->SetOperand(kDebugValueOperandExpressionIndex,
                                {deref_expr->result_id()});
    if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse)) {
      context()->get_def_use_mgr()->AnalyzeInstUse(added_dbg_value);
    }
    ++idx;
  }
  return true;
}

bool ScalarReplacementPass::ReplaceWholeDebugValue(
    Instruction* dbg_value, const std::vector<Instruction*>& replacements) {
  int32_t idx = 0;
  BasicBlock* block = context()->get_instr_block(dbg_value);
  for (auto var : replacements) {
    // Clone the DebugValue.
    std::unique_ptr<Instruction> new_dbg_value(dbg_value->Clone(context()));
    uint32_t new_id = TakeNextId();
    if (new_id == 0) return false;
    new_dbg_value->SetResultId(new_id);
    // Update 'Value' operand to the |replacements|.
    new_dbg_value->SetOperand(kDebugValueOperandValueIndex, {var->result_id()});
    // Append 'Indexes' operand.
    new_dbg_value->AddOperand(
        {SPV_OPERAND_TYPE_ID,
         {context()->get_constant_mgr()->GetSIntConstId(idx)}});
    // Insert the new DebugValue to the basic block.
    auto* added_instr = dbg_value->InsertBefore(std::move(new_dbg_value));
    get_def_use_mgr()->AnalyzeInstDefUse(added_instr);
    context()->set_instr_block(added_instr, block);
    ++idx;
  }
  return true;
}

bool ScalarReplacementPass::ReplaceWholeLoad(
    Instruction* load, const std::vector<Instruction*>& replacements) {
  // Replaces the load of the entire composite with a load from each replacement
  // variable followed by a composite construction.
  BasicBlock* block = context()->get_instr_block(load);
  std::vector<Instruction*> loads;
  loads.reserve(replacements.size());
  BasicBlock::iterator where(load);
  for (auto var : replacements) {
    // Create a load of each replacement variable.
    if (var->opcode() != spv::Op::OpVariable) {
      loads.push_back(var);
      continue;
    }

    Instruction* type = GetStorageType(var);
    uint32_t loadId = TakeNextId();
    if (loadId == 0) {
      return false;
    }
    std::unique_ptr<Instruction> newLoad(
        new Instruction(context(), spv::Op::OpLoad, type->result_id(), loadId,
                        std::initializer_list<Operand>{
                            {SPV_OPERAND_TYPE_ID, {var->result_id()}}}));
    // Copy memory access attributes which start at index 1. Index 0 is the
    // pointer to load.
    for (uint32_t i = 1; i < load->NumInOperands(); ++i) {
      Operand copy(load->GetInOperand(i));
      newLoad->AddOperand(std::move(copy));
    }
    where = where.InsertBefore(std::move(newLoad));
    get_def_use_mgr()->AnalyzeInstDefUse(&*where);
    context()->set_instr_block(&*where, block);
    where->UpdateDebugInfoFrom(load);
    loads.push_back(&*where);
  }

  // Construct a new composite.
  uint32_t compositeId = TakeNextId();
  if (compositeId == 0) {
    return false;
  }
  where = load;
  std::unique_ptr<Instruction> compositeConstruct(
      new Instruction(context(), spv::Op::OpCompositeConstruct, load->type_id(),
                      compositeId, {}));
  for (auto l : loads) {
    Operand op(SPV_OPERAND_TYPE_ID,
               std::initializer_list<uint32_t>{l->result_id()});
    compositeConstruct->AddOperand(std::move(op));
  }
  where = where.InsertBefore(std::move(compositeConstruct));
  get_def_use_mgr()->AnalyzeInstDefUse(&*where);
  where->UpdateDebugInfoFrom(load);
  context()->set_instr_block(&*where, block);
  context()->ReplaceAllUsesWith(load->result_id(), compositeId);
  return true;
}

bool ScalarReplacementPass::ReplaceWholeStore(
    Instruction* store, const std::vector<Instruction*>& replacements) {
  // Replaces a store to the whole composite with a series of extract and stores
  // to each element.
  uint32_t storeInput = store->GetSingleWordInOperand(1u);
  BasicBlock* block = context()->get_instr_block(store);
  BasicBlock::iterator where(store);
  uint32_t elementIndex = 0;
  for (auto var : replacements) {
    // Create the extract.
    if (var->opcode() != spv::Op::OpVariable) {
      elementIndex++;
      continue;
    }

    Instruction* type = GetStorageType(var);
    uint32_t extractId = TakeNextId();
    if (extractId == 0) {
      return false;
    }
    std::unique_ptr<Instruction> extract(new Instruction(
        context(), spv::Op::OpCompositeExtract, type->result_id(), extractId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_ID, {storeInput}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {elementIndex++}}}));
    auto iter = where.InsertBefore(std::move(extract));
    iter->UpdateDebugInfoFrom(store);
    get_def_use_mgr()->AnalyzeInstDefUse(&*iter);
    context()->set_instr_block(&*iter, block);

    // Create the store.
    std::unique_ptr<Instruction> newStore(
        new Instruction(context(), spv::Op::OpStore, 0, 0,
                        std::initializer_list<Operand>{
                            {SPV_OPERAND_TYPE_ID, {var->result_id()}},
                            {SPV_OPERAND_TYPE_ID, {extractId}}}));
    // Copy memory access attributes which start at index 2. Index 0 is the
    // pointer and index 1 is the data.
    for (uint32_t i = 2; i < store->NumInOperands(); ++i) {
      Operand copy(store->GetInOperand(i));
      newStore->AddOperand(std::move(copy));
    }
    iter = where.InsertBefore(std::move(newStore));
    iter->UpdateDebugInfoFrom(store);
    get_def_use_mgr()->AnalyzeInstDefUse(&*iter);
    context()->set_instr_block(&*iter, block);
  }
  return true;
}

bool ScalarReplacementPass::ReplaceAccessChain(
    Instruction* chain, const std::vector<Instruction*>& replacements) {
  // Replaces the access chain with either another access chain (with one fewer
  // indexes) or a direct use of the replacement variable.
  uint32_t indexId = chain->GetSingleWordInOperand(1u);
  const Instruction* index = get_def_use_mgr()->GetDef(indexId);
  int64_t indexValue = context()
                           ->get_constant_mgr()
                           ->GetConstantFromInst(index)
                           ->GetSignExtendedValue();
  if (indexValue < 0 ||
      indexValue >= static_cast<int64_t>(replacements.size())) {
    // Out of bounds access, this is illegal IR.  Notice that OpAccessChain
    // indexing is 0-based, so we should also reject index == size-of-array.
    return false;
  } else {
    const Instruction* var = replacements[static_cast<size_t>(indexValue)];
    if (chain->NumInOperands() > 2) {
      // Replace input access chain with another access chain.
      BasicBlock::iterator chainIter(chain);
      uint32_t replacementId = TakeNextId();
      if (replacementId == 0) {
        return false;
      }
      std::unique_ptr<Instruction> replacementChain(new Instruction(
          context(), chain->opcode(), chain->type_id(), replacementId,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {var->result_id()}}}));
      // Add the remaining indexes.
      for (uint32_t i = 2; i < chain->NumInOperands(); ++i) {
        Operand copy(chain->GetInOperand(i));
        replacementChain->AddOperand(std::move(copy));
      }
      replacementChain->UpdateDebugInfoFrom(chain);
      auto iter = chainIter.InsertBefore(std::move(replacementChain));
      get_def_use_mgr()->AnalyzeInstDefUse(&*iter);
      context()->set_instr_block(&*iter, context()->get_instr_block(chain));
      context()->ReplaceAllUsesWith(chain->result_id(), replacementId);
    } else {
      // Replace with a use of the variable.
      context()->ReplaceAllUsesWith(chain->result_id(), var->result_id());
    }
  }

  return true;
}

bool ScalarReplacementPass::CreateReplacementVariables(
    Instruction* inst, std::vector<Instruction*>* replacements) {
  Instruction* type = GetStorageType(inst);

  std::unique_ptr<std::unordered_set<int64_t>> components_used =
      GetUsedComponents(inst);

  uint32_t elem = 0;
  switch (type->opcode()) {
    case spv::Op::OpTypeStruct:
      type->ForEachInOperand(
          [this, inst, &elem, replacements, &components_used](uint32_t* id) {
            if (!components_used || components_used->count(elem)) {
              CreateVariable(*id, inst, elem, replacements);
            } else {
              replacements->push_back(GetUndef(*id));
            }
            elem++;
          });
      break;
    case spv::Op::OpTypeArray:
      for (uint32_t i = 0; i != GetArrayLength(type); ++i) {
        if (!components_used || components_used->count(i)) {
          CreateVariable(type->GetSingleWordInOperand(0u), inst, i,
                         replacements);
        } else {
          uint32_t element_type_id = type->GetSingleWordInOperand(0);
          replacements->push_back(GetUndef(element_type_id));
        }
      }
      break;

    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector:
      for (uint32_t i = 0; i != GetNumElements(type); ++i) {
        CreateVariable(type->GetSingleWordInOperand(0u), inst, i, replacements);
      }
      break;

    default:
      assert(false && "Unexpected type.");
      break;
  }

  TransferAnnotations(inst, replacements);
  return std::find(replacements->begin(), replacements->end(), nullptr) ==
         replacements->end();
}

Instruction* ScalarReplacementPass::GetUndef(uint32_t type_id) {
  return get_def_use_mgr()->GetDef(Type2Undef(type_id));
}

void ScalarReplacementPass::TransferAnnotations(
    const Instruction* source, std::vector<Instruction*>* replacements) {
  // Only transfer invariant and restrict decorations on the variable. There are
  // no type or member decorations that are necessary to transfer.
  for (auto inst :
       get_decoration_mgr()->GetDecorationsFor(source->result_id(), false)) {
    assert(inst->opcode() == spv::Op::OpDecorate);
    auto decoration = spv::Decoration(inst->GetSingleWordInOperand(1u));
    if (decoration == spv::Decoration::Invariant ||
        decoration == spv::Decoration::Restrict) {
      for (auto var : *replacements) {
        if (var == nullptr) {
          continue;
        }

        std::unique_ptr<Instruction> annotation(new Instruction(
            context(), spv::Op::OpDecorate, 0, 0,
            std::initializer_list<Operand>{
                {SPV_OPERAND_TYPE_ID, {var->result_id()}},
                {SPV_OPERAND_TYPE_DECORATION, {uint32_t(decoration)}}}));
        for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
          Operand copy(inst->GetInOperand(i));
          annotation->AddOperand(std::move(copy));
        }
        context()->AddAnnotationInst(std::move(annotation));
        get_def_use_mgr()->AnalyzeInstUse(&*--context()->annotation_end());
      }
    }
  }
}

void ScalarReplacementPass::CreateVariable(
    uint32_t type_id, Instruction* var_inst, uint32_t index,
    std::vector<Instruction*>* replacements) {
  uint32_t ptr_id = GetOrCreatePointerType(type_id);
  uint32_t id = TakeNextId();

  if (id == 0) {
    replacements->push_back(nullptr);
  }

  std::unique_ptr<Instruction> variable(
      new Instruction(context(), spv::Op::OpVariable, ptr_id, id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_STORAGE_CLASS,
                           {uint32_t(spv::StorageClass::Function)}}}));

  BasicBlock* block = context()->get_instr_block(var_inst);
  block->begin().InsertBefore(std::move(variable));
  Instruction* inst = &*block->begin();

  // If varInst was initialized, make sure to initialize its replacement.
  GetOrCreateInitialValue(var_inst, index, inst);
  get_def_use_mgr()->AnalyzeInstDefUse(inst);
  context()->set_instr_block(inst, block);

  CopyDecorationsToVariable(var_inst, inst, index);
  inst->UpdateDebugInfoFrom(var_inst);

  replacements->push_back(inst);
}

uint32_t ScalarReplacementPass::GetOrCreatePointerType(uint32_t id) {
  auto iter = pointee_to_pointer_.find(id);
  if (iter != pointee_to_pointer_.end()) return iter->second;

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  uint32_t ptr_type_id =
      type_mgr->FindPointerToType(id, spv::StorageClass::Function);
  pointee_to_pointer_[id] = ptr_type_id;
  return ptr_type_id;
}

void ScalarReplacementPass::GetOrCreateInitialValue(Instruction* source,
                                                    uint32_t index,
                                                    Instruction* newVar) {
  assert(source->opcode() == spv::Op::OpVariable);
  if (source->NumInOperands() < 2) return;

  uint32_t initId = source->GetSingleWordInOperand(1u);
  uint32_t storageId = GetStorageType(newVar)->result_id();
  Instruction* init = get_def_use_mgr()->GetDef(initId);
  uint32_t newInitId = 0;
  // TODO(dnovillo): Refactor this with constant propagation.
  if (init->opcode() == spv::Op::OpConstantNull) {
    // Initialize to appropriate NULL.
    auto iter = type_to_null_.find(storageId);
    if (iter == type_to_null_.end()) {
      newInitId = TakeNextId();
      type_to_null_[storageId] = newInitId;
      context()->AddGlobalValue(
          MakeUnique<Instruction>(context(), spv::Op::OpConstantNull, storageId,
                                  newInitId, std::initializer_list<Operand>{}));
      Instruction* newNull = &*--context()->types_values_end();
      get_def_use_mgr()->AnalyzeInstDefUse(newNull);
    } else {
      newInitId = iter->second;
    }
  } else if (IsSpecConstantInst(init->opcode())) {
    // Create a new constant extract.
    newInitId = TakeNextId();
    context()->AddGlobalValue(MakeUnique<Instruction>(
        context(), spv::Op::OpSpecConstantOp, storageId, newInitId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER,
             {uint32_t(spv::Op::OpCompositeExtract)}},
            {SPV_OPERAND_TYPE_ID, {init->result_id()}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}}}));
    Instruction* newSpecConst = &*--context()->types_values_end();
    get_def_use_mgr()->AnalyzeInstDefUse(newSpecConst);
  } else if (init->opcode() == spv::Op::OpConstantComposite) {
    // Get the appropriate index constant.
    newInitId = init->GetSingleWordInOperand(index);
    Instruction* element = get_def_use_mgr()->GetDef(newInitId);
    if (element->opcode() == spv::Op::OpUndef) {
      // Undef is not a valid initializer for a variable.
      newInitId = 0;
    }
  } else {
    assert(false);
  }

  if (newInitId != 0) {
    newVar->AddOperand({SPV_OPERAND_TYPE_ID, {newInitId}});
  }
}

uint64_t ScalarReplacementPass::GetArrayLength(
    const Instruction* arrayType) const {
  assert(arrayType->opcode() == spv::Op::OpTypeArray);
  const Instruction* length =
      get_def_use_mgr()->GetDef(arrayType->GetSingleWordInOperand(1u));
  return context()
      ->get_constant_mgr()
      ->GetConstantFromInst(length)
      ->GetZeroExtendedValue();
}

uint64_t ScalarReplacementPass::GetNumElements(const Instruction* type) const {
  assert(type->opcode() == spv::Op::OpTypeVector ||
         type->opcode() == spv::Op::OpTypeMatrix);
  const Operand& op = type->GetInOperand(1u);
  assert(op.words.size() <= 2);
  uint64_t len = 0;
  for (size_t i = 0; i != op.words.size(); ++i) {
    len |= (static_cast<uint64_t>(op.words[i]) << (32ull * i));
  }
  return len;
}

bool ScalarReplacementPass::IsSpecConstant(uint32_t id) const {
  const Instruction* inst = get_def_use_mgr()->GetDef(id);
  assert(inst);
  return spvOpcodeIsSpecConstant(inst->opcode());
}

Instruction* ScalarReplacementPass::GetStorageType(
    const Instruction* inst) const {
  assert(inst->opcode() == spv::Op::OpVariable);

  uint32_t ptrTypeId = inst->type_id();
  uint32_t typeId =
      get_def_use_mgr()->GetDef(ptrTypeId)->GetSingleWordInOperand(1u);
  return get_def_use_mgr()->GetDef(typeId);
}

bool ScalarReplacementPass::CanReplaceVariable(
    const Instruction* varInst) const {
  assert(varInst->opcode() == spv::Op::OpVariable);

  // Can only replace function scope variables.
  if (spv::StorageClass(varInst->GetSingleWordInOperand(0u)) !=
      spv::StorageClass::Function) {
    return false;
  }

  if (!CheckTypeAnnotations(get_def_use_mgr()->GetDef(varInst->type_id()))) {
    return false;
  }

  const Instruction* typeInst = GetStorageType(varInst);
  if (!CheckType(typeInst)) {
    return false;
  }

  if (!CheckAnnotations(varInst)) {
    return false;
  }

  if (!CheckUses(varInst)) {
    return false;
  }

  return true;
}

bool ScalarReplacementPass::CheckType(const Instruction* typeInst) const {
  if (!CheckTypeAnnotations(typeInst)) {
    return false;
  }

  switch (typeInst->opcode()) {
    case spv::Op::OpTypeStruct:
      // Don't bother with empty structs or very large structs.
      if (typeInst->NumInOperands() == 0 ||
          IsLargerThanSizeLimit(typeInst->NumInOperands())) {
        return false;
      }
      return true;
    case spv::Op::OpTypeArray:
      if (IsSpecConstant(typeInst->GetSingleWordInOperand(1u))) {
        return false;
      }
      if (IsLargerThanSizeLimit(GetArrayLength(typeInst))) {
        return false;
      }
      return true;
      // TODO(alanbaker): Develop some heuristics for when this should be
      // re-enabled.
      //// Specifically including matrix and vector in an attempt to reduce the
      //// number of vector registers required.
      // case spv::Op::OpTypeMatrix:
      // case spv::Op::OpTypeVector:
      //  if (IsLargerThanSizeLimit(GetNumElements(typeInst))) return false;
      //  return true;

    case spv::Op::OpTypeRuntimeArray:
    default:
      return false;
  }
}

bool ScalarReplacementPass::CheckTypeAnnotations(
    const Instruction* typeInst) const {
  for (auto inst :
       get_decoration_mgr()->GetDecorationsFor(typeInst->result_id(), false)) {
    uint32_t decoration;
    if (inst->opcode() == spv::Op::OpDecorate) {
      decoration = inst->GetSingleWordInOperand(1u);
    } else {
      assert(inst->opcode() == spv::Op::OpMemberDecorate);
      decoration = inst->GetSingleWordInOperand(2u);
    }

    switch (spv::Decoration(decoration)) {
      case spv::Decoration::RowMajor:
      case spv::Decoration::ColMajor:
      case spv::Decoration::ArrayStride:
      case spv::Decoration::MatrixStride:
      case spv::Decoration::CPacked:
      case spv::Decoration::Invariant:
      case spv::Decoration::Restrict:
      case spv::Decoration::Offset:
      case spv::Decoration::Alignment:
      case spv::Decoration::AlignmentId:
      case spv::Decoration::MaxByteOffset:
      case spv::Decoration::RelaxedPrecision:
      case spv::Decoration::AliasedPointer:
      case spv::Decoration::RestrictPointer:
        break;
      default:
        return false;
    }
  }

  return true;
}

bool ScalarReplacementPass::CheckAnnotations(const Instruction* varInst) const {
  for (auto inst :
       get_decoration_mgr()->GetDecorationsFor(varInst->result_id(), false)) {
    assert(inst->opcode() == spv::Op::OpDecorate);
    auto decoration = spv::Decoration(inst->GetSingleWordInOperand(1u));
    switch (decoration) {
      case spv::Decoration::Invariant:
      case spv::Decoration::Restrict:
      case spv::Decoration::Alignment:
      case spv::Decoration::AlignmentId:
      case spv::Decoration::MaxByteOffset:
      case spv::Decoration::AliasedPointer:
      case spv::Decoration::RestrictPointer:
        break;
      default:
        return false;
    }
  }

  return true;
}

bool ScalarReplacementPass::CheckUses(const Instruction* inst) const {
  VariableStats stats = {0, 0};
  bool ok = CheckUses(inst, &stats);

  // TODO(alanbaker/greg-lunarg): Add some meaningful heuristics about when
  // SRoA is costly, such as when the structure has many (unaccessed?)
  // members.

  return ok;
}

bool ScalarReplacementPass::CheckUses(const Instruction* inst,
                                      VariableStats* stats) const {
  uint64_t max_legal_index = GetMaxLegalIndex(inst);

  bool ok = true;
  get_def_use_mgr()->ForEachUse(inst, [this, max_legal_index, stats, &ok](
                                          const Instruction* user,
                                          uint32_t index) {
    if (user->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare ||
        user->GetCommonDebugOpcode() == CommonDebugInfoDebugValue) {
      // TODO: include num_partial_accesses if it uses Fragment operation or
      // DebugValue has Indexes operand.
      stats->num_full_accesses++;
      return;
    }

    // Annotations are check as a group separately.
    if (!IsAnnotationInst(user->opcode())) {
      switch (user->opcode()) {
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
          if (index == 2u && user->NumInOperands() > 1) {
            uint32_t id = user->GetSingleWordInOperand(1u);
            const Instruction* opInst = get_def_use_mgr()->GetDef(id);
            const auto* constant =
                context()->get_constant_mgr()->GetConstantFromInst(opInst);
            if (!constant) {
              ok = false;
            } else if (constant->GetZeroExtendedValue() >= max_legal_index) {
              ok = false;
            } else {
              if (!CheckUsesRelaxed(user)) ok = false;
            }
            stats->num_partial_accesses++;
          } else {
            ok = false;
          }
          break;
        case spv::Op::OpLoad:
          if (!CheckLoad(user, index)) ok = false;
          stats->num_full_accesses++;
          break;
        case spv::Op::OpStore:
          if (!CheckStore(user, index)) ok = false;
          stats->num_full_accesses++;
          break;
        case spv::Op::OpName:
        case spv::Op::OpMemberName:
          break;
        default:
          ok = false;
          break;
      }
    }
  });

  return ok;
}

bool ScalarReplacementPass::CheckUsesRelaxed(const Instruction* inst) const {
  bool ok = true;
  get_def_use_mgr()->ForEachUse(
      inst, [this, &ok](const Instruction* user, uint32_t index) {
        switch (user->opcode()) {
          case spv::Op::OpAccessChain:
          case spv::Op::OpInBoundsAccessChain:
            if (index != 2u) {
              ok = false;
            } else {
              if (!CheckUsesRelaxed(user)) ok = false;
            }
            break;
          case spv::Op::OpLoad:
            if (!CheckLoad(user, index)) ok = false;
            break;
          case spv::Op::OpStore:
            if (!CheckStore(user, index)) ok = false;
            break;
          case spv::Op::OpImageTexelPointer:
            if (!CheckImageTexelPointer(index)) ok = false;
            break;
          case spv::Op::OpExtInst:
            if (user->GetCommonDebugOpcode() != CommonDebugInfoDebugDeclare ||
                !CheckDebugDeclare(index))
              ok = false;
            break;
          default:
            ok = false;
            break;
        }
      });

  return ok;
}

bool ScalarReplacementPass::CheckImageTexelPointer(uint32_t index) const {
  return index == 2u;
}

bool ScalarReplacementPass::CheckLoad(const Instruction* inst,
                                      uint32_t index) const {
  if (index != 2u) return false;
  if (inst->NumInOperands() >= 2 &&
      inst->GetSingleWordInOperand(1u) &
          uint32_t(spv::MemoryAccessMask::Volatile))
    return false;
  return true;
}

bool ScalarReplacementPass::CheckStore(const Instruction* inst,
                                       uint32_t index) const {
  if (index != 0u) return false;
  if (inst->NumInOperands() >= 3 &&
      inst->GetSingleWordInOperand(2u) &
          uint32_t(spv::MemoryAccessMask::Volatile))
    return false;
  return true;
}

bool ScalarReplacementPass::CheckDebugDeclare(uint32_t index) const {
  if (index != kDebugDeclareOperandVariableIndex) return false;
  return true;
}

bool ScalarReplacementPass::IsLargerThanSizeLimit(uint64_t length) const {
  if (max_num_elements_ == 0) {
    return false;
  }
  return length > max_num_elements_;
}

std::unique_ptr<std::unordered_set<int64_t>>
ScalarReplacementPass::GetUsedComponents(Instruction* inst) {
  std::unique_ptr<std::unordered_set<int64_t>> result(
      new std::unordered_set<int64_t>());

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  def_use_mgr->WhileEachUser(inst, [&result, def_use_mgr,
                                    this](Instruction* use) {
    switch (use->opcode()) {
      case spv::Op::OpLoad: {
        // Look for extract from the load.
        std::vector<uint32_t> t;
        if (def_use_mgr->WhileEachUser(use, [&t](Instruction* use2) {
              if (use2->opcode() != spv::Op::OpCompositeExtract ||
                  use2->NumInOperands() <= 1) {
                return false;
              }
              t.push_back(use2->GetSingleWordInOperand(1));
              return true;
            })) {
          result->insert(t.begin(), t.end());
          return true;
        } else {
          result.reset(nullptr);
          return false;
        }
      }
      case spv::Op::OpName:
      case spv::Op::OpMemberName:
      case spv::Op::OpStore:
        // No components are used.
        return true;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain: {
        // Add the first index it if is a constant.
        // TODO: Could be improved by checking if the address is used in a load.
        analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
        uint32_t index_id = use->GetSingleWordInOperand(1);
        const analysis::Constant* index_const =
            const_mgr->FindDeclaredConstant(index_id);
        if (index_const) {
          result->insert(index_const->GetSignExtendedValue());
          return true;
        } else {
          // Could be any element.  Assuming all are used.
          result.reset(nullptr);
          return false;
        }
      }
      default:
        // We do not know what is happening.  Have to assume the worst.
        result.reset(nullptr);
        return false;
    }
  });

  return result;
}

uint64_t ScalarReplacementPass::GetMaxLegalIndex(
    const Instruction* var_inst) const {
  assert(var_inst->opcode() == spv::Op::OpVariable &&
         "|var_inst| must be a variable instruction.");
  Instruction* type = GetStorageType(var_inst);
  switch (type->opcode()) {
    case spv::Op::OpTypeStruct:
      return type->NumInOperands();
    case spv::Op::OpTypeArray:
      return GetArrayLength(type);
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector:
      return GetNumElements(type);
    default:
      return 0;
  }
  return 0;
}

void ScalarReplacementPass::CopyDecorationsToVariable(Instruction* from,
                                                      Instruction* to,
                                                      uint32_t member_index) {
  CopyPointerDecorationsToVariable(from, to);
  CopyNecessaryMemberDecorationsToVariable(from, to, member_index);
}

void ScalarReplacementPass::CopyPointerDecorationsToVariable(Instruction* from,
                                                             Instruction* to) {
  // The RestrictPointer and AliasedPointer decorations are copied to all
  // members even if the new variable does not contain a pointer. It does
  // not hurt to do so.
  for (auto dec_inst :
       get_decoration_mgr()->GetDecorationsFor(from->result_id(), false)) {
    uint32_t decoration;
    decoration = dec_inst->GetSingleWordInOperand(1u);
    switch (spv::Decoration(decoration)) {
      case spv::Decoration::AliasedPointer:
      case spv::Decoration::RestrictPointer: {
        std::unique_ptr<Instruction> new_dec_inst(dec_inst->Clone(context()));
        new_dec_inst->SetInOperand(0, {to->result_id()});
        context()->AddAnnotationInst(std::move(new_dec_inst));
      } break;
      default:
        break;
    }
  }
}

void ScalarReplacementPass::CopyNecessaryMemberDecorationsToVariable(
    Instruction* from, Instruction* to, uint32_t member_index) {
  Instruction* type_inst = GetStorageType(from);
  for (auto dec_inst :
       get_decoration_mgr()->GetDecorationsFor(type_inst->result_id(), false)) {
    uint32_t decoration;
    if (dec_inst->opcode() == spv::Op::OpMemberDecorate) {
      if (dec_inst->GetSingleWordInOperand(1) != member_index) {
        continue;
      }

      decoration = dec_inst->GetSingleWordInOperand(2u);
      switch (spv::Decoration(decoration)) {
        case spv::Decoration::ArrayStride:
        case spv::Decoration::Alignment:
        case spv::Decoration::AlignmentId:
        case spv::Decoration::MaxByteOffset:
        case spv::Decoration::MaxByteOffsetId:
        case spv::Decoration::RelaxedPrecision: {
          std::unique_ptr<Instruction> new_dec_inst(
              new Instruction(context(), spv::Op::OpDecorate, 0, 0, {}));
          new_dec_inst->AddOperand(
              Operand(SPV_OPERAND_TYPE_ID, {to->result_id()}));
          for (uint32_t i = 2; i < dec_inst->NumInOperandWords(); ++i) {
            new_dec_inst->AddOperand(Operand(dec_inst->GetInOperand(i)));
          }
          context()->AddAnnotationInst(std::move(new_dec_inst));
        } break;
        default:
          break;
      }
    }
  }
}

}  // namespace opt
}  // namespace spvtools
