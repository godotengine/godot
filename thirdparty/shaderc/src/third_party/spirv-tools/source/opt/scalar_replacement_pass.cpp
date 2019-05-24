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

#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status ScalarReplacementPass::Process() {
  Status status = Status::SuccessWithoutChange;
  for (auto& f : *get_module()) {
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
    if (iter->opcode() != SpvOpVariable) break;

    Instruction* varInst = &*iter;
    if (CanReplaceVariable(varInst)) {
      worklist.push(varInst);
    }
  }

  Status status = Status::SuccessWithoutChange;
  while (!worklist.empty()) {
    Instruction* varInst = worklist.front();
    worklist.pop();

    if (!ReplaceVariable(varInst, &worklist))
      return Status::Failure;
    else
      status = Status::SuccessWithChange;
  }

  return status;
}

bool ScalarReplacementPass::ReplaceVariable(
    Instruction* inst, std::queue<Instruction*>* worklist) {
  std::vector<Instruction*> replacements;
  CreateReplacementVariables(inst, &replacements);

  std::vector<Instruction*> dead;
  dead.push_back(inst);
  if (!get_def_use_mgr()->WhileEachUser(
          inst, [this, &replacements, &dead](Instruction* user) {
            if (!IsAnnotationInst(user->opcode())) {
              switch (user->opcode()) {
                case SpvOpLoad:
                  ReplaceWholeLoad(user, replacements);
                  dead.push_back(user);
                  break;
                case SpvOpStore:
                  ReplaceWholeStore(user, replacements);
                  dead.push_back(user);
                  break;
                case SpvOpAccessChain:
                case SpvOpInBoundsAccessChain:
                  if (!ReplaceAccessChain(user, replacements)) return false;
                  dead.push_back(user);
                  break;
                case SpvOpName:
                case SpvOpMemberName:
                  break;
                default:
                  assert(false && "Unexpected opcode");
                  break;
              }
            }
            return true;
          }))
    return false;

  // Clean up some dead code.
  while (!dead.empty()) {
    Instruction* toKill = dead.back();
    dead.pop_back();
    context()->KillInst(toKill);
  }

  // Attempt to further scalarize.
  for (auto var : replacements) {
    if (var->opcode() == SpvOpVariable) {
      if (get_def_use_mgr()->NumUsers(var) == 0) {
        context()->KillInst(var);
      } else if (CanReplaceVariable(var)) {
        worklist->push(var);
      }
    }
  }

  return true;
}

void ScalarReplacementPass::ReplaceWholeLoad(
    Instruction* load, const std::vector<Instruction*>& replacements) {
  // Replaces the load of the entire composite with a load from each replacement
  // variable followed by a composite construction.
  BasicBlock* block = context()->get_instr_block(load);
  std::vector<Instruction*> loads;
  loads.reserve(replacements.size());
  BasicBlock::iterator where(load);
  for (auto var : replacements) {
    // Create a load of each replacement variable.
    if (var->opcode() != SpvOpVariable) {
      loads.push_back(var);
      continue;
    }

    Instruction* type = GetStorageType(var);
    uint32_t loadId = TakeNextId();
    std::unique_ptr<Instruction> newLoad(
        new Instruction(context(), SpvOpLoad, type->result_id(), loadId,
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
    loads.push_back(&*where);
  }

  // Construct a new composite.
  uint32_t compositeId = TakeNextId();
  where = load;
  std::unique_ptr<Instruction> compositeConstruct(new Instruction(
      context(), SpvOpCompositeConstruct, load->type_id(), compositeId, {}));
  for (auto l : loads) {
    Operand op(SPV_OPERAND_TYPE_ID,
               std::initializer_list<uint32_t>{l->result_id()});
    compositeConstruct->AddOperand(std::move(op));
  }
  where = where.InsertBefore(std::move(compositeConstruct));
  get_def_use_mgr()->AnalyzeInstDefUse(&*where);
  context()->set_instr_block(&*where, block);
  context()->ReplaceAllUsesWith(load->result_id(), compositeId);
}

void ScalarReplacementPass::ReplaceWholeStore(
    Instruction* store, const std::vector<Instruction*>& replacements) {
  // Replaces a store to the whole composite with a series of extract and stores
  // to each element.
  uint32_t storeInput = store->GetSingleWordInOperand(1u);
  BasicBlock* block = context()->get_instr_block(store);
  BasicBlock::iterator where(store);
  uint32_t elementIndex = 0;
  for (auto var : replacements) {
    // Create the extract.
    if (var->opcode() != SpvOpVariable) {
      elementIndex++;
      continue;
    }

    Instruction* type = GetStorageType(var);
    uint32_t extractId = TakeNextId();
    std::unique_ptr<Instruction> extract(new Instruction(
        context(), SpvOpCompositeExtract, type->result_id(), extractId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_ID, {storeInput}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {elementIndex++}}}));
    auto iter = where.InsertBefore(std::move(extract));
    get_def_use_mgr()->AnalyzeInstDefUse(&*iter);
    context()->set_instr_block(&*iter, block);

    // Create the store.
    std::unique_ptr<Instruction> newStore(
        new Instruction(context(), SpvOpStore, 0, 0,
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
    get_def_use_mgr()->AnalyzeInstDefUse(&*iter);
    context()->set_instr_block(&*iter, block);
  }
}

bool ScalarReplacementPass::ReplaceAccessChain(
    Instruction* chain, const std::vector<Instruction*>& replacements) {
  // Replaces the access chain with either another access chain (with one fewer
  // indexes) or a direct use of the replacement variable.
  uint32_t indexId = chain->GetSingleWordInOperand(1u);
  const Instruction* index = get_def_use_mgr()->GetDef(indexId);
  size_t indexValue = GetConstantInteger(index);
  if (indexValue > replacements.size()) {
    // Out of bounds access, this is illegal IR.
    return false;
  } else {
    const Instruction* var = replacements[indexValue];
    if (chain->NumInOperands() > 2) {
      // Replace input access chain with another access chain.
      BasicBlock::iterator chainIter(chain);
      uint32_t replacementId = TakeNextId();
      std::unique_ptr<Instruction> replacementChain(new Instruction(
          context(), chain->opcode(), chain->type_id(), replacementId,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {var->result_id()}}}));
      // Add the remaining indexes.
      for (uint32_t i = 2; i < chain->NumInOperands(); ++i) {
        Operand copy(chain->GetInOperand(i));
        replacementChain->AddOperand(std::move(copy));
      }
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

void ScalarReplacementPass::CreateReplacementVariables(
    Instruction* inst, std::vector<Instruction*>* replacements) {
  Instruction* type = GetStorageType(inst);

  std::unique_ptr<std::unordered_set<uint64_t>> components_used =
      GetUsedComponents(inst);

  uint32_t elem = 0;
  switch (type->opcode()) {
    case SpvOpTypeStruct:
      type->ForEachInOperand(
          [this, inst, &elem, replacements, &components_used](uint32_t* id) {
            if (!components_used || components_used->count(elem)) {
              CreateVariable(*id, inst, elem, replacements);
            } else {
              replacements->push_back(CreateNullConstant(*id));
            }
            elem++;
          });
      break;
    case SpvOpTypeArray:
      for (uint32_t i = 0; i != GetArrayLength(type); ++i) {
        if (!components_used || components_used->count(i)) {
          CreateVariable(type->GetSingleWordInOperand(0u), inst, i,
                         replacements);
        } else {
          replacements->push_back(
              CreateNullConstant(type->GetSingleWordInOperand(0u)));
        }
      }
      break;

    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
      for (uint32_t i = 0; i != GetNumElements(type); ++i) {
        CreateVariable(type->GetSingleWordInOperand(0u), inst, i, replacements);
      }
      break;

    default:
      assert(false && "Unexpected type.");
      break;
  }

  TransferAnnotations(inst, replacements);
}

void ScalarReplacementPass::TransferAnnotations(
    const Instruction* source, std::vector<Instruction*>* replacements) {
  // Only transfer invariant and restrict decorations on the variable. There are
  // no type or member decorations that are necessary to transfer.
  for (auto inst :
       get_decoration_mgr()->GetDecorationsFor(source->result_id(), false)) {
    assert(inst->opcode() == SpvOpDecorate);
    uint32_t decoration = inst->GetSingleWordInOperand(1u);
    if (decoration == SpvDecorationInvariant ||
        decoration == SpvDecorationRestrict) {
      for (auto var : *replacements) {
        std::unique_ptr<Instruction> annotation(
            new Instruction(context(), SpvOpDecorate, 0, 0,
                            std::initializer_list<Operand>{
                                {SPV_OPERAND_TYPE_ID, {var->result_id()}},
                                {SPV_OPERAND_TYPE_DECORATION, {decoration}}}));
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
    uint32_t typeId, Instruction* varInst, uint32_t index,
    std::vector<Instruction*>* replacements) {
  uint32_t ptrId = GetOrCreatePointerType(typeId);
  uint32_t id = TakeNextId();
  std::unique_ptr<Instruction> variable(new Instruction(
      context(), SpvOpVariable, ptrId, id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}}}));

  BasicBlock* block = context()->get_instr_block(varInst);
  block->begin().InsertBefore(std::move(variable));
  Instruction* inst = &*block->begin();

  // If varInst was initialized, make sure to initialize its replacement.
  GetOrCreateInitialValue(varInst, index, inst);
  get_def_use_mgr()->AnalyzeInstDefUse(inst);
  context()->set_instr_block(inst, block);

  replacements->push_back(inst);
}

uint32_t ScalarReplacementPass::GetOrCreatePointerType(uint32_t id) {
  auto iter = pointee_to_pointer_.find(id);
  if (iter != pointee_to_pointer_.end()) return iter->second;

  analysis::Type* pointeeTy;
  std::unique_ptr<analysis::Pointer> pointerTy;
  std::tie(pointeeTy, pointerTy) =
      context()->get_type_mgr()->GetTypeAndPointerType(id,
                                                       SpvStorageClassFunction);
  uint32_t ptrId = 0;
  if (pointeeTy->IsUniqueType()) {
    // Non-ambiguous type, just ask the type manager for an id.
    ptrId = context()->get_type_mgr()->GetTypeInstruction(pointerTy.get());
    pointee_to_pointer_[id] = ptrId;
    return ptrId;
  }

  // Ambiguous type. We must perform a linear search to try and find the right
  // type.
  for (auto global : context()->types_values()) {
    if (global.opcode() == SpvOpTypePointer &&
        global.GetSingleWordInOperand(0u) == SpvStorageClassFunction &&
        global.GetSingleWordInOperand(1u) == id) {
      if (get_decoration_mgr()->GetDecorationsFor(id, false).empty()) {
        // Only reuse a decoration-less pointer of the correct type.
        ptrId = global.result_id();
        break;
      }
    }
  }

  if (ptrId != 0) {
    pointee_to_pointer_[id] = ptrId;
    return ptrId;
  }

  ptrId = TakeNextId();
  context()->AddType(MakeUnique<Instruction>(
      context(), SpvOpTypePointer, 0, ptrId,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}},
          {SPV_OPERAND_TYPE_ID, {id}}}));
  Instruction* ptr = &*--context()->types_values_end();
  get_def_use_mgr()->AnalyzeInstDefUse(ptr);
  pointee_to_pointer_[id] = ptrId;
  // Register with the type manager if necessary.
  context()->get_type_mgr()->RegisterType(ptrId, *pointerTy);

  return ptrId;
}

void ScalarReplacementPass::GetOrCreateInitialValue(Instruction* source,
                                                    uint32_t index,
                                                    Instruction* newVar) {
  assert(source->opcode() == SpvOpVariable);
  if (source->NumInOperands() < 2) return;

  uint32_t initId = source->GetSingleWordInOperand(1u);
  uint32_t storageId = GetStorageType(newVar)->result_id();
  Instruction* init = get_def_use_mgr()->GetDef(initId);
  uint32_t newInitId = 0;
  // TODO(dnovillo): Refactor this with constant propagation.
  if (init->opcode() == SpvOpConstantNull) {
    // Initialize to appropriate NULL.
    auto iter = type_to_null_.find(storageId);
    if (iter == type_to_null_.end()) {
      newInitId = TakeNextId();
      type_to_null_[storageId] = newInitId;
      context()->AddGlobalValue(
          MakeUnique<Instruction>(context(), SpvOpConstantNull, storageId,
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
        context(), SpvOpSpecConstantOp, storageId, newInitId,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER, {SpvOpCompositeExtract}},
            {SPV_OPERAND_TYPE_ID, {init->result_id()}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}}}));
    Instruction* newSpecConst = &*--context()->types_values_end();
    get_def_use_mgr()->AnalyzeInstDefUse(newSpecConst);
  } else if (init->opcode() == SpvOpConstantComposite) {
    // Get the appropriate index constant.
    newInitId = init->GetSingleWordInOperand(index);
    Instruction* element = get_def_use_mgr()->GetDef(newInitId);
    if (element->opcode() == SpvOpUndef) {
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

size_t ScalarReplacementPass::GetIntegerLiteral(const Operand& op) const {
  assert(op.words.size() <= 2);
  size_t len = 0;
  for (uint32_t i = 0; i != op.words.size(); ++i) {
    len |= (op.words[i] << (32 * i));
  }
  return len;
}

size_t ScalarReplacementPass::GetConstantInteger(
    const Instruction* constant) const {
  assert(get_def_use_mgr()->GetDef(constant->type_id())->opcode() ==
         SpvOpTypeInt);
  assert(constant->opcode() == SpvOpConstant ||
         constant->opcode() == SpvOpConstantNull);
  if (constant->opcode() == SpvOpConstantNull) {
    return 0;
  }

  const Operand& op = constant->GetInOperand(0u);
  return GetIntegerLiteral(op);
}

size_t ScalarReplacementPass::GetArrayLength(
    const Instruction* arrayType) const {
  assert(arrayType->opcode() == SpvOpTypeArray);
  const Instruction* length =
      get_def_use_mgr()->GetDef(arrayType->GetSingleWordInOperand(1u));
  return GetConstantInteger(length);
}

size_t ScalarReplacementPass::GetNumElements(const Instruction* type) const {
  assert(type->opcode() == SpvOpTypeVector ||
         type->opcode() == SpvOpTypeMatrix);
  const Operand& op = type->GetInOperand(1u);
  assert(op.words.size() <= 2);
  size_t len = 0;
  for (uint32_t i = 0; i != op.words.size(); ++i) {
    len |= (op.words[i] << (32 * i));
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
  assert(inst->opcode() == SpvOpVariable);

  uint32_t ptrTypeId = inst->type_id();
  uint32_t typeId =
      get_def_use_mgr()->GetDef(ptrTypeId)->GetSingleWordInOperand(1u);
  return get_def_use_mgr()->GetDef(typeId);
}

bool ScalarReplacementPass::CanReplaceVariable(
    const Instruction* varInst) const {
  assert(varInst->opcode() == SpvOpVariable);

  // Can only replace function scope variables.
  if (varInst->GetSingleWordInOperand(0u) != SpvStorageClassFunction)
    return false;

  if (!CheckTypeAnnotations(get_def_use_mgr()->GetDef(varInst->type_id())))
    return false;

  const Instruction* typeInst = GetStorageType(varInst);
  return CheckType(typeInst) && CheckAnnotations(varInst) && CheckUses(varInst);
}

bool ScalarReplacementPass::CheckType(const Instruction* typeInst) const {
  if (!CheckTypeAnnotations(typeInst)) return false;

  switch (typeInst->opcode()) {
    case SpvOpTypeStruct:
      // Don't bother with empty structs or very large structs.
      if (typeInst->NumInOperands() == 0 ||
          IsLargerThanSizeLimit(typeInst->NumInOperands()))
        return false;
      return true;
    case SpvOpTypeArray:
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
      // case SpvOpTypeMatrix:
      // case SpvOpTypeVector:
      //  if (IsLargerThanSizeLimit(GetNumElements(typeInst))) return false;
      //  return true;

    case SpvOpTypeRuntimeArray:
    default:
      return false;
  }
}

bool ScalarReplacementPass::CheckTypeAnnotations(
    const Instruction* typeInst) const {
  for (auto inst :
       get_decoration_mgr()->GetDecorationsFor(typeInst->result_id(), false)) {
    uint32_t decoration;
    if (inst->opcode() == SpvOpDecorate) {
      decoration = inst->GetSingleWordInOperand(1u);
    } else {
      assert(inst->opcode() == SpvOpMemberDecorate);
      decoration = inst->GetSingleWordInOperand(2u);
    }

    switch (decoration) {
      case SpvDecorationRowMajor:
      case SpvDecorationColMajor:
      case SpvDecorationArrayStride:
      case SpvDecorationMatrixStride:
      case SpvDecorationCPacked:
      case SpvDecorationInvariant:
      case SpvDecorationRestrict:
      case SpvDecorationOffset:
      case SpvDecorationAlignment:
      case SpvDecorationAlignmentId:
      case SpvDecorationMaxByteOffset:
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
    assert(inst->opcode() == SpvOpDecorate);
    uint32_t decoration = inst->GetSingleWordInOperand(1u);
    switch (decoration) {
      case SpvDecorationInvariant:
      case SpvDecorationRestrict:
      case SpvDecorationAlignment:
      case SpvDecorationAlignmentId:
      case SpvDecorationMaxByteOffset:
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
  bool ok = true;
  get_def_use_mgr()->ForEachUse(
      inst, [this, stats, &ok](const Instruction* user, uint32_t index) {
        // Annotations are check as a group separately.
        if (!IsAnnotationInst(user->opcode())) {
          switch (user->opcode()) {
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
              if (index == 2u && user->NumInOperands() > 1) {
                uint32_t id = user->GetSingleWordInOperand(1u);
                const Instruction* opInst = get_def_use_mgr()->GetDef(id);
                if (!IsCompileTimeConstantInst(opInst->opcode())) {
                  ok = false;
                } else {
                  if (!CheckUsesRelaxed(user)) ok = false;
                }
                stats->num_partial_accesses++;
              } else {
                ok = false;
              }
              break;
            case SpvOpLoad:
              if (!CheckLoad(user, index)) ok = false;
              stats->num_full_accesses++;
              break;
            case SpvOpStore:
              if (!CheckStore(user, index)) ok = false;
              stats->num_full_accesses++;
              break;
            case SpvOpName:
            case SpvOpMemberName:
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
          case SpvOpAccessChain:
          case SpvOpInBoundsAccessChain:
            if (index != 2u) {
              ok = false;
            } else {
              if (!CheckUsesRelaxed(user)) ok = false;
            }
            break;
          case SpvOpLoad:
            if (!CheckLoad(user, index)) ok = false;
            break;
          case SpvOpStore:
            if (!CheckStore(user, index)) ok = false;
            break;
          default:
            ok = false;
            break;
        }
      });

  return ok;
}

bool ScalarReplacementPass::CheckLoad(const Instruction* inst,
                                      uint32_t index) const {
  if (index != 2u) return false;
  if (inst->NumInOperands() >= 2 &&
      inst->GetSingleWordInOperand(1u) & SpvMemoryAccessVolatileMask)
    return false;
  return true;
}

bool ScalarReplacementPass::CheckStore(const Instruction* inst,
                                       uint32_t index) const {
  if (index != 0u) return false;
  if (inst->NumInOperands() >= 3 &&
      inst->GetSingleWordInOperand(2u) & SpvMemoryAccessVolatileMask)
    return false;
  return true;
}
bool ScalarReplacementPass::IsLargerThanSizeLimit(size_t length) const {
  if (max_num_elements_ == 0) {
    return false;
  }
  return length > max_num_elements_;
}

std::unique_ptr<std::unordered_set<uint64_t>>
ScalarReplacementPass::GetUsedComponents(Instruction* inst) {
  std::unique_ptr<std::unordered_set<uint64_t>> result(
      new std::unordered_set<uint64_t>());

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  def_use_mgr->WhileEachUser(inst, [&result, def_use_mgr,
                                    this](Instruction* use) {
    switch (use->opcode()) {
      case SpvOpLoad: {
        // Look for extract from the load.
        std::vector<uint32_t> t;
        if (def_use_mgr->WhileEachUser(use, [&t](Instruction* use2) {
              if (use2->opcode() != SpvOpCompositeExtract) {
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
      case SpvOpName:
      case SpvOpMemberName:
      case SpvOpStore:
        // No components are used.
        return true;
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain: {
        // Add the first index it if is a constant.
        // TODO: Could be improved by checking if the address is used in a load.
        analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
        uint32_t index_id = use->GetSingleWordInOperand(1);
        const analysis::Constant* index_const =
            const_mgr->FindDeclaredConstant(index_id);
        if (index_const) {
          const analysis::Integer* index_type =
              index_const->type()->AsInteger();
          assert(index_type);
          if (index_type->width() == 32) {
            result->insert(index_const->GetU32());
            return true;
          } else if (index_type->width() == 64) {
            result->insert(index_const->GetU64());
            return true;
          }
          result.reset(nullptr);
          return false;
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

Instruction* ScalarReplacementPass::CreateNullConstant(uint32_t type_id) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  const analysis::Type* type = type_mgr->GetType(type_id);
  const analysis::Constant* null_const = const_mgr->GetConstant(type, {});
  Instruction* null_inst =
      const_mgr->GetDefiningInstruction(null_const, type_id);
  context()->UpdateDefUse(null_inst);
  return null_inst;
}

}  // namespace opt
}  // namespace spvtools
