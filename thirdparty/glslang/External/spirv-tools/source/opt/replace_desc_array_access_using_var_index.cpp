// Copyright (c) 2021 Google LLC
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

#include "source/opt/replace_desc_array_access_using_var_index.h"

#include "source/opt/desc_sroa_util.h"
#include "source/opt/ir_builder.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kOpAccessChainInOperandIndexes = 1;
constexpr uint32_t kOpTypePointerInOperandType = 1;
constexpr uint32_t kOpTypeArrayInOperandType = 0;
constexpr uint32_t kOpTypeStructInOperandMember = 0;
IRContext::Analysis kAnalysisDefUseAndInstrToBlockMapping =
    IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping;

uint32_t GetValueWithKeyExistenceCheck(
    uint32_t key, const std::unordered_map<uint32_t, uint32_t>& map) {
  auto itr = map.find(key);
  assert(itr != map.end() && "Key does not exist");
  return itr->second;
}

}  // namespace

Pass::Status ReplaceDescArrayAccessUsingVarIndex::Process() {
  Status status = Status::SuccessWithoutChange;
  for (Instruction& var : context()->types_values()) {
    if (descsroautil::IsDescriptorArray(context(), &var)) {
      if (ReplaceVariableAccessesWithConstantElements(&var))
        status = Status::SuccessWithChange;
    }
  }
  return status;
}

bool ReplaceDescArrayAccessUsingVarIndex::
    ReplaceVariableAccessesWithConstantElements(Instruction* var) const {
  std::vector<Instruction*> work_list;
  get_def_use_mgr()->ForEachUser(var, [&work_list](Instruction* use) {
    switch (use->opcode()) {
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
        work_list.push_back(use);
        break;
      default:
        break;
    }
  });

  bool updated = false;
  for (Instruction* access_chain : work_list) {
    if (descsroautil::GetAccessChainIndexAsConst(context(), access_chain) ==
        nullptr) {
      ReplaceAccessChain(var, access_chain);
      updated = true;
    }
  }
  // Note that we do not consider OpLoad and OpCompositeExtract because
  // OpCompositeExtract always has constant literals for indices.
  return updated;
}

void ReplaceDescArrayAccessUsingVarIndex::ReplaceAccessChain(
    Instruction* var, Instruction* access_chain) const {
  uint32_t number_of_elements =
      descsroautil::GetNumberOfElementsForArrayOrStruct(context(), var);
  assert(number_of_elements != 0 && "Number of element is 0");
  if (number_of_elements == 1) {
    UseConstIndexForAccessChain(access_chain, 0);
    get_def_use_mgr()->AnalyzeInstUse(access_chain);
    return;
  }
  ReplaceUsersOfAccessChain(access_chain, number_of_elements);
}

void ReplaceDescArrayAccessUsingVarIndex::ReplaceUsersOfAccessChain(
    Instruction* access_chain, uint32_t number_of_elements) const {
  std::vector<Instruction*> final_users;
  CollectRecursiveUsersWithConcreteType(access_chain, &final_users);
  for (auto* inst : final_users) {
    std::deque<Instruction*> insts_to_be_cloned =
        CollectRequiredImageAndAccessInsts(inst);
    ReplaceNonUniformAccessWithSwitchCase(
        inst, access_chain, number_of_elements, insts_to_be_cloned);
  }
}

void ReplaceDescArrayAccessUsingVarIndex::CollectRecursiveUsersWithConcreteType(
    Instruction* access_chain, std::vector<Instruction*>* final_users) const {
  std::queue<Instruction*> work_list;
  work_list.push(access_chain);
  while (!work_list.empty()) {
    auto* inst_from_work_list = work_list.front();
    work_list.pop();
    get_def_use_mgr()->ForEachUser(
        inst_from_work_list, [this, final_users, &work_list](Instruction* use) {
          // TODO: Support Boolean type as well.
          if (!use->HasResultId() || IsConcreteType(use->type_id())) {
            final_users->push_back(use);
          } else {
            work_list.push(use);
          }
        });
  }
}

std::deque<Instruction*>
ReplaceDescArrayAccessUsingVarIndex::CollectRequiredImageAndAccessInsts(
    Instruction* user) const {
  std::unordered_set<uint32_t> seen_inst_ids;
  std::queue<Instruction*> work_list;

  auto decision_to_include_operand = [this, &seen_inst_ids,
                                      &work_list](uint32_t* idp) {
    if (!seen_inst_ids.insert(*idp).second) return;
    Instruction* operand = get_def_use_mgr()->GetDef(*idp);
    if (context()->get_instr_block(operand) != nullptr &&
        (HasImageOrImagePtrType(operand) ||
         operand->opcode() == spv::Op::OpAccessChain ||
         operand->opcode() == spv::Op::OpInBoundsAccessChain)) {
      work_list.push(operand);
    }
  };

  std::deque<Instruction*> required_insts;
  required_insts.push_front(user);
  user->ForEachInId(decision_to_include_operand);
  while (!work_list.empty()) {
    auto* inst_from_work_list = work_list.front();
    work_list.pop();
    required_insts.push_front(inst_from_work_list);
    inst_from_work_list->ForEachInId(decision_to_include_operand);
  }
  return required_insts;
}

bool ReplaceDescArrayAccessUsingVarIndex::HasImageOrImagePtrType(
    const Instruction* inst) const {
  assert(inst != nullptr && inst->type_id() != 0 && "Invalid instruction");
  return IsImageOrImagePtrType(get_def_use_mgr()->GetDef(inst->type_id()));
}

bool ReplaceDescArrayAccessUsingVarIndex::IsImageOrImagePtrType(
    const Instruction* type_inst) const {
  if (type_inst->opcode() == spv::Op::OpTypeImage ||
      type_inst->opcode() == spv::Op::OpTypeSampler ||
      type_inst->opcode() == spv::Op::OpTypeSampledImage) {
    return true;
  }
  if (type_inst->opcode() == spv::Op::OpTypePointer) {
    Instruction* pointee_type_inst = get_def_use_mgr()->GetDef(
        type_inst->GetSingleWordInOperand(kOpTypePointerInOperandType));
    return IsImageOrImagePtrType(pointee_type_inst);
  }
  if (type_inst->opcode() == spv::Op::OpTypeArray) {
    Instruction* element_type_inst = get_def_use_mgr()->GetDef(
        type_inst->GetSingleWordInOperand(kOpTypeArrayInOperandType));
    return IsImageOrImagePtrType(element_type_inst);
  }
  if (type_inst->opcode() != spv::Op::OpTypeStruct) return false;
  for (uint32_t in_operand_idx = kOpTypeStructInOperandMember;
       in_operand_idx < type_inst->NumInOperands(); ++in_operand_idx) {
    Instruction* member_type_inst = get_def_use_mgr()->GetDef(
        type_inst->GetSingleWordInOperand(kOpTypeStructInOperandMember));
    if (IsImageOrImagePtrType(member_type_inst)) return true;
  }
  return false;
}

bool ReplaceDescArrayAccessUsingVarIndex::IsConcreteType(
    uint32_t type_id) const {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  if (type_inst->opcode() == spv::Op::OpTypeInt ||
      type_inst->opcode() == spv::Op::OpTypeFloat) {
    return true;
  }
  if (type_inst->opcode() == spv::Op::OpTypeVector ||
      type_inst->opcode() == spv::Op::OpTypeMatrix ||
      type_inst->opcode() == spv::Op::OpTypeArray) {
    return IsConcreteType(type_inst->GetSingleWordInOperand(0));
  }
  if (type_inst->opcode() == spv::Op::OpTypeStruct) {
    for (uint32_t i = 0; i < type_inst->NumInOperands(); ++i) {
      if (!IsConcreteType(type_inst->GetSingleWordInOperand(i))) return false;
    }
    return true;
  }
  return false;
}

BasicBlock* ReplaceDescArrayAccessUsingVarIndex::CreateCaseBlock(
    Instruction* access_chain, uint32_t element_index,
    const std::deque<Instruction*>& insts_to_be_cloned,
    uint32_t branch_target_id,
    std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const {
  auto* case_block = CreateNewBlock();
  AddConstElementAccessToCaseBlock(case_block, access_chain, element_index,
                                   old_ids_to_new_ids);
  CloneInstsToBlock(case_block, access_chain, insts_to_be_cloned,
                    old_ids_to_new_ids);
  AddBranchToBlock(case_block, branch_target_id);
  UseNewIdsInBlock(case_block, *old_ids_to_new_ids);
  return case_block;
}

void ReplaceDescArrayAccessUsingVarIndex::CloneInstsToBlock(
    BasicBlock* block, Instruction* inst_to_skip_cloning,
    const std::deque<Instruction*>& insts_to_be_cloned,
    std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const {
  for (auto* inst_to_be_cloned : insts_to_be_cloned) {
    if (inst_to_be_cloned == inst_to_skip_cloning) continue;
    std::unique_ptr<Instruction> clone(inst_to_be_cloned->Clone(context()));
    if (inst_to_be_cloned->HasResultId()) {
      uint32_t new_id = context()->TakeNextId();
      clone->SetResultId(new_id);
      (*old_ids_to_new_ids)[inst_to_be_cloned->result_id()] = new_id;
    }
    get_def_use_mgr()->AnalyzeInstDefUse(clone.get());
    context()->set_instr_block(clone.get(), block);
    block->AddInstruction(std::move(clone));
  }
}

void ReplaceDescArrayAccessUsingVarIndex::UseNewIdsInBlock(
    BasicBlock* block,
    const std::unordered_map<uint32_t, uint32_t>& old_ids_to_new_ids) const {
  for (auto block_itr = block->begin(); block_itr != block->end();
       ++block_itr) {
    (&*block_itr)->ForEachInId([&old_ids_to_new_ids](uint32_t* idp) {
      auto old_ids_to_new_ids_itr = old_ids_to_new_ids.find(*idp);
      if (old_ids_to_new_ids_itr == old_ids_to_new_ids.end()) return;
      *idp = old_ids_to_new_ids_itr->second;
    });
    get_def_use_mgr()->AnalyzeInstUse(&*block_itr);
  }
}

void ReplaceDescArrayAccessUsingVarIndex::ReplaceNonUniformAccessWithSwitchCase(
    Instruction* access_chain_final_user, Instruction* access_chain,
    uint32_t number_of_elements,
    const std::deque<Instruction*>& insts_to_be_cloned) const {
  auto* block = context()->get_instr_block(access_chain_final_user);
  // If the instruction does not belong to a block (i.e. in the case of
  // OpDecorate), no replacement is needed.
  if (!block) return;

  // Create merge block and add terminator
  auto* merge_block = SeparateInstructionsIntoNewBlock(
      block, access_chain_final_user->NextNode());

  auto* function = block->GetParent();

  // Add case blocks
  std::vector<uint32_t> phi_operands;
  std::vector<uint32_t> case_block_ids;
  for (uint32_t idx = 0; idx < number_of_elements; ++idx) {
    std::unordered_map<uint32_t, uint32_t> old_ids_to_new_ids_for_cloned_insts;
    std::unique_ptr<BasicBlock> case_block(CreateCaseBlock(
        access_chain, idx, insts_to_be_cloned, merge_block->id(),
        &old_ids_to_new_ids_for_cloned_insts));
    case_block_ids.push_back(case_block->id());
    function->InsertBasicBlockBefore(std::move(case_block), merge_block);

    // Keep the operand for OpPhi
    if (!access_chain_final_user->HasResultId()) continue;
    uint32_t phi_operand =
        GetValueWithKeyExistenceCheck(access_chain_final_user->result_id(),
                                      old_ids_to_new_ids_for_cloned_insts);
    phi_operands.push_back(phi_operand);
  }

  // Create default block
  std::unique_ptr<BasicBlock> default_block(
      CreateDefaultBlock(access_chain_final_user->HasResultId(), &phi_operands,
                         merge_block->id()));
  uint32_t default_block_id = default_block->id();
  function->InsertBasicBlockBefore(std::move(default_block), merge_block);

  // Create OpSwitch
  uint32_t access_chain_index_var_id =
      descsroautil::GetFirstIndexOfAccessChain(access_chain);
  AddSwitchForAccessChain(block, access_chain_index_var_id, default_block_id,
                          merge_block->id(), case_block_ids);

  // Create phi instructions
  if (!phi_operands.empty()) {
    uint32_t phi_id = CreatePhiInstruction(merge_block, phi_operands,
                                           case_block_ids, default_block_id);
    context()->ReplaceAllUsesWith(access_chain_final_user->result_id(), phi_id);
  }

  // Replace OpPhi incoming block operand that uses |block| with |merge_block|
  ReplacePhiIncomingBlock(block->id(), merge_block->id());
}

BasicBlock*
ReplaceDescArrayAccessUsingVarIndex::SeparateInstructionsIntoNewBlock(
    BasicBlock* block, Instruction* separation_begin_inst) const {
  auto separation_begin = block->begin();
  while (separation_begin != block->end() &&
         &*separation_begin != separation_begin_inst) {
    ++separation_begin;
  }
  return block->SplitBasicBlock(context(), context()->TakeNextId(),
                                separation_begin);
}

BasicBlock* ReplaceDescArrayAccessUsingVarIndex::CreateNewBlock() const {
  auto* new_block = new BasicBlock(std::unique_ptr<Instruction>(new Instruction(
      context(), spv::Op::OpLabel, 0, context()->TakeNextId(), {})));
  get_def_use_mgr()->AnalyzeInstDefUse(new_block->GetLabelInst());
  context()->set_instr_block(new_block->GetLabelInst(), new_block);
  return new_block;
}

void ReplaceDescArrayAccessUsingVarIndex::UseConstIndexForAccessChain(
    Instruction* access_chain, uint32_t const_element_idx) const {
  uint32_t const_element_idx_id =
      context()->get_constant_mgr()->GetUIntConstId(const_element_idx);
  access_chain->SetInOperand(kOpAccessChainInOperandIndexes,
                             {const_element_idx_id});
}

void ReplaceDescArrayAccessUsingVarIndex::AddConstElementAccessToCaseBlock(
    BasicBlock* case_block, Instruction* access_chain,
    uint32_t const_element_idx,
    std::unordered_map<uint32_t, uint32_t>* old_ids_to_new_ids) const {
  std::unique_ptr<Instruction> access_clone(access_chain->Clone(context()));
  UseConstIndexForAccessChain(access_clone.get(), const_element_idx);

  uint32_t new_access_id = context()->TakeNextId();
  (*old_ids_to_new_ids)[access_clone->result_id()] = new_access_id;
  access_clone->SetResultId(new_access_id);
  get_def_use_mgr()->AnalyzeInstDefUse(access_clone.get());

  context()->set_instr_block(access_clone.get(), case_block);
  case_block->AddInstruction(std::move(access_clone));
}

void ReplaceDescArrayAccessUsingVarIndex::AddBranchToBlock(
    BasicBlock* parent_block, uint32_t branch_destination) const {
  InstructionBuilder builder{context(), parent_block,
                             kAnalysisDefUseAndInstrToBlockMapping};
  builder.AddBranch(branch_destination);
}

BasicBlock* ReplaceDescArrayAccessUsingVarIndex::CreateDefaultBlock(
    bool null_const_for_phi_is_needed, std::vector<uint32_t>* phi_operands,
    uint32_t merge_block_id) const {
  auto* default_block = CreateNewBlock();
  AddBranchToBlock(default_block, merge_block_id);
  if (!null_const_for_phi_is_needed) return default_block;

  // Create null value for OpPhi
  Instruction* inst = context()->get_def_use_mgr()->GetDef((*phi_operands)[0]);
  auto* null_const_inst = GetConstNull(inst->type_id());
  phi_operands->push_back(null_const_inst->result_id());
  return default_block;
}

Instruction* ReplaceDescArrayAccessUsingVarIndex::GetConstNull(
    uint32_t type_id) const {
  assert(type_id != 0 && "Result type is expected");
  auto* type = context()->get_type_mgr()->GetType(type_id);
  auto* null_const = context()->get_constant_mgr()->GetConstant(type, {});
  return context()->get_constant_mgr()->GetDefiningInstruction(null_const);
}

void ReplaceDescArrayAccessUsingVarIndex::AddSwitchForAccessChain(
    BasicBlock* parent_block, uint32_t access_chain_index_var_id,
    uint32_t default_id, uint32_t merge_id,
    const std::vector<uint32_t>& case_block_ids) const {
  InstructionBuilder builder{context(), parent_block,
                             kAnalysisDefUseAndInstrToBlockMapping};
  std::vector<std::pair<Operand::OperandData, uint32_t>> cases;
  for (uint32_t i = 0; i < static_cast<uint32_t>(case_block_ids.size()); ++i) {
    cases.emplace_back(Operand::OperandData{i}, case_block_ids[i]);
  }
  builder.AddSwitch(access_chain_index_var_id, default_id, cases, merge_id);
}

uint32_t ReplaceDescArrayAccessUsingVarIndex::CreatePhiInstruction(
    BasicBlock* parent_block, const std::vector<uint32_t>& phi_operands,
    const std::vector<uint32_t>& case_block_ids,
    uint32_t default_block_id) const {
  std::vector<uint32_t> incomings;
  assert(case_block_ids.size() + 1 == phi_operands.size() &&
         "Number of Phi operands must be exactly 1 bigger than the one of case "
         "blocks");
  for (size_t i = 0; i < case_block_ids.size(); ++i) {
    incomings.push_back(phi_operands[i]);
    incomings.push_back(case_block_ids[i]);
  }
  incomings.push_back(phi_operands.back());
  incomings.push_back(default_block_id);

  InstructionBuilder builder{context(), &*parent_block->begin(),
                             kAnalysisDefUseAndInstrToBlockMapping};
  uint32_t phi_result_type_id =
      context()->get_def_use_mgr()->GetDef(phi_operands[0])->type_id();
  auto* phi = builder.AddPhi(phi_result_type_id, incomings);
  return phi->result_id();
}

void ReplaceDescArrayAccessUsingVarIndex::ReplacePhiIncomingBlock(
    uint32_t old_incoming_block_id, uint32_t new_incoming_block_id) const {
  context()->ReplaceAllUsesWithPredicate(
      old_incoming_block_id, new_incoming_block_id,
      [](Instruction* use) { return use->opcode() == spv::Op::OpPhi; });
}

}  // namespace opt
}  // namespace spvtools
