// Copyright (c) 2018 Google LLC
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

#include "upgrade_memory_model.h"

#include <utility>

#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/spirv_constant.h"
#include "source/util/make_unique.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {

Pass::Status UpgradeMemoryModel::Process() {
  // TODO: This pass needs changes to support cooperative matrices.
  if (context()->get_feature_mgr()->HasCapability(
          spv::Capability::CooperativeMatrixNV)) {
    return Pass::Status::SuccessWithoutChange;
  }

  // Only update Logical GLSL450 to Logical VulkanKHR.
  Instruction* memory_model = get_module()->GetMemoryModel();
  if (memory_model->GetSingleWordInOperand(0u) !=
          uint32_t(spv::AddressingModel::Logical) ||
      memory_model->GetSingleWordInOperand(1u) !=
          uint32_t(spv::MemoryModel::GLSL450)) {
    return Pass::Status::SuccessWithoutChange;
  }

  UpgradeMemoryModelInstruction();
  UpgradeInstructions();
  CleanupDecorations();
  UpgradeBarriers();
  UpgradeMemoryScope();

  return Pass::Status::SuccessWithChange;
}

void UpgradeMemoryModel::UpgradeMemoryModelInstruction() {
  // Overall changes necessary:
  // 1. Add the OpExtension.
  // 2. Add the OpCapability.
  // 3. Modify the memory model.
  Instruction* memory_model = get_module()->GetMemoryModel();
  context()->AddCapability(MakeUnique<Instruction>(
      context(), spv::Op::OpCapability, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_CAPABILITY,
           {uint32_t(spv::Capability::VulkanMemoryModelKHR)}}}));
  const std::string extension = "SPV_KHR_vulkan_memory_model";
  std::vector<uint32_t> words = spvtools::utils::MakeVector(extension);
  context()->AddExtension(
      MakeUnique<Instruction>(context(), spv::Op::OpExtension, 0, 0,
                              std::initializer_list<Operand>{
                                  {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
  memory_model->SetInOperand(1u, {uint32_t(spv::MemoryModel::VulkanKHR)});
}

void UpgradeMemoryModel::UpgradeInstructions() {
  // Coherent and Volatile decorations are deprecated. Remove them and replace
  // with flags on the memory/image operations. The decorations can occur on
  // OpVariable, OpFunctionParameter (of pointer type) and OpStructType (member
  // decoration). Trace from the decoration target(s) to the final memory/image
  // instructions. Additionally, Workgroup storage class variables and function
  // parameters are implicitly coherent in GLSL450.

  // Upgrade modf and frexp first since they generate new stores.
  // In SPIR-V 1.4 or later, normalize OpCopyMemory* access operands.
  for (auto& func : *get_module()) {
    func.ForEachInst([this](Instruction* inst) {
      if (inst->opcode() == spv::Op::OpExtInst) {
        auto ext_inst = inst->GetSingleWordInOperand(1u);
        if (ext_inst == GLSLstd450Modf || ext_inst == GLSLstd450Frexp) {
          auto import =
              get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
          if (import->GetInOperand(0u).AsString() == "GLSL.std.450") {
            UpgradeExtInst(inst);
          }
        }
      } else if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
        if (inst->opcode() == spv::Op::OpCopyMemory ||
            inst->opcode() == spv::Op::OpCopyMemorySized) {
          uint32_t start_operand =
              inst->opcode() == spv::Op::OpCopyMemory ? 2u : 3u;
          if (inst->NumInOperands() > start_operand) {
            auto num_access_words = MemoryAccessNumWords(
                inst->GetSingleWordInOperand(start_operand));
            if ((num_access_words + start_operand) == inst->NumInOperands()) {
              // There is a single memory access operand. Duplicate it to have a
              // separate operand for both source and target.
              for (uint32_t i = 0; i < num_access_words; ++i) {
                auto operand = inst->GetInOperand(start_operand + i);
                inst->AddOperand(std::move(operand));
              }
            }
          } else {
            // Add two memory access operands.
            inst->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                              {uint32_t(spv::MemoryAccessMask::MaskNone)}});
            inst->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                              {uint32_t(spv::MemoryAccessMask::MaskNone)}});
          }
        }
      }
    });
  }

  UpgradeMemoryAndImages();
  UpgradeAtomics();
}

void UpgradeMemoryModel::UpgradeMemoryAndImages() {
  for (auto& func : *get_module()) {
    func.ForEachInst([this](Instruction* inst) {
      bool is_coherent = false;
      bool is_volatile = false;
      bool src_coherent = false;
      bool src_volatile = false;
      bool dst_coherent = false;
      bool dst_volatile = false;
      uint32_t start_operand = 0u;
      spv::Scope scope = spv::Scope::QueueFamilyKHR;
      spv::Scope src_scope = spv::Scope::QueueFamilyKHR;
      spv::Scope dst_scope = spv::Scope::QueueFamilyKHR;
      switch (inst->opcode()) {
        case spv::Op::OpLoad:
        case spv::Op::OpStore:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case spv::Op::OpImageRead:
        case spv::Op::OpImageSparseRead:
        case spv::Op::OpImageWrite:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case spv::Op::OpCopyMemory:
        case spv::Op::OpCopyMemorySized:
          std::tie(dst_coherent, dst_volatile, dst_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          std::tie(src_coherent, src_volatile, src_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(1u));
          break;
        default:
          break;
      }

      switch (inst->opcode()) {
        case spv::Op::OpLoad:
          UpgradeFlags(inst, 1u, is_coherent, is_volatile, kVisibility,
                       kMemory);
          break;
        case spv::Op::OpStore:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kAvailability,
                       kMemory);
          break;
        case spv::Op::OpCopyMemory:
        case spv::Op::OpCopyMemorySized:
          start_operand = inst->opcode() == spv::Op::OpCopyMemory ? 2u : 3u;
          if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
            // There are guaranteed to be two memory access operands at this
            // point so treat source and target separately.
            uint32_t num_access_words = MemoryAccessNumWords(
                inst->GetSingleWordInOperand(start_operand));
            UpgradeFlags(inst, start_operand, dst_coherent, dst_volatile,
                         kAvailability, kMemory);
            UpgradeFlags(inst, start_operand + num_access_words, src_coherent,
                         src_volatile, kVisibility, kMemory);
          } else {
            UpgradeFlags(inst, start_operand, dst_coherent, dst_volatile,
                         kAvailability, kMemory);
            UpgradeFlags(inst, start_operand, src_coherent, src_volatile,
                         kVisibility, kMemory);
          }
          break;
        case spv::Op::OpImageRead:
        case spv::Op::OpImageSparseRead:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kVisibility, kImage);
          break;
        case spv::Op::OpImageWrite:
          UpgradeFlags(inst, 3u, is_coherent, is_volatile, kAvailability,
                       kImage);
          break;
        default:
          break;
      }

      // |is_coherent| is never used for the same instructions as
      // |src_coherent| and |dst_coherent|.
      if (is_coherent) {
        inst->AddOperand(
            {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(scope)}});
      }
      if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
        // There are two memory access operands. The first is for the target and
        // the second is for the source.
        if (dst_coherent || src_coherent) {
          start_operand = inst->opcode() == spv::Op::OpCopyMemory ? 2u : 3u;
          std::vector<Operand> new_operands;
          uint32_t num_access_words =
              MemoryAccessNumWords(inst->GetSingleWordInOperand(start_operand));
          // The flags were already updated so subtract if we're adding a
          // scope.
          if (dst_coherent) --num_access_words;
          for (uint32_t i = 0; i < start_operand + num_access_words; ++i) {
            new_operands.push_back(inst->GetInOperand(i));
          }
          // Add the target scope if necessary.
          if (dst_coherent) {
            new_operands.push_back(
                {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(dst_scope)}});
          }
          // Copy the remaining current operands.
          for (uint32_t i = start_operand + num_access_words;
               i < inst->NumInOperands(); ++i) {
            new_operands.push_back(inst->GetInOperand(i));
          }
          // Add the source scope if necessary.
          if (src_coherent) {
            new_operands.push_back(
                {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(src_scope)}});
          }
          inst->SetInOperands(std::move(new_operands));
        }
      } else {
        // According to SPV_KHR_vulkan_memory_model, if both available and
        // visible flags are used the first scope operand is for availability
        // (writes) and the second is for visibility (reads).
        if (dst_coherent) {
          inst->AddOperand(
              {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(dst_scope)}});
        }
        if (src_coherent) {
          inst->AddOperand(
              {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(src_scope)}});
        }
      }
    });
  }
}

void UpgradeMemoryModel::UpgradeAtomics() {
  for (auto& func : *get_module()) {
    func.ForEachInst([this](Instruction* inst) {
      if (spvOpcodeIsAtomicOp(inst->opcode())) {
        bool unused_coherent = false;
        bool is_volatile = false;
        spv::Scope unused_scope = spv::Scope::QueueFamilyKHR;
        std::tie(unused_coherent, is_volatile, unused_scope) =
            GetInstructionAttributes(inst->GetSingleWordInOperand(0));

        UpgradeSemantics(inst, 2u, is_volatile);
        if (inst->opcode() == spv::Op::OpAtomicCompareExchange ||
            inst->opcode() == spv::Op::OpAtomicCompareExchangeWeak) {
          UpgradeSemantics(inst, 3u, is_volatile);
        }
      }
    });
  }
}

void UpgradeMemoryModel::UpgradeSemantics(Instruction* inst,
                                          uint32_t in_operand,
                                          bool is_volatile) {
  if (!is_volatile) return;

  uint32_t semantics_id = inst->GetSingleWordInOperand(in_operand);
  const analysis::Constant* constant =
      context()->get_constant_mgr()->FindDeclaredConstant(semantics_id);
  const analysis::Integer* type = constant->type()->AsInteger();
  assert(type && type->width() == 32);
  uint32_t value = 0;
  if (type->IsSigned()) {
    value = static_cast<uint32_t>(constant->GetS32());
  } else {
    value = constant->GetU32();
  }

  value |= uint32_t(spv::MemorySemanticsMask::Volatile);
  auto new_constant = context()->get_constant_mgr()->GetConstant(type, {value});
  auto new_semantics =
      context()->get_constant_mgr()->GetDefiningInstruction(new_constant);
  inst->SetInOperand(in_operand, {new_semantics->result_id()});
}

std::tuple<bool, bool, spv::Scope> UpgradeMemoryModel::GetInstructionAttributes(
    uint32_t id) {
  // |id| is a pointer used in a memory/image instruction. Need to determine if
  // that pointer points to volatile or coherent memory. Workgroup storage
  // class is implicitly coherent and cannot be decorated with volatile, so
  // short circuit that case.
  Instruction* inst = context()->get_def_use_mgr()->GetDef(id);
  analysis::Type* type = context()->get_type_mgr()->GetType(inst->type_id());
  if (type->AsPointer() &&
      type->AsPointer()->storage_class() == spv::StorageClass::Workgroup) {
    return std::make_tuple(true, false, spv::Scope::Workgroup);
  }

  bool is_coherent = false;
  bool is_volatile = false;
  std::unordered_set<uint32_t> visited;
  std::tie(is_coherent, is_volatile) =
      TraceInstruction(context()->get_def_use_mgr()->GetDef(id),
                       std::vector<uint32_t>(), &visited);

  return std::make_tuple(is_coherent, is_volatile, spv::Scope::QueueFamilyKHR);
}

std::pair<bool, bool> UpgradeMemoryModel::TraceInstruction(
    Instruction* inst, std::vector<uint32_t> indices,
    std::unordered_set<uint32_t>* visited) {
  auto iter = cache_.find(std::make_pair(inst->result_id(), indices));
  if (iter != cache_.end()) {
    return iter->second;
  }

  if (!visited->insert(inst->result_id()).second) {
    return std::make_pair(false, false);
  }

  // Initialize the cache before |indices| is (potentially) modified.
  auto& cached_result = cache_[std::make_pair(inst->result_id(), indices)];
  cached_result.first = false;
  cached_result.second = false;

  bool is_coherent = false;
  bool is_volatile = false;
  switch (inst->opcode()) {
    case spv::Op::OpVariable:
    case spv::Op::OpFunctionParameter:
      is_coherent |= HasDecoration(inst, 0, spv::Decoration::Coherent);
      is_volatile |= HasDecoration(inst, 0, spv::Decoration::Volatile);
      if (!is_coherent || !is_volatile) {
        bool type_coherent = false;
        bool type_volatile = false;
        std::tie(type_coherent, type_volatile) =
            CheckType(inst->type_id(), indices);
        is_coherent |= type_coherent;
        is_volatile |= type_volatile;
      }
      break;
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
      // Store indices in reverse order.
      for (uint32_t i = inst->NumInOperands() - 1; i > 0; --i) {
        indices.push_back(inst->GetSingleWordInOperand(i));
      }
      break;
    case spv::Op::OpPtrAccessChain:
      // Store indices in reverse order. Skip the |Element| operand.
      for (uint32_t i = inst->NumInOperands() - 1; i > 1; --i) {
        indices.push_back(inst->GetSingleWordInOperand(i));
      }
      break;
    default:
      break;
  }

  // No point searching further.
  if (is_coherent && is_volatile) {
    cached_result.first = true;
    cached_result.second = true;
    return std::make_pair(true, true);
  }

  // Variables and function parameters are sources. Continue searching until we
  // reach them.
  if (inst->opcode() != spv::Op::OpVariable &&
      inst->opcode() != spv::Op::OpFunctionParameter) {
    inst->ForEachInId([this, &is_coherent, &is_volatile, &indices,
                       &visited](const uint32_t* id_ptr) {
      Instruction* op_inst = context()->get_def_use_mgr()->GetDef(*id_ptr);
      const analysis::Type* type =
          context()->get_type_mgr()->GetType(op_inst->type_id());
      if (type &&
          (type->AsPointer() || type->AsImage() || type->AsSampledImage())) {
        bool operand_coherent = false;
        bool operand_volatile = false;
        std::tie(operand_coherent, operand_volatile) =
            TraceInstruction(op_inst, indices, visited);
        is_coherent |= operand_coherent;
        is_volatile |= operand_volatile;
      }
    });
  }

  cached_result.first = is_coherent;
  cached_result.second = is_volatile;
  return std::make_pair(is_coherent, is_volatile);
}

std::pair<bool, bool> UpgradeMemoryModel::CheckType(
    uint32_t type_id, const std::vector<uint32_t>& indices) {
  bool is_coherent = false;
  bool is_volatile = false;
  Instruction* type_inst = context()->get_def_use_mgr()->GetDef(type_id);
  assert(type_inst->opcode() == spv::Op::OpTypePointer);
  Instruction* element_inst = context()->get_def_use_mgr()->GetDef(
      type_inst->GetSingleWordInOperand(1u));
  for (int i = (int)indices.size() - 1; i >= 0; --i) {
    if (is_coherent && is_volatile) break;

    if (element_inst->opcode() == spv::Op::OpTypePointer) {
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(1u));
    } else if (element_inst->opcode() == spv::Op::OpTypeStruct) {
      uint32_t index = indices.at(i);
      Instruction* index_inst = context()->get_def_use_mgr()->GetDef(index);
      assert(index_inst->opcode() == spv::Op::OpConstant);
      uint64_t value = GetIndexValue(index_inst);
      is_coherent |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   spv::Decoration::Coherent);
      is_volatile |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   spv::Decoration::Volatile);
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(static_cast<uint32_t>(value)));
    } else {
      assert(spvOpcodeIsComposite(element_inst->opcode()));
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(0u));
    }
  }

  if (!is_coherent || !is_volatile) {
    bool remaining_coherent = false;
    bool remaining_volatile = false;
    std::tie(remaining_coherent, remaining_volatile) =
        CheckAllTypes(element_inst);
    is_coherent |= remaining_coherent;
    is_volatile |= remaining_volatile;
  }

  return std::make_pair(is_coherent, is_volatile);
}

std::pair<bool, bool> UpgradeMemoryModel::CheckAllTypes(
    const Instruction* inst) {
  std::unordered_set<const Instruction*> visited;
  std::vector<const Instruction*> stack;
  stack.push_back(inst);

  bool is_coherent = false;
  bool is_volatile = false;
  while (!stack.empty()) {
    const Instruction* def = stack.back();
    stack.pop_back();

    if (!visited.insert(def).second) continue;

    if (def->opcode() == spv::Op::OpTypeStruct) {
      // Any member decorated with coherent and/or volatile is enough to have
      // the related operation be flagged as coherent and/or volatile.
      is_coherent |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   spv::Decoration::Coherent);
      is_volatile |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   spv::Decoration::Volatile);
      if (is_coherent && is_volatile)
        return std::make_pair(is_coherent, is_volatile);

      // Check the subtypes.
      for (uint32_t i = 0; i < def->NumInOperands(); ++i) {
        stack.push_back(context()->get_def_use_mgr()->GetDef(
            def->GetSingleWordInOperand(i)));
      }
    } else if (spvOpcodeIsComposite(def->opcode())) {
      stack.push_back(context()->get_def_use_mgr()->GetDef(
          def->GetSingleWordInOperand(0u)));
    } else if (def->opcode() == spv::Op::OpTypePointer) {
      stack.push_back(context()->get_def_use_mgr()->GetDef(
          def->GetSingleWordInOperand(1u)));
    }
  }

  return std::make_pair(is_coherent, is_volatile);
}

uint64_t UpgradeMemoryModel::GetIndexValue(Instruction* index_inst) {
  const analysis::Constant* index_constant =
      context()->get_constant_mgr()->GetConstantFromInst(index_inst);
  assert(index_constant->AsIntConstant());
  if (index_constant->type()->AsInteger()->IsSigned()) {
    if (index_constant->type()->AsInteger()->width() == 32) {
      return index_constant->GetS32();
    } else {
      return index_constant->GetS64();
    }
  } else {
    if (index_constant->type()->AsInteger()->width() == 32) {
      return index_constant->GetU32();
    } else {
      return index_constant->GetU64();
    }
  }
}

bool UpgradeMemoryModel::HasDecoration(const Instruction* inst, uint32_t value,
                                       spv::Decoration decoration) {
  // If the iteration was terminated early then an appropriate decoration was
  // found.
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      inst->result_id(), (uint32_t)decoration, [value](const Instruction& i) {
        if (i.opcode() == spv::Op::OpDecorate ||
            i.opcode() == spv::Op::OpDecorateId) {
          return false;
        } else if (i.opcode() == spv::Op::OpMemberDecorate) {
          if (value == i.GetSingleWordInOperand(1u) ||
              value == std::numeric_limits<uint32_t>::max())
            return false;
        }

        return true;
      });
}

void UpgradeMemoryModel::UpgradeFlags(Instruction* inst, uint32_t in_operand,
                                      bool is_coherent, bool is_volatile,
                                      OperationType operation_type,
                                      InstructionType inst_type) {
  if (!is_coherent && !is_volatile) return;

  uint32_t flags = 0;
  if (inst->NumInOperands() > in_operand) {
    flags |= inst->GetSingleWordInOperand(in_operand);
  }
  if (is_coherent) {
    if (inst_type == kMemory) {
      flags |= uint32_t(spv::MemoryAccessMask::NonPrivatePointerKHR);
      if (operation_type == kVisibility) {
        flags |= uint32_t(spv::MemoryAccessMask::MakePointerVisibleKHR);
      } else {
        flags |= uint32_t(spv::MemoryAccessMask::MakePointerAvailableKHR);
      }
    } else {
      flags |= uint32_t(spv::ImageOperandsMask::NonPrivateTexelKHR);
      if (operation_type == kVisibility) {
        flags |= uint32_t(spv::ImageOperandsMask::MakeTexelVisibleKHR);
      } else {
        flags |= uint32_t(spv::ImageOperandsMask::MakeTexelAvailableKHR);
      }
    }
  }

  if (is_volatile) {
    if (inst_type == kMemory) {
      flags |= uint32_t(spv::MemoryAccessMask::Volatile);
    } else {
      flags |= uint32_t(spv::ImageOperandsMask::VolatileTexelKHR);
    }
  }

  if (inst->NumInOperands() > in_operand) {
    inst->SetInOperand(in_operand, {flags});
  } else if (inst_type == kMemory) {
    inst->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS, {flags}});
  } else {
    inst->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_IMAGE, {flags}});
  }
}

uint32_t UpgradeMemoryModel::GetScopeConstant(spv::Scope scope) {
  analysis::Integer int_ty(32, false);
  uint32_t int_id = context()->get_type_mgr()->GetTypeInstruction(&int_ty);
  const analysis::Constant* constant =
      context()->get_constant_mgr()->GetConstant(
          context()->get_type_mgr()->GetType(int_id),
          {static_cast<uint32_t>(scope)});
  return context()
      ->get_constant_mgr()
      ->GetDefiningInstruction(constant)
      ->result_id();
}

void UpgradeMemoryModel::CleanupDecorations() {
  // All of the volatile and coherent decorations have been dealt with, so now
  // we can just remove them.
  get_module()->ForEachInst([this](Instruction* inst) {
    if (inst->result_id() != 0) {
      context()->get_decoration_mgr()->RemoveDecorationsFrom(
          inst->result_id(), [](const Instruction& dec) {
            switch (dec.opcode()) {
              case spv::Op::OpDecorate:
              case spv::Op::OpDecorateId:
                if (spv::Decoration(dec.GetSingleWordInOperand(1u)) ==
                        spv::Decoration::Coherent ||
                    spv::Decoration(dec.GetSingleWordInOperand(1u)) ==
                        spv::Decoration::Volatile)
                  return true;
                break;
              case spv::Op::OpMemberDecorate:
                if (spv::Decoration(dec.GetSingleWordInOperand(2u)) ==
                        spv::Decoration::Coherent ||
                    spv::Decoration(dec.GetSingleWordInOperand(2u)) ==
                        spv::Decoration::Volatile)
                  return true;
                break;
              default:
                break;
            }
            return false;
          });
    }
  });
}

void UpgradeMemoryModel::UpgradeBarriers() {
  std::vector<Instruction*> barriers;
  // Collects all the control barriers in |function|. Returns true if the
  // function operates on the Output storage class.
  ProcessFunction CollectBarriers = [this, &barriers](Function* function) {
    bool operates_on_output = false;
    for (auto& block : *function) {
      block.ForEachInst([this, &barriers,
                         &operates_on_output](Instruction* inst) {
        if (inst->opcode() == spv::Op::OpControlBarrier) {
          barriers.push_back(inst);
        } else if (!operates_on_output) {
          // This instruction operates on output storage class if it is a
          // pointer to output type or any input operand is a pointer to output
          // type.
          analysis::Type* type =
              context()->get_type_mgr()->GetType(inst->type_id());
          if (type && type->AsPointer() &&
              type->AsPointer()->storage_class() == spv::StorageClass::Output) {
            operates_on_output = true;
            return;
          }
          inst->ForEachInId([this, &operates_on_output](uint32_t* id_ptr) {
            Instruction* op_inst =
                context()->get_def_use_mgr()->GetDef(*id_ptr);
            analysis::Type* op_type =
                context()->get_type_mgr()->GetType(op_inst->type_id());
            if (op_type && op_type->AsPointer() &&
                op_type->AsPointer()->storage_class() ==
                    spv::StorageClass::Output)
              operates_on_output = true;
          });
        }
      });
    }
    return operates_on_output;
  };

  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points())
    if (spv::ExecutionModel(e.GetSingleWordInOperand(0u)) ==
        spv::ExecutionModel::TessellationControl) {
      roots.push(e.GetSingleWordInOperand(1u));
      if (context()->ProcessCallTreeFromRoots(CollectBarriers, &roots)) {
        for (auto barrier : barriers) {
          // Add OutputMemoryKHR to the semantics of the barriers.
          uint32_t semantics_id = barrier->GetSingleWordInOperand(2u);
          Instruction* semantics_inst =
              context()->get_def_use_mgr()->GetDef(semantics_id);
          analysis::Type* semantics_type =
              context()->get_type_mgr()->GetType(semantics_inst->type_id());
          uint64_t semantics_value = GetIndexValue(semantics_inst);
          const analysis::Constant* constant =
              context()->get_constant_mgr()->GetConstant(
                  semantics_type,
                  {static_cast<uint32_t>(semantics_value) |
                   uint32_t(spv::MemorySemanticsMask::OutputMemoryKHR)});
          barrier->SetInOperand(2u, {context()
                                         ->get_constant_mgr()
                                         ->GetDefiningInstruction(constant)
                                         ->result_id()});
        }
      }
      barriers.clear();
    }
}

void UpgradeMemoryModel::UpgradeMemoryScope() {
  get_module()->ForEachInst([this](Instruction* inst) {
    // Don't need to handle all the operations that take a scope.
    // * Group operations can only be subgroup
    // * Non-uniform can only be workgroup or subgroup
    // * Named barriers are not supported by Vulkan
    // * Workgroup ops (e.g. async_copy) have at most workgroup scope.
    if (spvOpcodeIsAtomicOp(inst->opcode())) {
      if (IsDeviceScope(inst->GetSingleWordInOperand(1))) {
        inst->SetInOperand(1, {GetScopeConstant(spv::Scope::QueueFamilyKHR)});
      }
    } else if (inst->opcode() == spv::Op::OpControlBarrier) {
      if (IsDeviceScope(inst->GetSingleWordInOperand(1))) {
        inst->SetInOperand(1, {GetScopeConstant(spv::Scope::QueueFamilyKHR)});
      }
    } else if (inst->opcode() == spv::Op::OpMemoryBarrier) {
      if (IsDeviceScope(inst->GetSingleWordInOperand(0))) {
        inst->SetInOperand(0, {GetScopeConstant(spv::Scope::QueueFamilyKHR)});
      }
    }
  });
}

bool UpgradeMemoryModel::IsDeviceScope(uint32_t scope_id) {
  const analysis::Constant* constant =
      context()->get_constant_mgr()->FindDeclaredConstant(scope_id);
  assert(constant && "Memory scope must be a constant");

  const analysis::Integer* type = constant->type()->AsInteger();
  assert(type);
  assert(type->width() == 32 || type->width() == 64);
  if (type->width() == 32) {
    if (type->IsSigned())
      return static_cast<spv::Scope>(constant->GetS32()) == spv::Scope::Device;
    else
      return static_cast<spv::Scope>(constant->GetU32()) == spv::Scope::Device;
  } else {
    if (type->IsSigned())
      return static_cast<spv::Scope>(constant->GetS64()) == spv::Scope::Device;
    else
      return static_cast<spv::Scope>(constant->GetU64()) == spv::Scope::Device;
  }

  assert(false);
  return false;
}

void UpgradeMemoryModel::UpgradeExtInst(Instruction* ext_inst) {
  const bool is_modf = ext_inst->GetSingleWordInOperand(1u) == GLSLstd450Modf;
  auto ptr_id = ext_inst->GetSingleWordInOperand(3u);
  auto ptr_type_id = get_def_use_mgr()->GetDef(ptr_id)->type_id();
  auto pointee_type_id =
      get_def_use_mgr()->GetDef(ptr_type_id)->GetSingleWordInOperand(1u);
  auto element_type_id = ext_inst->type_id();
  std::vector<const analysis::Type*> element_types(2);
  element_types[0] = context()->get_type_mgr()->GetType(element_type_id);
  element_types[1] = context()->get_type_mgr()->GetType(pointee_type_id);
  analysis::Struct struct_type(element_types);
  uint32_t struct_id =
      context()->get_type_mgr()->GetTypeInstruction(&struct_type);
  // Change the operation
  GLSLstd450 new_op = is_modf ? GLSLstd450ModfStruct : GLSLstd450FrexpStruct;
  ext_inst->SetOperand(3u, {static_cast<uint32_t>(new_op)});
  // Remove the pointer argument
  ext_inst->RemoveOperand(5u);
  // Set the type id to the new struct.
  ext_inst->SetResultType(struct_id);

  // The result is now a struct of the original result. The zero'th element is
  // old result and should replace the old result. The one'th element needs to
  // be stored via a new instruction.
  auto where = ext_inst->NextNode();
  InstructionBuilder builder(
      context(), where,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  auto extract_0 =
      builder.AddCompositeExtract(element_type_id, ext_inst->result_id(), {0});
  context()->ReplaceAllUsesWith(ext_inst->result_id(), extract_0->result_id());
  // The extract's input was just changed to itself, so fix that.
  extract_0->SetInOperand(0u, {ext_inst->result_id()});
  auto extract_1 =
      builder.AddCompositeExtract(pointee_type_id, ext_inst->result_id(), {1});
  builder.AddStore(ptr_id, extract_1->result_id());
}

uint32_t UpgradeMemoryModel::MemoryAccessNumWords(uint32_t mask) {
  uint32_t result = 1;
  if (mask & uint32_t(spv::MemoryAccessMask::Aligned)) ++result;
  if (mask & uint32_t(spv::MemoryAccessMask::MakePointerAvailableKHR)) ++result;
  if (mask & uint32_t(spv::MemoryAccessMask::MakePointerVisibleKHR)) ++result;
  return result;
}

}  // namespace opt
}  // namespace spvtools
