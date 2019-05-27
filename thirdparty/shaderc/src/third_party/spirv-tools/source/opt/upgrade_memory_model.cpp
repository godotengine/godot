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
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status UpgradeMemoryModel::Process() {
  // Only update Logical GLSL450 to Logical VulkanKHR.
  Instruction* memory_model = get_module()->GetMemoryModel();
  if (memory_model->GetSingleWordInOperand(0u) != SpvAddressingModelLogical ||
      memory_model->GetSingleWordInOperand(1u) != SpvMemoryModelGLSL450) {
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
  get_module()->AddCapability(MakeUnique<Instruction>(
      context(), SpvOpCapability, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_CAPABILITY, {SpvCapabilityVulkanMemoryModelKHR}}}));
  const std::string extension = "SPV_KHR_vulkan_memory_model";
  std::vector<uint32_t> words(extension.size() / 4 + 1, 0);
  char* dst = reinterpret_cast<char*>(words.data());
  strncpy(dst, extension.c_str(), extension.size());
  get_module()->AddExtension(
      MakeUnique<Instruction>(context(), SpvOpExtension, 0, 0,
                              std::initializer_list<Operand>{
                                  {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
  memory_model->SetInOperand(1u, {SpvMemoryModelVulkanKHR});
}

void UpgradeMemoryModel::UpgradeInstructions() {
  // Coherent and Volatile decorations are deprecated. Remove them and replace
  // with flags on the memory/image operations. The decorations can occur on
  // OpVariable, OpFunctionParameter (of pointer type) and OpStructType (member
  // decoration). Trace from the decoration target(s) to the final memory/image
  // instructions. Additionally, Workgroup storage class variables and function
  // parameters are implicitly coherent in GLSL450.

  // Upgrade modf and frexp first since they generate new stores.
  for (auto& func : *get_module()) {
    func.ForEachInst([this](Instruction* inst) {
      if (inst->opcode() == SpvOpExtInst) {
        auto ext_inst = inst->GetSingleWordInOperand(1u);
        if (ext_inst == GLSLstd450Modf || ext_inst == GLSLstd450Frexp) {
          auto import =
              get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
          if (reinterpret_cast<char*>(import->GetInOperand(0u).words.data()) ==
              std::string("GLSL.std.450")) {
            UpgradeExtInst(inst);
          }
        }
      }
    });
  }
  for (auto& func : *get_module()) {
    func.ForEachInst([this](Instruction* inst) {
      bool is_coherent = false;
      bool is_volatile = false;
      bool src_coherent = false;
      bool src_volatile = false;
      bool dst_coherent = false;
      bool dst_volatile = false;
      SpvScope scope = SpvScopeQueueFamilyKHR;
      SpvScope src_scope = SpvScopeQueueFamilyKHR;
      SpvScope dst_scope = SpvScopeQueueFamilyKHR;
      switch (inst->opcode()) {
        case SpvOpLoad:
        case SpvOpStore:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
        case SpvOpImageWrite:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized:
          std::tie(dst_coherent, dst_volatile, dst_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          std::tie(src_coherent, src_volatile, src_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(1u));
          break;
        default:
          break;
      }

      switch (inst->opcode()) {
        case SpvOpLoad:
          UpgradeFlags(inst, 1u, is_coherent, is_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpStore:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kAvailability,
                       kMemory);
          break;
        case SpvOpCopyMemory:
          UpgradeFlags(inst, 2u, dst_coherent, dst_volatile, kAvailability,
                       kMemory);
          UpgradeFlags(inst, 2u, src_coherent, src_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpCopyMemorySized:
          UpgradeFlags(inst, 3u, dst_coherent, dst_volatile, kAvailability,
                       kMemory);
          UpgradeFlags(inst, 3u, src_coherent, src_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kVisibility, kImage);
          break;
        case SpvOpImageWrite:
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
    });
  }
}

std::tuple<bool, bool, SpvScope> UpgradeMemoryModel::GetInstructionAttributes(
    uint32_t id) {
  // |id| is a pointer used in a memory/image instruction. Need to determine if
  // that pointer points to volatile or coherent memory. Workgroup storage
  // class is implicitly coherent and cannot be decorated with volatile, so
  // short circuit that case.
  Instruction* inst = context()->get_def_use_mgr()->GetDef(id);
  analysis::Type* type = context()->get_type_mgr()->GetType(inst->type_id());
  if (type->AsPointer() &&
      type->AsPointer()->storage_class() == SpvStorageClassWorkgroup) {
    return std::make_tuple(true, false, SpvScopeWorkgroup);
  }

  bool is_coherent = false;
  bool is_volatile = false;
  std::unordered_set<uint32_t> visited;
  std::tie(is_coherent, is_volatile) =
      TraceInstruction(context()->get_def_use_mgr()->GetDef(id),
                       std::vector<uint32_t>(), &visited);

  return std::make_tuple(is_coherent, is_volatile, SpvScopeQueueFamilyKHR);
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
    case SpvOpVariable:
    case SpvOpFunctionParameter:
      is_coherent |= HasDecoration(inst, 0, SpvDecorationCoherent);
      is_volatile |= HasDecoration(inst, 0, SpvDecorationVolatile);
      if (!is_coherent || !is_volatile) {
        bool type_coherent = false;
        bool type_volatile = false;
        std::tie(type_coherent, type_volatile) =
            CheckType(inst->type_id(), indices);
        is_coherent |= type_coherent;
        is_volatile |= type_volatile;
      }
      break;
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
      // Store indices in reverse order.
      for (uint32_t i = inst->NumInOperands() - 1; i > 0; --i) {
        indices.push_back(inst->GetSingleWordInOperand(i));
      }
      break;
    case SpvOpPtrAccessChain:
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
  if (inst->opcode() != SpvOpVariable &&
      inst->opcode() != SpvOpFunctionParameter) {
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
  assert(type_inst->opcode() == SpvOpTypePointer);
  Instruction* element_inst = context()->get_def_use_mgr()->GetDef(
      type_inst->GetSingleWordInOperand(1u));
  for (int i = (int)indices.size() - 1; i >= 0; --i) {
    if (is_coherent && is_volatile) break;

    if (element_inst->opcode() == SpvOpTypePointer) {
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(1u));
    } else if (element_inst->opcode() == SpvOpTypeStruct) {
      uint32_t index = indices.at(i);
      Instruction* index_inst = context()->get_def_use_mgr()->GetDef(index);
      assert(index_inst->opcode() == SpvOpConstant);
      uint64_t value = GetIndexValue(index_inst);
      is_coherent |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   SpvDecorationCoherent);
      is_volatile |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   SpvDecorationVolatile);
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(static_cast<uint32_t>(value)));
    } else {
      assert(spvOpcodeIsComposite(element_inst->opcode()));
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(1u));
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

    if (def->opcode() == SpvOpTypeStruct) {
      // Any member decorated with coherent and/or volatile is enough to have
      // the related operation be flagged as coherent and/or volatile.
      is_coherent |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   SpvDecorationCoherent);
      is_volatile |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   SpvDecorationVolatile);
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
    } else if (def->opcode() == SpvOpTypePointer) {
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
                                       SpvDecoration decoration) {
  // If the iteration was terminated early then an appropriate decoration was
  // found.
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      inst->result_id(), decoration, [value](const Instruction& i) {
        if (i.opcode() == SpvOpDecorate || i.opcode() == SpvOpDecorateId) {
          return false;
        } else if (i.opcode() == SpvOpMemberDecorate) {
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
      flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
      if (operation_type == kVisibility) {
        flags |= SpvMemoryAccessMakePointerVisibleKHRMask;
      } else {
        flags |= SpvMemoryAccessMakePointerAvailableKHRMask;
      }
    } else {
      flags |= SpvImageOperandsNonPrivateTexelKHRMask;
      if (operation_type == kVisibility) {
        flags |= SpvImageOperandsMakeTexelVisibleKHRMask;
      } else {
        flags |= SpvImageOperandsMakeTexelAvailableKHRMask;
      }
    }
  }

  if (is_volatile) {
    if (inst_type == kMemory) {
      flags |= SpvMemoryAccessVolatileMask;
    } else {
      flags |= SpvImageOperandsVolatileTexelKHRMask;
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

uint32_t UpgradeMemoryModel::GetScopeConstant(SpvScope scope) {
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
              case SpvOpDecorate:
              case SpvOpDecorateId:
                if (dec.GetSingleWordInOperand(1u) == SpvDecorationCoherent ||
                    dec.GetSingleWordInOperand(1u) == SpvDecorationVolatile)
                  return true;
                break;
              case SpvOpMemberDecorate:
                if (dec.GetSingleWordInOperand(2u) == SpvDecorationCoherent ||
                    dec.GetSingleWordInOperand(2u) == SpvDecorationVolatile)
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
        if (inst->opcode() == SpvOpControlBarrier) {
          barriers.push_back(inst);
        } else if (!operates_on_output) {
          // This instruction operates on output storage class if it is a
          // pointer to output type or any input operand is a pointer to output
          // type.
          analysis::Type* type =
              context()->get_type_mgr()->GetType(inst->type_id());
          if (type && type->AsPointer() &&
              type->AsPointer()->storage_class() == SpvStorageClassOutput) {
            operates_on_output = true;
            return;
          }
          inst->ForEachInId([this, &operates_on_output](uint32_t* id_ptr) {
            Instruction* op_inst =
                context()->get_def_use_mgr()->GetDef(*id_ptr);
            analysis::Type* op_type =
                context()->get_type_mgr()->GetType(op_inst->type_id());
            if (op_type && op_type->AsPointer() &&
                op_type->AsPointer()->storage_class() == SpvStorageClassOutput)
              operates_on_output = true;
          });
        }
      });
    }
    return operates_on_output;
  };

  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points())
    if (e.GetSingleWordInOperand(0u) == SpvExecutionModelTessellationControl) {
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
                  semantics_type, {static_cast<uint32_t>(semantics_value) |
                                   SpvMemorySemanticsOutputMemoryKHRMask});
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
        inst->SetInOperand(1, {GetScopeConstant(SpvScopeQueueFamilyKHR)});
      }
    } else if (inst->opcode() == SpvOpControlBarrier) {
      if (IsDeviceScope(inst->GetSingleWordInOperand(1))) {
        inst->SetInOperand(1, {GetScopeConstant(SpvScopeQueueFamilyKHR)});
      }
    } else if (inst->opcode() == SpvOpMemoryBarrier) {
      if (IsDeviceScope(inst->GetSingleWordInOperand(0))) {
        inst->SetInOperand(0, {GetScopeConstant(SpvScopeQueueFamilyKHR)});
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
      return static_cast<uint32_t>(constant->GetS32()) == SpvScopeDevice;
    else
      return static_cast<uint32_t>(constant->GetU32()) == SpvScopeDevice;
  } else {
    if (type->IsSigned())
      return static_cast<uint32_t>(constant->GetS64()) == SpvScopeDevice;
    else
      return static_cast<uint32_t>(constant->GetU64()) == SpvScopeDevice;
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

}  // namespace opt
}  // namespace spvtools
