// Copyright (c) 2016 Google Inc.
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

#include "source/opt/instruction.h"

#include <initializer_list>

#include "OpenCLDebugInfo100.h"
#include "source/disassemble.h"
#include "source/opt/fold.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {
namespace {
// Indices used to get particular operands out of instructions using InOperand.
constexpr uint32_t kTypeImageDimIndex = 1;
constexpr uint32_t kLoadBaseIndex = 0;
constexpr uint32_t kPointerTypeStorageClassIndex = 0;
constexpr uint32_t kVariableStorageClassIndex = 0;
constexpr uint32_t kTypeImageSampledIndex = 5;

// Constants for OpenCL.DebugInfo.100 / NonSemantic.Shader.DebugInfo.100
// extension instructions.
constexpr uint32_t kExtInstSetIdInIdx = 0;
constexpr uint32_t kExtInstInstructionInIdx = 1;
constexpr uint32_t kDebugScopeNumWords = 7;
constexpr uint32_t kDebugScopeNumWordsWithoutInlinedAt = 6;
constexpr uint32_t kDebugNoScopeNumWords = 5;

// Number of operands of an OpBranchConditional instruction
// with weights.
constexpr uint32_t kOpBranchConditionalWithWeightsNumOperands = 5;
}  // namespace

Instruction::Instruction(IRContext* c)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(spv::Op::OpNop),
      has_type_id_(false),
      has_result_id_(false),
      unique_id_(c->TakeNextUniqueId()),
      dbg_scope_(kNoDebugScope, kNoInlinedAt) {}

Instruction::Instruction(IRContext* c, spv::Op op)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      has_type_id_(false),
      has_result_id_(false),
      unique_id_(c->TakeNextUniqueId()),
      dbg_scope_(kNoDebugScope, kNoInlinedAt) {}

Instruction::Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
                         std::vector<Instruction>&& dbg_line)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(static_cast<spv::Op>(inst.opcode)),
      has_type_id_(inst.type_id != 0),
      has_result_id_(inst.result_id != 0),
      unique_id_(c->TakeNextUniqueId()),
      dbg_line_insts_(std::move(dbg_line)),
      dbg_scope_(kNoDebugScope, kNoInlinedAt) {
  operands_.reserve(inst.num_operands);
  for (uint32_t i = 0; i < inst.num_operands; ++i) {
    const auto& current_payload = inst.operands[i];
    operands_.emplace_back(
        current_payload.type, inst.words + current_payload.offset,
        inst.words + current_payload.offset + current_payload.num_words);
  }
  assert((!IsLineInst() || dbg_line.empty()) &&
         "Op(No)Line attaching to Op(No)Line found");
}

Instruction::Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
                         const DebugScope& dbg_scope)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(static_cast<spv::Op>(inst.opcode)),
      has_type_id_(inst.type_id != 0),
      has_result_id_(inst.result_id != 0),
      unique_id_(c->TakeNextUniqueId()),
      dbg_scope_(dbg_scope) {
  operands_.reserve(inst.num_operands);
  for (uint32_t i = 0; i < inst.num_operands; ++i) {
    const auto& current_payload = inst.operands[i];
    operands_.emplace_back(
        current_payload.type, inst.words + current_payload.offset,
        inst.words + current_payload.offset + current_payload.num_words);
  }
}

Instruction::Instruction(IRContext* c, spv::Op op, uint32_t ty_id,
                         uint32_t res_id, const OperandList& in_operands)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      has_type_id_(ty_id != 0),
      has_result_id_(res_id != 0),
      unique_id_(c->TakeNextUniqueId()),
      operands_(),
      dbg_scope_(kNoDebugScope, kNoInlinedAt) {
  size_t operands_size = in_operands.size();
  if (has_type_id_) {
    operands_size++;
  }
  if (has_result_id_) {
    operands_size++;
  }
  operands_.reserve(operands_size);
  if (has_type_id_) {
    operands_.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_TYPE_ID,
                           std::initializer_list<uint32_t>{ty_id});
  }
  if (has_result_id_) {
    operands_.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
                           std::initializer_list<uint32_t>{res_id});
  }
  operands_.insert(operands_.end(), in_operands.begin(), in_operands.end());
}

Instruction::Instruction(Instruction&& that)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(that.context_),
      opcode_(that.opcode_),
      has_type_id_(that.has_type_id_),
      has_result_id_(that.has_result_id_),
      unique_id_(that.unique_id_),
      operands_(std::move(that.operands_)),
      dbg_line_insts_(std::move(that.dbg_line_insts_)),
      dbg_scope_(that.dbg_scope_) {
  for (auto& i : dbg_line_insts_) {
    i.dbg_scope_ = that.dbg_scope_;
  }
}

Instruction& Instruction::operator=(Instruction&& that) {
  context_ = that.context_;
  opcode_ = that.opcode_;
  has_type_id_ = that.has_type_id_;
  has_result_id_ = that.has_result_id_;
  unique_id_ = that.unique_id_;
  operands_ = std::move(that.operands_);
  dbg_line_insts_ = std::move(that.dbg_line_insts_);
  dbg_scope_ = that.dbg_scope_;
  return *this;
}

Instruction* Instruction::Clone(IRContext* c) const {
  Instruction* clone = new Instruction(c);
  clone->opcode_ = opcode_;
  clone->has_type_id_ = has_type_id_;
  clone->has_result_id_ = has_result_id_;
  clone->unique_id_ = c->TakeNextUniqueId();
  clone->operands_ = operands_;
  clone->dbg_line_insts_ = dbg_line_insts_;
  for (auto& i : clone->dbg_line_insts_) {
    i.unique_id_ = c->TakeNextUniqueId();
    if (i.IsDebugLineInst()) i.SetResultId(c->TakeNextId());
  }
  clone->dbg_scope_ = dbg_scope_;
  return clone;
}

uint32_t Instruction::GetSingleWordOperand(uint32_t index) const {
  const auto& words = GetOperand(index).words;
  assert(words.size() == 1 && "expected the operand only taking one word");
  return words.front();
}

uint32_t Instruction::NumInOperandWords() const {
  uint32_t size = 0;
  for (uint32_t i = TypeResultIdCount(); i < operands_.size(); ++i)
    size += static_cast<uint32_t>(operands_[i].words.size());
  return size;
}

bool Instruction::HasBranchWeights() const {
  if (opcode_ == spv::Op::OpBranchConditional &&
      NumOperands() == kOpBranchConditionalWithWeightsNumOperands) {
    return true;
  }

  return false;
}

void Instruction::ToBinaryWithoutAttachedDebugInsts(
    std::vector<uint32_t>* binary) const {
  const uint32_t num_words = 1 + NumOperandWords();
  binary->push_back((num_words << 16) | static_cast<uint16_t>(opcode_));
  for (const auto& operand : operands_) {
    binary->insert(binary->end(), operand.words.begin(), operand.words.end());
  }
}

void Instruction::ReplaceOperands(const OperandList& new_operands) {
  operands_.clear();
  operands_.insert(operands_.begin(), new_operands.begin(), new_operands.end());
}

bool Instruction::IsReadOnlyLoad() const {
  if (IsLoad()) {
    Instruction* address_def = GetBaseAddress();
    if (!address_def) {
      return false;
    }

    if (address_def->opcode() == spv::Op::OpVariable) {
      if (address_def->IsReadOnlyPointer()) {
        return true;
      }
    }

    if (address_def->opcode() == spv::Op::OpLoad) {
      const analysis::Type* address_type =
          context()->get_type_mgr()->GetType(address_def->type_id());
      if (address_type->AsSampledImage() != nullptr) {
        const auto* image_type =
            address_type->AsSampledImage()->image_type()->AsImage();
        if (image_type->sampled() == 1) {
          return true;
        }
      }
    }
  }
  return false;
}

Instruction* Instruction::GetBaseAddress() const {
  uint32_t base = GetSingleWordInOperand(kLoadBaseIndex);
  Instruction* base_inst = context()->get_def_use_mgr()->GetDef(base);
  bool done = false;
  while (!done) {
    switch (base_inst->opcode()) {
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpPtrAccessChain:
      case spv::Op::OpInBoundsPtrAccessChain:
      case spv::Op::OpImageTexelPointer:
      case spv::Op::OpCopyObject:
        // All of these instructions have the base pointer use a base pointer
        // in in-operand 0.
        base = base_inst->GetSingleWordInOperand(0);
        base_inst = context()->get_def_use_mgr()->GetDef(base);
        break;
      default:
        done = true;
        break;
    }
  }
  return base_inst;
}

bool Instruction::IsReadOnlyPointer() const {
  if (context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return IsReadOnlyPointerShaders();
  else
    return IsReadOnlyPointerKernel();
}

bool Instruction::IsVulkanStorageImage() const {
  if (opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kPointerTypeStorageClassIndex));
  if (storage_class != spv::StorageClass::UniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  // Unpack the optional layer of arraying.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = context()->get_def_use_mgr()->GetDef(
        base_type->GetSingleWordInOperand(0));
  }

  if (base_type->opcode() != spv::Op::OpTypeImage) {
    return false;
  }

  if (spv::Dim(base_type->GetSingleWordInOperand(kTypeImageDimIndex)) ==
      spv::Dim::Buffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage image.
  return base_type->GetSingleWordInOperand(kTypeImageSampledIndex) != 1;
}

bool Instruction::IsVulkanSampledImage() const {
  if (opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kPointerTypeStorageClassIndex));
  if (storage_class != spv::StorageClass::UniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  // Unpack the optional layer of arraying.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = context()->get_def_use_mgr()->GetDef(
        base_type->GetSingleWordInOperand(0));
  }

  if (base_type->opcode() != spv::Op::OpTypeImage) {
    return false;
  }

  if (spv::Dim(base_type->GetSingleWordInOperand(kTypeImageDimIndex)) ==
      spv::Dim::Buffer) {
    return false;
  }

  // Check if the image is sampled.  If we know for sure that it is,
  // then return true.
  return base_type->GetSingleWordInOperand(kTypeImageSampledIndex) == 1;
}

bool Instruction::IsVulkanStorageTexelBuffer() const {
  if (opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kPointerTypeStorageClassIndex));
  if (storage_class != spv::StorageClass::UniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  // Unpack the optional layer of arraying.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = context()->get_def_use_mgr()->GetDef(
        base_type->GetSingleWordInOperand(0));
  }

  if (base_type->opcode() != spv::Op::OpTypeImage) {
    return false;
  }

  if (spv::Dim(base_type->GetSingleWordInOperand(kTypeImageDimIndex)) !=
      spv::Dim::Buffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage texel buffer.
  return base_type->GetSingleWordInOperand(kTypeImageSampledIndex) != 1;
}

bool Instruction::IsVulkanStorageBuffer() const {
  // Is there a difference between a "Storage buffer" and a "dynamic storage
  // buffer" in SPIR-V and do we care about the difference?
  if (opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  // Unpack the optional layer of arraying.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = context()->get_def_use_mgr()->GetDef(
        base_type->GetSingleWordInOperand(0));
  }

  if (base_type->opcode() != spv::Op::OpTypeStruct) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kPointerTypeStorageClassIndex));
  if (storage_class == spv::StorageClass::Uniform) {
    bool is_buffer_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), uint32_t(spv::Decoration::BufferBlock),
        [&is_buffer_block](const Instruction&) { is_buffer_block = true; });
    return is_buffer_block;
  } else if (storage_class == spv::StorageClass::StorageBuffer) {
    bool is_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), uint32_t(spv::Decoration::Block),
        [&is_block](const Instruction&) { is_block = true; });
    return is_block;
  }
  return false;
}

bool Instruction::IsVulkanStorageBufferVariable() const {
  if (opcode() != spv::Op::OpVariable) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kVariableStorageClassIndex));
  if (storage_class == spv::StorageClass::StorageBuffer ||
      storage_class == spv::StorageClass::Uniform) {
    Instruction* var_type = context()->get_def_use_mgr()->GetDef(type_id());
    return var_type != nullptr && var_type->IsVulkanStorageBuffer();
  }

  return false;
}

bool Instruction::IsVulkanUniformBuffer() const {
  if (opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class =
      spv::StorageClass(GetSingleWordInOperand(kPointerTypeStorageClassIndex));
  if (storage_class != spv::StorageClass::Uniform) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  // Unpack the optional layer of arraying.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = context()->get_def_use_mgr()->GetDef(
        base_type->GetSingleWordInOperand(0));
  }

  if (base_type->opcode() != spv::Op::OpTypeStruct) {
    return false;
  }

  bool is_block = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      base_type->result_id(), uint32_t(spv::Decoration::Block),
      [&is_block](const Instruction&) { is_block = true; });
  return is_block;
}

bool Instruction::IsReadOnlyPointerShaders() const {
  if (type_id() == 0) {
    return false;
  }

  Instruction* type_def = context()->get_def_use_mgr()->GetDef(type_id());
  if (type_def->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class = spv::StorageClass(
      type_def->GetSingleWordInOperand(kPointerTypeStorageClassIndex));

  switch (storage_class) {
    case spv::StorageClass::UniformConstant:
      if (!type_def->IsVulkanStorageImage() &&
          !type_def->IsVulkanStorageTexelBuffer()) {
        return true;
      }
      break;
    case spv::StorageClass::Uniform:
      if (!type_def->IsVulkanStorageBuffer()) {
        return true;
      }
      break;
    case spv::StorageClass::PushConstant:
    case spv::StorageClass::Input:
      return true;
    default:
      break;
  }

  bool is_nonwritable = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      result_id(), uint32_t(spv::Decoration::NonWritable),
      [&is_nonwritable](const Instruction&) { is_nonwritable = true; });
  return is_nonwritable;
}

bool Instruction::IsReadOnlyPointerKernel() const {
  if (type_id() == 0) {
    return false;
  }

  Instruction* type_def = context()->get_def_use_mgr()->GetDef(type_id());
  if (type_def->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  spv::StorageClass storage_class = spv::StorageClass(
      type_def->GetSingleWordInOperand(kPointerTypeStorageClassIndex));

  return storage_class == spv::StorageClass::UniformConstant;
}

void Instruction::UpdateLexicalScope(uint32_t scope) {
  dbg_scope_.SetLexicalScope(scope);
  for (auto& i : dbg_line_insts_) {
    i.dbg_scope_.SetLexicalScope(scope);
  }
  if (!IsLineInst() &&
      context()->AreAnalysesValid(IRContext::kAnalysisDebugInfo)) {
    context()->get_debug_info_mgr()->AnalyzeDebugInst(this);
  }
}

void Instruction::UpdateDebugInlinedAt(uint32_t new_inlined_at) {
  dbg_scope_.SetInlinedAt(new_inlined_at);
  for (auto& i : dbg_line_insts_) {
    i.dbg_scope_.SetInlinedAt(new_inlined_at);
  }
  if (!IsLineInst() &&
      context()->AreAnalysesValid(IRContext::kAnalysisDebugInfo)) {
    context()->get_debug_info_mgr()->AnalyzeDebugInst(this);
  }
}

void Instruction::ClearDbgLineInsts() {
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    auto def_use_mgr = context()->get_def_use_mgr();
    for (auto& l_inst : dbg_line_insts_) def_use_mgr->ClearInst(&l_inst);
  }
  clear_dbg_line_insts();
}

void Instruction::UpdateDebugInfoFrom(const Instruction* from) {
  if (from == nullptr) return;
  ClearDbgLineInsts();
  if (!from->dbg_line_insts().empty())
    AddDebugLine(&from->dbg_line_insts().back());
  SetDebugScope(from->GetDebugScope());
  if (!IsLineInst() &&
      context()->AreAnalysesValid(IRContext::kAnalysisDebugInfo)) {
    context()->get_debug_info_mgr()->AnalyzeDebugInst(this);
  }
}

void Instruction::AddDebugLine(const Instruction* inst) {
  dbg_line_insts_.push_back(*inst);
  dbg_line_insts_.back().unique_id_ = context()->TakeNextUniqueId();
  if (inst->IsDebugLineInst())
    dbg_line_insts_.back().SetResultId(context_->TakeNextId());
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(&dbg_line_insts_.back());
}

bool Instruction::IsDebugLineInst() const {
  NonSemanticShaderDebugInfo100Instructions ext_opt = GetShader100DebugOpcode();
  return ((ext_opt == NonSemanticShaderDebugInfo100DebugLine) ||
          (ext_opt == NonSemanticShaderDebugInfo100DebugNoLine));
}

bool Instruction::IsLineInst() const { return IsLine() || IsNoLine(); }

bool Instruction::IsLine() const {
  if (opcode() == spv::Op::OpLine) return true;
  NonSemanticShaderDebugInfo100Instructions ext_opt = GetShader100DebugOpcode();
  return ext_opt == NonSemanticShaderDebugInfo100DebugLine;
}

bool Instruction::IsNoLine() const {
  if (opcode() == spv::Op::OpNoLine) return true;
  NonSemanticShaderDebugInfo100Instructions ext_opt = GetShader100DebugOpcode();
  return ext_opt == NonSemanticShaderDebugInfo100DebugNoLine;
}

Instruction* Instruction::InsertBefore(std::unique_ptr<Instruction>&& inst) {
  inst.get()->InsertBefore(this);
  return inst.release();
}

Instruction* Instruction::InsertBefore(
    std::vector<std::unique_ptr<Instruction>>&& list) {
  Instruction* first_node = list.front().get();
  for (auto& inst : list) {
    inst.release()->InsertBefore(this);
  }
  list.clear();
  return first_node;
}

bool Instruction::IsValidBasePointer() const {
  uint32_t tid = type_id();
  if (tid == 0) {
    return false;
  }

  Instruction* type = context()->get_def_use_mgr()->GetDef(tid);
  if (type->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  auto feature_mgr = context()->get_feature_mgr();
  if (feature_mgr->HasCapability(spv::Capability::Addresses)) {
    // TODO: The rules here could be more restrictive.
    return true;
  }

  if (opcode() == spv::Op::OpVariable ||
      opcode() == spv::Op::OpFunctionParameter) {
    return true;
  }

  // With variable pointers, there are more valid base pointer objects.
  // Variable pointers implicitly declares Variable pointers storage buffer.
  spv::StorageClass storage_class =
      static_cast<spv::StorageClass>(type->GetSingleWordInOperand(0));
  if ((feature_mgr->HasCapability(
           spv::Capability::VariablePointersStorageBuffer) &&
       storage_class == spv::StorageClass::StorageBuffer) ||
      (feature_mgr->HasCapability(spv::Capability::VariablePointers) &&
       storage_class == spv::StorageClass::Workgroup)) {
    switch (opcode()) {
      case spv::Op::OpPhi:
      case spv::Op::OpSelect:
      case spv::Op::OpFunctionCall:
      case spv::Op::OpConstantNull:
        return true;
      default:
        break;
    }
  }

  uint32_t pointee_type_id = type->GetSingleWordInOperand(1);
  Instruction* pointee_type_inst =
      context()->get_def_use_mgr()->GetDef(pointee_type_id);

  if (pointee_type_inst->IsOpaqueType()) {
    return true;
  }
  return false;
}

OpenCLDebugInfo100Instructions Instruction::GetOpenCL100DebugOpcode() const {
  if (opcode() != spv::Op::OpExtInst) {
    return OpenCLDebugInfo100InstructionsMax;
  }

  if (!context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo()) {
    return OpenCLDebugInfo100InstructionsMax;
  }

  if (GetSingleWordInOperand(kExtInstSetIdInIdx) !=
      context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo()) {
    return OpenCLDebugInfo100InstructionsMax;
  }

  return OpenCLDebugInfo100Instructions(
      GetSingleWordInOperand(kExtInstInstructionInIdx));
}

NonSemanticShaderDebugInfo100Instructions Instruction::GetShader100DebugOpcode()
    const {
  if (opcode() != spv::Op::OpExtInst) {
    return NonSemanticShaderDebugInfo100InstructionsMax;
  }

  if (!context()->get_feature_mgr()->GetExtInstImportId_Shader100DebugInfo()) {
    return NonSemanticShaderDebugInfo100InstructionsMax;
  }

  if (GetSingleWordInOperand(kExtInstSetIdInIdx) !=
      context()->get_feature_mgr()->GetExtInstImportId_Shader100DebugInfo()) {
    return NonSemanticShaderDebugInfo100InstructionsMax;
  }

  uint32_t opcode = GetSingleWordInOperand(kExtInstInstructionInIdx);
  if (opcode >= NonSemanticShaderDebugInfo100InstructionsMax) {
    return NonSemanticShaderDebugInfo100InstructionsMax;
  }

  return NonSemanticShaderDebugInfo100Instructions(opcode);
}

CommonDebugInfoInstructions Instruction::GetCommonDebugOpcode() const {
  if (opcode() != spv::Op::OpExtInst) {
    return CommonDebugInfoInstructionsMax;
  }

  const uint32_t opencl_set_id =
      context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo();
  const uint32_t shader_set_id =
      context()->get_feature_mgr()->GetExtInstImportId_Shader100DebugInfo();

  if (!opencl_set_id && !shader_set_id) {
    return CommonDebugInfoInstructionsMax;
  }

  const uint32_t used_set_id = GetSingleWordInOperand(kExtInstSetIdInIdx);

  if (used_set_id != opencl_set_id && used_set_id != shader_set_id) {
    return CommonDebugInfoInstructionsMax;
  }

  return CommonDebugInfoInstructions(
      GetSingleWordInOperand(kExtInstInstructionInIdx));
}

bool Instruction::IsValidBaseImage() const {
  uint32_t tid = type_id();
  if (tid == 0) {
    return false;
  }

  Instruction* type = context()->get_def_use_mgr()->GetDef(tid);
  return (type->opcode() == spv::Op::OpTypeImage ||
          type->opcode() == spv::Op::OpTypeSampledImage);
}

bool Instruction::IsOpaqueType() const {
  if (opcode() == spv::Op::OpTypeStruct) {
    bool is_opaque = false;
    ForEachInOperand([&is_opaque, this](const uint32_t* op_id) {
      Instruction* type_inst = context()->get_def_use_mgr()->GetDef(*op_id);
      is_opaque |= type_inst->IsOpaqueType();
    });
    return is_opaque;
  } else if (opcode() == spv::Op::OpTypeArray) {
    uint32_t sub_type_id = GetSingleWordInOperand(0);
    Instruction* sub_type_inst =
        context()->get_def_use_mgr()->GetDef(sub_type_id);
    return sub_type_inst->IsOpaqueType();
  } else {
    return opcode() == spv::Op::OpTypeRuntimeArray ||
           spvOpcodeIsBaseOpaqueType(opcode());
  }
}

bool Instruction::IsFoldable() const {
  return IsFoldableByFoldScalar() ||
         context()->get_instruction_folder().HasConstFoldingRule(this);
}

bool Instruction::IsFoldableByFoldScalar() const {
  const InstructionFolder& folder = context()->get_instruction_folder();
  if (!folder.IsFoldableOpcode(opcode())) {
    return false;
  }

  Instruction* type = context()->get_def_use_mgr()->GetDef(type_id());
  if (!folder.IsFoldableType(type)) {
    return false;
  }

  // Even if the type of the instruction is foldable, its operands may not be
  // foldable (e.g., comparisons of 64bit types).  Check that all operand types
  // are foldable before accepting the instruction.
  return WhileEachInOperand([&folder, this](const uint32_t* op_id) {
    Instruction* def_inst = context()->get_def_use_mgr()->GetDef(*op_id);
    Instruction* def_inst_type =
        context()->get_def_use_mgr()->GetDef(def_inst->type_id());
    return folder.IsFoldableType(def_inst_type);
  });
}

bool Instruction::IsFloatingPointFoldingAllowed() const {
  // TODO: Add the rules for kernels.  For now it will be pessimistic.
  // For now, do not support capabilities introduced by SPV_KHR_float_controls.
  if (!context_->get_feature_mgr()->HasCapability(spv::Capability::Shader) ||
      context_->get_feature_mgr()->HasCapability(
          spv::Capability::DenormPreserve) ||
      context_->get_feature_mgr()->HasCapability(
          spv::Capability::DenormFlushToZero) ||
      context_->get_feature_mgr()->HasCapability(
          spv::Capability::SignedZeroInfNanPreserve) ||
      context_->get_feature_mgr()->HasCapability(
          spv::Capability::RoundingModeRTZ) ||
      context_->get_feature_mgr()->HasCapability(
          spv::Capability::RoundingModeRTE)) {
    return false;
  }

  bool is_nocontract = false;
  context_->get_decoration_mgr()->WhileEachDecoration(
      result_id(), uint32_t(spv::Decoration::NoContraction),
      [&is_nocontract](const Instruction&) {
        is_nocontract = true;
        return false;
      });
  return !is_nocontract;
}

std::string Instruction::PrettyPrint(uint32_t options) const {
  // Convert the module to binary.
  std::vector<uint32_t> module_binary;
  context()->module()->ToBinary(&module_binary, /* skip_nop = */ false);

  // Convert the instruction to binary. This is used to identify the correct
  // stream of words to output from the module.
  std::vector<uint32_t> inst_binary;
  ToBinaryWithoutAttachedDebugInsts(&inst_binary);

  // Do not generate a header.
  return spvInstructionBinaryToText(
      context()->grammar().target_env(), inst_binary.data(), inst_binary.size(),
      module_binary.data(), module_binary.size(),
      options | SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
}

std::ostream& operator<<(std::ostream& str, const Instruction& inst) {
  str << inst.PrettyPrint();
  return str;
}

void Instruction::Dump() const {
  std::cerr << "Instruction #" << unique_id() << "\n" << *this << "\n";
}

bool Instruction::IsOpcodeCodeMotionSafe() const {
  switch (opcode_) {
    case spv::Op::OpNop:
    case spv::Op::OpUndef:
    case spv::Op::OpLoad:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpArrayLength:
    case spv::Op::OpVectorExtractDynamic:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpVectorShuffle:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpCopyObject:
    case spv::Op::OpTranspose:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpQuantizeToF16:
    case spv::Op::OpBitcast:
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpUMod:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpFRem:
    case spv::Op::OpFMod:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpMatrixTimesScalar:
    case spv::Op::OpVectorTimesMatrix:
    case spv::Op::OpMatrixTimesVector:
    case spv::Op::OpMatrixTimesMatrix:
    case spv::Op::OpOuterProduct:
    case spv::Op::OpDot:
    case spv::Op::OpIAddCarry:
    case spv::Op::OpISubBorrow:
    case spv::Op::OpUMulExtended:
    case spv::Op::OpSMulExtended:
    case spv::Op::OpAny:
    case spv::Op::OpAll:
    case spv::Op::OpIsNan:
    case spv::Op::OpIsInf:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot:
    case spv::Op::OpSelect:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFUnordNotEqual:
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFUnordLessThan:
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFUnordGreaterThan:
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFUnordLessThanEqual:
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
    case spv::Op::OpBitFieldInsert:
    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
    case spv::Op::OpSizeOf:
      return true;
    default:
      return false;
  }
}

bool Instruction::IsScalarizable() const {
  if (spvOpcodeIsScalarizable(opcode())) {
    return true;
  }

  if (opcode() == spv::Op::OpExtInst) {
    uint32_t instSetId =
        context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

    if (GetSingleWordInOperand(kExtInstSetIdInIdx) == instSetId) {
      switch (GetSingleWordInOperand(kExtInstInstructionInIdx)) {
        case GLSLstd450Round:
        case GLSLstd450RoundEven:
        case GLSLstd450Trunc:
        case GLSLstd450FAbs:
        case GLSLstd450SAbs:
        case GLSLstd450FSign:
        case GLSLstd450SSign:
        case GLSLstd450Floor:
        case GLSLstd450Ceil:
        case GLSLstd450Fract:
        case GLSLstd450Radians:
        case GLSLstd450Degrees:
        case GLSLstd450Sin:
        case GLSLstd450Cos:
        case GLSLstd450Tan:
        case GLSLstd450Asin:
        case GLSLstd450Acos:
        case GLSLstd450Atan:
        case GLSLstd450Sinh:
        case GLSLstd450Cosh:
        case GLSLstd450Tanh:
        case GLSLstd450Asinh:
        case GLSLstd450Acosh:
        case GLSLstd450Atanh:
        case GLSLstd450Atan2:
        case GLSLstd450Pow:
        case GLSLstd450Exp:
        case GLSLstd450Log:
        case GLSLstd450Exp2:
        case GLSLstd450Log2:
        case GLSLstd450Sqrt:
        case GLSLstd450InverseSqrt:
        case GLSLstd450Modf:
        case GLSLstd450FMin:
        case GLSLstd450UMin:
        case GLSLstd450SMin:
        case GLSLstd450FMax:
        case GLSLstd450UMax:
        case GLSLstd450SMax:
        case GLSLstd450FClamp:
        case GLSLstd450UClamp:
        case GLSLstd450SClamp:
        case GLSLstd450FMix:
        case GLSLstd450Step:
        case GLSLstd450SmoothStep:
        case GLSLstd450Fma:
        case GLSLstd450Frexp:
        case GLSLstd450Ldexp:
        case GLSLstd450FindILsb:
        case GLSLstd450FindSMsb:
        case GLSLstd450FindUMsb:
        case GLSLstd450NMin:
        case GLSLstd450NMax:
        case GLSLstd450NClamp:
          return true;
        default:
          return false;
      }
    }
  }
  return false;
}

bool Instruction::IsOpcodeSafeToDelete() const {
  if (context()->IsCombinatorInstruction(this)) {
    return true;
  }

  switch (opcode()) {
    case spv::Op::OpDPdx:
    case spv::Op::OpDPdy:
    case spv::Op::OpFwidth:
    case spv::Op::OpDPdxFine:
    case spv::Op::OpDPdyFine:
    case spv::Op::OpFwidthFine:
    case spv::Op::OpDPdxCoarse:
    case spv::Op::OpDPdyCoarse:
    case spv::Op::OpFwidthCoarse:
    case spv::Op::OpImageQueryLod:
      return true;
    default:
      return false;
  }
}

bool Instruction::IsNonSemanticInstruction() const {
  if (!HasResultId()) return false;
  if (opcode() != spv::Op::OpExtInst) return false;

  auto import_inst =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(0));
  std::string import_name = import_inst->GetInOperand(0).AsString();
  return import_name.find("NonSemantic.") == 0;
}

void DebugScope::ToBinary(uint32_t type_id, uint32_t result_id,
                          uint32_t ext_set,
                          std::vector<uint32_t>* binary) const {
  uint32_t num_words = kDebugScopeNumWords;
  CommonDebugInfoInstructions dbg_opcode = CommonDebugInfoDebugScope;
  if (GetLexicalScope() == kNoDebugScope) {
    num_words = kDebugNoScopeNumWords;
    dbg_opcode = CommonDebugInfoDebugNoScope;
  } else if (GetInlinedAt() == kNoInlinedAt) {
    num_words = kDebugScopeNumWordsWithoutInlinedAt;
  }
  std::vector<uint32_t> operands = {
      (num_words << 16) | static_cast<uint16_t>(spv::Op::OpExtInst),
      type_id,
      result_id,
      ext_set,
      static_cast<uint32_t>(dbg_opcode),
  };
  binary->insert(binary->end(), operands.begin(), operands.end());
  if (GetLexicalScope() != kNoDebugScope) {
    binary->push_back(GetLexicalScope());
    if (GetInlinedAt() != kNoInlinedAt) binary->push_back(GetInlinedAt());
  }
}

}  // namespace opt
}  // namespace spvtools
