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

#include "source/disassemble.h"
#include "source/opt/fold.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

namespace {
// Indices used to get particular operands out of instructions using InOperand.
const uint32_t kTypeImageDimIndex = 1;
const uint32_t kLoadBaseIndex = 0;
const uint32_t kVariableStorageClassIndex = 0;
const uint32_t kTypeImageSampledIndex = 5;
}  // namespace

Instruction::Instruction(IRContext* c)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(SpvOpNop),
      has_type_id_(false),
      has_result_id_(false),
      unique_id_(c->TakeNextUniqueId()) {}

Instruction::Instruction(IRContext* c, SpvOp op)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      has_type_id_(false),
      has_result_id_(false),
      unique_id_(c->TakeNextUniqueId()) {}

Instruction::Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
                         std::vector<Instruction>&& dbg_line)
    : context_(c),
      opcode_(static_cast<SpvOp>(inst.opcode)),
      has_type_id_(inst.type_id != 0),
      has_result_id_(inst.result_id != 0),
      unique_id_(c->TakeNextUniqueId()),
      dbg_line_insts_(std::move(dbg_line)) {
  assert((!IsDebugLineInst(opcode_) || dbg_line.empty()) &&
         "Op(No)Line attaching to Op(No)Line found");
  for (uint32_t i = 0; i < inst.num_operands; ++i) {
    const auto& current_payload = inst.operands[i];
    std::vector<uint32_t> words(
        inst.words + current_payload.offset,
        inst.words + current_payload.offset + current_payload.num_words);
    operands_.emplace_back(current_payload.type, std::move(words));
  }
}

Instruction::Instruction(IRContext* c, SpvOp op, uint32_t ty_id,
                         uint32_t res_id, const OperandList& in_operands)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      has_type_id_(ty_id != 0),
      has_result_id_(res_id != 0),
      unique_id_(c->TakeNextUniqueId()),
      operands_() {
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
      opcode_(that.opcode_),
      has_type_id_(that.has_type_id_),
      has_result_id_(that.has_result_id_),
      unique_id_(that.unique_id_),
      operands_(std::move(that.operands_)),
      dbg_line_insts_(std::move(that.dbg_line_insts_)) {}

Instruction& Instruction::operator=(Instruction&& that) {
  opcode_ = that.opcode_;
  has_type_id_ = that.has_type_id_;
  has_result_id_ = that.has_result_id_;
  unique_id_ = that.unique_id_;
  operands_ = std::move(that.operands_);
  dbg_line_insts_ = std::move(that.dbg_line_insts_);
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

void Instruction::ToBinaryWithoutAttachedDebugInsts(
    std::vector<uint32_t>* binary) const {
  const uint32_t num_words = 1 + NumOperandWords();
  binary->push_back((num_words << 16) | static_cast<uint16_t>(opcode_));
  for (const auto& operand : operands_)
    binary->insert(binary->end(), operand.words.begin(), operand.words.end());
}

void Instruction::ReplaceOperands(const OperandList& new_operands) {
  operands_.clear();
  operands_.insert(operands_.begin(), new_operands.begin(), new_operands.end());
}

bool Instruction::IsReadOnlyLoad() const {
  if (IsLoad()) {
    Instruction* address_def = GetBaseAddress();
    if (!address_def || address_def->opcode() != SpvOpVariable) {
      return false;
    }
    return address_def->IsReadOnlyVariable();
  }
  return false;
}

Instruction* Instruction::GetBaseAddress() const {
  uint32_t base = GetSingleWordInOperand(kLoadBaseIndex);
  Instruction* base_inst = context()->get_def_use_mgr()->GetDef(base);
  bool done = false;
  while (!done) {
    switch (base_inst->opcode()) {
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpInBoundsPtrAccessChain:
      case SpvOpImageTexelPointer:
      case SpvOpCopyObject:
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

bool Instruction::IsReadOnlyVariable() const {
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return IsReadOnlyVariableShaders();
  else
    return IsReadOnlyVariableKernel();
}

bool Instruction::IsVulkanStorageImage() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) == SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage image.
  auto s = base_type->GetSingleWordInOperand(kTypeImageSampledIndex);
  return s != 1;
}

bool Instruction::IsVulkanSampledImage() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) == SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we know for sure that it is,
  // then return true.
  auto s = base_type->GetSingleWordInOperand(kTypeImageSampledIndex);
  return s == 1;
}

bool Instruction::IsVulkanStorageTexelBuffer() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) != SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage texel buffer.
  return base_type->GetSingleWordInOperand(kTypeImageSampledIndex) != 1;
}

bool Instruction::IsVulkanStorageBuffer() const {
  // Is there a difference between a "Storage buffer" and a "dynamic storage
  // buffer" in SPIR-V and do we care about the difference?
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  if (base_type->opcode() != SpvOpTypeStruct) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class == SpvStorageClassUniform) {
    bool is_buffer_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), SpvDecorationBufferBlock,
        [&is_buffer_block](const Instruction&) { is_buffer_block = true; });
    return is_buffer_block;
  } else if (storage_class == SpvStorageClassStorageBuffer) {
    bool is_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), SpvDecorationBlock,
        [&is_block](const Instruction&) { is_block = true; });
    return is_block;
  }
  return false;
}

bool Instruction::IsVulkanUniformBuffer() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniform) {
    return false;
  }

  Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeStruct) {
    return false;
  }

  bool is_block = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      base_type->result_id(), SpvDecorationBlock,
      [&is_block](const Instruction&) { is_block = true; });
  return is_block;
}

bool Instruction::IsReadOnlyVariableShaders() const {
  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  Instruction* type_def = context()->get_def_use_mgr()->GetDef(type_id());

  switch (storage_class) {
    case SpvStorageClassUniformConstant:
      if (!type_def->IsVulkanStorageImage() &&
          !type_def->IsVulkanStorageTexelBuffer()) {
        return true;
      }
      break;
    case SpvStorageClassUniform:
      if (!type_def->IsVulkanStorageBuffer()) {
        return true;
      }
      break;
    case SpvStorageClassPushConstant:
    case SpvStorageClassInput:
      return true;
    default:
      break;
  }

  bool is_nonwritable = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      result_id(), SpvDecorationNonWritable,
      [&is_nonwritable](const Instruction&) { is_nonwritable = true; });
  return is_nonwritable;
}

bool Instruction::IsReadOnlyVariableKernel() const {
  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  return storage_class == SpvStorageClassUniformConstant;
}

uint32_t Instruction::GetTypeComponent(uint32_t element) const {
  uint32_t subtype = 0;
  switch (opcode()) {
    case SpvOpTypeStruct:
      subtype = GetSingleWordInOperand(element);
      break;
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
      // These types all have uniform subtypes.
      subtype = GetSingleWordInOperand(0u);
      break;
    default:
      break;
  }

  return subtype;
}

Instruction* Instruction::InsertBefore(
    std::vector<std::unique_ptr<Instruction>>&& list) {
  Instruction* first_node = list.front().get();
  for (auto& i : list) {
    i.release()->InsertBefore(this);
  }
  list.clear();
  return first_node;
}

Instruction* Instruction::InsertBefore(std::unique_ptr<Instruction>&& i) {
  i.get()->InsertBefore(this);
  return i.release();
}

bool Instruction::IsValidBasePointer() const {
  uint32_t tid = type_id();
  if (tid == 0) {
    return false;
  }

  Instruction* type = context()->get_def_use_mgr()->GetDef(tid);
  if (type->opcode() != SpvOpTypePointer) {
    return false;
  }

  auto feature_mgr = context()->get_feature_mgr();
  if (feature_mgr->HasCapability(SpvCapabilityAddresses)) {
    // TODO: The rules here could be more restrictive.
    return true;
  }

  if (opcode() == SpvOpVariable || opcode() == SpvOpFunctionParameter) {
    return true;
  }

  // With variable pointers, there are more valid base pointer objects.
  // Variable pointers implicitly declares Variable pointers storage buffer.
  SpvStorageClass storage_class =
      static_cast<SpvStorageClass>(type->GetSingleWordInOperand(0));
  if ((feature_mgr->HasCapability(SpvCapabilityVariablePointersStorageBuffer) &&
       storage_class == SpvStorageClassStorageBuffer) ||
      (feature_mgr->HasCapability(SpvCapabilityVariablePointers) &&
       storage_class == SpvStorageClassWorkgroup)) {
    switch (opcode()) {
      case SpvOpPhi:
      case SpvOpSelect:
      case SpvOpFunctionCall:
      case SpvOpConstantNull:
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

bool Instruction::IsValidBaseImage() const {
  uint32_t tid = type_id();
  if (tid == 0) {
    return false;
  }

  Instruction* type = context()->get_def_use_mgr()->GetDef(tid);
  return (type->opcode() == SpvOpTypeImage ||
          type->opcode() == SpvOpTypeSampledImage);
}

bool Instruction::IsOpaqueType() const {
  if (opcode() == SpvOpTypeStruct) {
    bool is_opaque = false;
    ForEachInOperand([&is_opaque, this](const uint32_t* op_id) {
      Instruction* type_inst = context()->get_def_use_mgr()->GetDef(*op_id);
      is_opaque |= type_inst->IsOpaqueType();
    });
    return is_opaque;
  } else if (opcode() == SpvOpTypeArray) {
    uint32_t sub_type_id = GetSingleWordInOperand(0);
    Instruction* sub_type_inst =
        context()->get_def_use_mgr()->GetDef(sub_type_id);
    return sub_type_inst->IsOpaqueType();
  } else {
    return opcode() == SpvOpTypeRuntimeArray ||
           spvOpcodeIsBaseOpaqueType(opcode());
  }
}

bool Instruction::IsFoldable() const {
  return IsFoldableByFoldScalar() ||
         context()->get_instruction_folder().HasConstFoldingRule(opcode());
}

bool Instruction::IsFoldableByFoldScalar() const {
  const InstructionFolder& folder = context()->get_instruction_folder();
  if (!folder.IsFoldableOpcode(opcode())) {
    return false;
  }
  Instruction* type = context()->get_def_use_mgr()->GetDef(type_id());
  return folder.IsFoldableType(type);
}

bool Instruction::IsFloatingPointFoldingAllowed() const {
  // TODO: Add the rules for kernels.  For now it will be pessimistic.
  if (!context_->get_feature_mgr()->HasCapability(SpvCapabilityShader)) {
    return false;
  }

  bool is_nocontract = false;
  context_->get_decoration_mgr()->WhileEachDecoration(
      result_id(), SpvDecorationNoContraction,
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
    case SpvOpNop:
    case SpvOpUndef:
    case SpvOpLoad:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpArrayLength:
    case SpvOpVectorExtractDynamic:
    case SpvOpVectorInsertDynamic:
    case SpvOpVectorShuffle:
    case SpvOpCompositeConstruct:
    case SpvOpCompositeExtract:
    case SpvOpCompositeInsert:
    case SpvOpCopyObject:
    case SpvOpTranspose:
    case SpvOpConvertFToU:
    case SpvOpConvertFToS:
    case SpvOpConvertSToF:
    case SpvOpConvertUToF:
    case SpvOpUConvert:
    case SpvOpSConvert:
    case SpvOpFConvert:
    case SpvOpQuantizeToF16:
    case SpvOpBitcast:
    case SpvOpSNegate:
    case SpvOpFNegate:
    case SpvOpIAdd:
    case SpvOpFAdd:
    case SpvOpISub:
    case SpvOpFSub:
    case SpvOpIMul:
    case SpvOpFMul:
    case SpvOpUDiv:
    case SpvOpSDiv:
    case SpvOpFDiv:
    case SpvOpUMod:
    case SpvOpSRem:
    case SpvOpSMod:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpVectorTimesScalar:
    case SpvOpMatrixTimesScalar:
    case SpvOpVectorTimesMatrix:
    case SpvOpMatrixTimesVector:
    case SpvOpMatrixTimesMatrix:
    case SpvOpOuterProduct:
    case SpvOpDot:
    case SpvOpIAddCarry:
    case SpvOpISubBorrow:
    case SpvOpUMulExtended:
    case SpvOpSMulExtended:
    case SpvOpAny:
    case SpvOpAll:
    case SpvOpIsNan:
    case SpvOpIsInf:
    case SpvOpLogicalEqual:
    case SpvOpLogicalNotEqual:
    case SpvOpLogicalOr:
    case SpvOpLogicalAnd:
    case SpvOpLogicalNot:
    case SpvOpSelect:
    case SpvOpIEqual:
    case SpvOpINotEqual:
    case SpvOpUGreaterThan:
    case SpvOpSGreaterThan:
    case SpvOpUGreaterThanEqual:
    case SpvOpSGreaterThanEqual:
    case SpvOpULessThan:
    case SpvOpSLessThan:
    case SpvOpULessThanEqual:
    case SpvOpSLessThanEqual:
    case SpvOpFOrdEqual:
    case SpvOpFUnordEqual:
    case SpvOpFOrdNotEqual:
    case SpvOpFUnordNotEqual:
    case SpvOpFOrdLessThan:
    case SpvOpFUnordLessThan:
    case SpvOpFOrdGreaterThan:
    case SpvOpFUnordGreaterThan:
    case SpvOpFOrdLessThanEqual:
    case SpvOpFUnordLessThanEqual:
    case SpvOpFOrdGreaterThanEqual:
    case SpvOpFUnordGreaterThanEqual:
    case SpvOpShiftRightLogical:
    case SpvOpShiftRightArithmetic:
    case SpvOpShiftLeftLogical:
    case SpvOpBitwiseOr:
    case SpvOpBitwiseXor:
    case SpvOpBitwiseAnd:
    case SpvOpNot:
    case SpvOpBitFieldInsert:
    case SpvOpBitFieldSExtract:
    case SpvOpBitFieldUExtract:
    case SpvOpBitReverse:
    case SpvOpBitCount:
    case SpvOpSizeOf:
      return true;
    default:
      return false;
  }
}

bool Instruction::IsScalarizable() const {
  if (spvOpcodeIsScalarizable(opcode())) {
    return true;
  }

  const uint32_t kExtInstSetIdInIdx = 0;
  const uint32_t kExtInstInstructionInIdx = 1;

  if (opcode() == SpvOpExtInst) {
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
    case SpvOpDPdx:
    case SpvOpDPdy:
    case SpvOpFwidth:
    case SpvOpDPdxFine:
    case SpvOpDPdyFine:
    case SpvOpFwidthFine:
    case SpvOpDPdxCoarse:
    case SpvOpDPdyCoarse:
    case SpvOpFwidthCoarse:
    case SpvOpImageQueryLod:
      return true;
    default:
      return false;
  }
}

}  // namespace opt
}  // namespace spvtools
