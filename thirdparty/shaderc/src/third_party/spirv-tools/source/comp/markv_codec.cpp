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

// MARK-V is a compression format for SPIR-V binaries. It strips away
// non-essential information (such as result IDs which can be regenerated) and
// uses various bit reduction techniques to reduce the size of the binary.

#include "source/comp/markv_codec.h"

#include "source/comp/markv_logger.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_opencl_std_header.h"
#include "source/opcode.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace comp {
namespace {

// Custom hash function used to produce short descriptors.
uint32_t ShortHashU32Array(const std::vector<uint32_t>& words) {
  // The hash function is a sum of hashes of each word seeded by word index.
  // Knuth's multiplicative hash is used to hash the words.
  const uint32_t kKnuthMulHash = 2654435761;
  uint32_t val = 0;
  for (uint32_t i = 0; i < words.size(); ++i) {
    val += (words[i] + i + 123) * kKnuthMulHash;
  }
  return 1 + val % ((1 << MarkvCodec::kShortDescriptorNumBits) - 1);
}

// Returns a set of mtf rank codecs based on a plausible hand-coded
// distribution.
std::map<uint64_t, std::unique_ptr<HuffmanCodec<uint32_t>>>
GetMtfHuffmanCodecs() {
  std::map<uint64_t, std::unique_ptr<HuffmanCodec<uint32_t>>> codecs;

  std::unique_ptr<HuffmanCodec<uint32_t>> codec;

  codec = MakeUnique<HuffmanCodec<uint32_t>>(std::map<uint32_t, uint32_t>({
      {0, 5},
      {1, 40},
      {2, 10},
      {3, 5},
      {4, 5},
      {5, 5},
      {6, 3},
      {7, 3},
      {8, 3},
      {9, 3},
      {MarkvCodec::kMtfRankEncodedByValueSignal, 10},
  }));
  codecs.emplace(kMtfAll, std::move(codec));

  codec = MakeUnique<HuffmanCodec<uint32_t>>(std::map<uint32_t, uint32_t>({
      {1, 50},
      {2, 20},
      {3, 5},
      {4, 5},
      {5, 2},
      {6, 1},
      {7, 1},
      {8, 1},
      {9, 1},
      {MarkvCodec::kMtfRankEncodedByValueSignal, 10},
  }));
  codecs.emplace(kMtfGenericNonZeroRank, std::move(codec));

  return codecs;
}

}  // namespace

const uint32_t MarkvCodec::kMarkvMagicNumber = 0x07230303;

const uint32_t MarkvCodec::kMtfSmallestRankEncodedByValue = 10;

const uint32_t MarkvCodec::kMtfRankEncodedByValueSignal =
    std::numeric_limits<uint32_t>::max();

const uint32_t MarkvCodec::kShortDescriptorNumBits = 8;

const size_t MarkvCodec::kByteBreakAfterInstIfLessThanUntilNextByte = 8;

MarkvCodec::MarkvCodec(spv_const_context context,
                       spv_validator_options validator_options,
                       const MarkvModel* model)
    : validator_options_(validator_options),
      grammar_(context),
      model_(model),
      short_id_descriptors_(ShortHashU32Array),
      mtf_huffman_codecs_(GetMtfHuffmanCodecs()),
      context_(context) {}

MarkvCodec::~MarkvCodec() { spvValidatorOptionsDestroy(validator_options_); }

MarkvCodec::MarkvHeader::MarkvHeader()
    : magic_number(MarkvCodec::kMarkvMagicNumber),
      markv_version(MarkvCodec::GetMarkvVersion()) {}

// Defines and returns current MARK-V version.
// static
uint32_t MarkvCodec::GetMarkvVersion() {
  const uint32_t kVersionMajor = 1;
  const uint32_t kVersionMinor = 4;
  return kVersionMinor | (kVersionMajor << 16);
}

size_t MarkvCodec::GetNumBitsToNextByte(size_t bit_pos) const {
  return (8 - (bit_pos % 8)) % 8;
}

// Returns true if the opcode has a fixed number of operands. May return a
// false negative.
bool MarkvCodec::OpcodeHasFixedNumberOfOperands(SpvOp opcode) const {
  switch (opcode) {
    // TODO(atgoo@github.com) This is not a complete list.
    case SpvOpNop:
    case SpvOpName:
    case SpvOpUndef:
    case SpvOpSizeOf:
    case SpvOpLine:
    case SpvOpNoLine:
    case SpvOpDecorationGroup:
    case SpvOpExtension:
    case SpvOpExtInstImport:
    case SpvOpMemoryModel:
    case SpvOpCapability:
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeArray:
    case SpvOpTypePointer:
    case SpvOpConstantTrue:
    case SpvOpConstantFalse:
    case SpvOpLabel:
    case SpvOpBranch:
    case SpvOpFunction:
    case SpvOpFunctionParameter:
    case SpvOpFunctionEnd:
    case SpvOpBitcast:
    case SpvOpCopyObject:
    case SpvOpTranspose:
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
      return true;
    default:
      break;
  }
  return false;
}

void MarkvCodec::ProcessCurInstruction() {
  instructions_.emplace_back(new val::Instruction(&inst_));

  const SpvOp opcode = SpvOp(inst_.opcode);

  if (inst_.result_id) {
    id_to_def_instruction_.emplace(inst_.result_id, instructions_.back().get());

    // Collect ids local to the current function.
    if (cur_function_id_) {
      ids_local_to_cur_function_.push_back(inst_.result_id);
    }

    // Starting new function.
    if (opcode == SpvOpFunction) {
      cur_function_id_ = inst_.result_id;
      cur_function_return_type_ = inst_.type_id;
      if (model_->id_fallback_strategy() ==
          MarkvModel::IdFallbackStrategy::kRuleBased) {
        multi_mtf_.Insert(GetMtfFunctionWithReturnType(inst_.type_id),
                          inst_.result_id);
      }

      // Store function parameter types in a queue, so that we know which types
      // to expect in the following OpFunctionParameter instructions.
      const val::Instruction* def_inst = FindDef(inst_.words[4]);
      assert(def_inst);
      assert(def_inst->opcode() == SpvOpTypeFunction);
      for (uint32_t i = 3; i < def_inst->words().size(); ++i) {
        remaining_function_parameter_types_.push_back(def_inst->word(i));
      }
    }
  }

  // Remove local ids from MTFs if function end.
  if (opcode == SpvOpFunctionEnd) {
    cur_function_id_ = 0;
    for (uint32_t id : ids_local_to_cur_function_) multi_mtf_.RemoveFromAll(id);
    ids_local_to_cur_function_.clear();
    assert(remaining_function_parameter_types_.empty());
  }

  if (!inst_.result_id) return;

  {
    // Save the result ID to type ID mapping.
    // In the grammar, type ID always appears before result ID.
    // A regular value maps to its type. Some instructions (e.g. OpLabel)
    // have no type Id, and will map to 0. The result Id for a
    // type-generating instruction (e.g. OpTypeInt) maps to itself.
    auto insertion_result = id_to_type_id_.emplace(
        inst_.result_id, spvOpcodeGeneratesType(SpvOp(inst_.opcode))
                             ? inst_.result_id
                             : inst_.type_id);
    (void)insertion_result;
    assert(insertion_result.second);
  }

  // Add result_id to MTFs.
  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kRuleBased) {
    switch (opcode) {
      case SpvOpTypeFloat:
      case SpvOpTypeInt:
      case SpvOpTypeBool:
      case SpvOpTypeVector:
      case SpvOpTypePointer:
      case SpvOpExtInstImport:
      case SpvOpTypeSampledImage:
      case SpvOpTypeImage:
      case SpvOpTypeSampler:
        multi_mtf_.Insert(GetMtfIdGeneratedByOpcode(opcode), inst_.result_id);
        break;
      default:
        break;
    }

    if (spvOpcodeIsComposite(opcode)) {
      multi_mtf_.Insert(kMtfTypeComposite, inst_.result_id);
    }

    if (opcode == SpvOpLabel) {
      multi_mtf_.InsertOrPromote(kMtfLabel, inst_.result_id);
    }

    if (opcode == SpvOpTypeInt) {
      multi_mtf_.Insert(kMtfTypeScalar, inst_.result_id);
      multi_mtf_.Insert(kMtfTypeIntScalarOrVector, inst_.result_id);
    }

    if (opcode == SpvOpTypeFloat) {
      multi_mtf_.Insert(kMtfTypeScalar, inst_.result_id);
      multi_mtf_.Insert(kMtfTypeFloatScalarOrVector, inst_.result_id);
    }

    if (opcode == SpvOpTypeBool) {
      multi_mtf_.Insert(kMtfTypeScalar, inst_.result_id);
      multi_mtf_.Insert(kMtfTypeBoolScalarOrVector, inst_.result_id);
    }

    if (opcode == SpvOpTypeVector) {
      const uint32_t component_type_id = inst_.words[2];
      const uint32_t size = inst_.words[3];
      if (multi_mtf_.HasValue(GetMtfIdGeneratedByOpcode(SpvOpTypeFloat),
                              component_type_id)) {
        multi_mtf_.Insert(kMtfTypeFloatScalarOrVector, inst_.result_id);
      } else if (multi_mtf_.HasValue(GetMtfIdGeneratedByOpcode(SpvOpTypeInt),
                                     component_type_id)) {
        multi_mtf_.Insert(kMtfTypeIntScalarOrVector, inst_.result_id);
      } else if (multi_mtf_.HasValue(GetMtfIdGeneratedByOpcode(SpvOpTypeBool),
                                     component_type_id)) {
        multi_mtf_.Insert(kMtfTypeBoolScalarOrVector, inst_.result_id);
      }
      multi_mtf_.Insert(GetMtfTypeVectorOfSize(size), inst_.result_id);
    }

    if (inst_.opcode == SpvOpTypeFunction) {
      const uint32_t return_type = inst_.words[2];
      multi_mtf_.Insert(kMtfTypeReturnedByFunction, return_type);
      multi_mtf_.Insert(GetMtfFunctionTypeWithReturnType(return_type),
                        inst_.result_id);
    }

    if (inst_.type_id) {
      const val::Instruction* type_inst = FindDef(inst_.type_id);
      assert(type_inst);

      multi_mtf_.Insert(kMtfObject, inst_.result_id);

      multi_mtf_.Insert(GetMtfIdOfType(inst_.type_id), inst_.result_id);

      if (multi_mtf_.HasValue(kMtfTypeFloatScalarOrVector, inst_.type_id)) {
        multi_mtf_.Insert(kMtfFloatScalarOrVector, inst_.result_id);
      }

      if (multi_mtf_.HasValue(kMtfTypeIntScalarOrVector, inst_.type_id))
        multi_mtf_.Insert(kMtfIntScalarOrVector, inst_.result_id);

      if (multi_mtf_.HasValue(kMtfTypeBoolScalarOrVector, inst_.type_id))
        multi_mtf_.Insert(kMtfBoolScalarOrVector, inst_.result_id);

      if (multi_mtf_.HasValue(kMtfTypeComposite, inst_.type_id))
        multi_mtf_.Insert(kMtfComposite, inst_.result_id);

      switch (type_inst->opcode()) {
        case SpvOpTypeInt:
        case SpvOpTypeBool:
        case SpvOpTypePointer:
        case SpvOpTypeVector:
        case SpvOpTypeImage:
        case SpvOpTypeSampledImage:
        case SpvOpTypeSampler:
          multi_mtf_.Insert(
              GetMtfIdWithTypeGeneratedByOpcode(type_inst->opcode()),
              inst_.result_id);
          break;
        default:
          break;
      }

      if (type_inst->opcode() == SpvOpTypeVector) {
        const uint32_t component_type = type_inst->word(2);
        multi_mtf_.Insert(GetMtfVectorOfComponentType(component_type),
                          inst_.result_id);
      }

      if (type_inst->opcode() == SpvOpTypePointer) {
        assert(type_inst->operands().size() > 2);
        assert(type_inst->words().size() > type_inst->operands()[2].offset);
        const uint32_t data_type =
            type_inst->word(type_inst->operands()[2].offset);
        multi_mtf_.Insert(GetMtfPointerToType(data_type), inst_.result_id);

        if (multi_mtf_.HasValue(kMtfTypeComposite, data_type))
          multi_mtf_.Insert(kMtfTypePointerToComposite, inst_.result_id);
      }
    }

    if (spvOpcodeGeneratesType(opcode)) {
      if (opcode != SpvOpTypeFunction) {
        multi_mtf_.Insert(kMtfTypeNonFunction, inst_.result_id);
      }
    }
  }

  if (model_->AnyDescriptorHasCodingScheme()) {
    const uint32_t long_descriptor =
        long_id_descriptors_.ProcessInstruction(inst_);
    if (model_->DescriptorHasCodingScheme(long_descriptor))
      multi_mtf_.Insert(GetMtfLongIdDescriptor(long_descriptor),
                        inst_.result_id);
  }

  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kShortDescriptor) {
    const uint32_t short_descriptor =
        short_id_descriptors_.ProcessInstruction(inst_);
    multi_mtf_.Insert(GetMtfShortIdDescriptor(short_descriptor),
                      inst_.result_id);
  }
}

uint64_t MarkvCodec::GetRuleBasedMtf() {
  // This function is only called for id operands (but not result ids).
  assert(spvIsIdType(operand_.type) ||
         operand_.type == SPV_OPERAND_TYPE_OPTIONAL_ID);
  assert(operand_.type != SPV_OPERAND_TYPE_RESULT_ID);

  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);

  // All operand slots which expect label id.
  if ((inst_.opcode == SpvOpLoopMerge && operand_index_ <= 1) ||
      (inst_.opcode == SpvOpSelectionMerge && operand_index_ == 0) ||
      (inst_.opcode == SpvOpBranch && operand_index_ == 0) ||
      (inst_.opcode == SpvOpBranchConditional &&
       (operand_index_ == 1 || operand_index_ == 2)) ||
      (inst_.opcode == SpvOpPhi && operand_index_ >= 3 &&
       operand_index_ % 2 == 1) ||
      (inst_.opcode == SpvOpSwitch && operand_index_ > 0)) {
    return kMtfLabel;
  }

  switch (opcode) {
    case SpvOpFAdd:
    case SpvOpFSub:
    case SpvOpFMul:
    case SpvOpFDiv:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpFNegate: {
      if (operand_index_ == 0) return kMtfTypeFloatScalarOrVector;
      return GetMtfIdOfType(inst_.type_id);
    }

    case SpvOpISub:
    case SpvOpIAdd:
    case SpvOpIMul:
    case SpvOpSDiv:
    case SpvOpUDiv:
    case SpvOpSMod:
    case SpvOpUMod:
    case SpvOpSRem:
    case SpvOpSNegate: {
      if (operand_index_ == 0) return kMtfTypeIntScalarOrVector;

      return kMtfIntScalarOrVector;
    }

      // TODO(atgoo@github.com) Add OpConvertFToU and other opcodes.

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
    case SpvOpFUnordGreaterThanEqual: {
      if (operand_index_ == 0) return kMtfTypeBoolScalarOrVector;
      if (operand_index_ == 2) return kMtfFloatScalarOrVector;
      if (operand_index_ == 3) {
        const uint32_t first_operand_id = GetInstWords()[3];
        const uint32_t first_operand_type = id_to_type_id_.at(first_operand_id);
        return GetMtfIdOfType(first_operand_type);
      }
      break;
    }

    case SpvOpVectorShuffle: {
      if (operand_index_ == 0) {
        assert(inst_.num_operands > 4);
        return GetMtfTypeVectorOfSize(inst_.num_operands - 4);
      }

      assert(inst_.type_id);
      if (operand_index_ == 2 || operand_index_ == 3)
        return GetMtfVectorOfComponentType(
            GetVectorComponentType(inst_.type_id));
      break;
    }

    case SpvOpVectorTimesScalar: {
      if (operand_index_ == 0) {
        // TODO(atgoo@github.com) Could be narrowed to vector of floats.
        return GetMtfIdGeneratedByOpcode(SpvOpTypeVector);
      }

      assert(inst_.type_id);
      if (operand_index_ == 2) return GetMtfIdOfType(inst_.type_id);
      if (operand_index_ == 3)
        return GetMtfIdOfType(GetVectorComponentType(inst_.type_id));
      break;
    }

    case SpvOpDot: {
      if (operand_index_ == 0) return GetMtfIdGeneratedByOpcode(SpvOpTypeFloat);

      assert(inst_.type_id);
      if (operand_index_ == 2)
        return GetMtfVectorOfComponentType(inst_.type_id);
      if (operand_index_ == 3) {
        const uint32_t vector_id = GetInstWords()[3];
        const uint32_t vector_type = id_to_type_id_.at(vector_id);
        return GetMtfIdOfType(vector_type);
      }
      break;
    }

    case SpvOpTypeVector: {
      if (operand_index_ == 1) {
        return kMtfTypeScalar;
      }
      break;
    }

    case SpvOpTypeMatrix: {
      if (operand_index_ == 1) {
        return GetMtfIdGeneratedByOpcode(SpvOpTypeVector);
      }
      break;
    }

    case SpvOpTypePointer: {
      if (operand_index_ == 2) {
        return kMtfTypeNonFunction;
      }
      break;
    }

    case SpvOpTypeStruct: {
      if (operand_index_ >= 1) {
        return kMtfTypeNonFunction;
      }
      break;
    }

    case SpvOpTypeFunction: {
      if (operand_index_ == 1) {
        return kMtfTypeNonFunction;
      }

      if (operand_index_ >= 2) {
        return kMtfTypeNonFunction;
      }
      break;
    }

    case SpvOpLoad: {
      if (operand_index_ == 0) return kMtfTypeNonFunction;

      if (operand_index_ == 2) {
        assert(inst_.type_id);
        return GetMtfPointerToType(inst_.type_id);
      }
      break;
    }

    case SpvOpStore: {
      if (operand_index_ == 0)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypePointer);
      if (operand_index_ == 1) {
        const uint32_t pointer_id = GetInstWords()[1];
        const uint32_t pointer_type = id_to_type_id_.at(pointer_id);
        const val::Instruction* pointer_inst = FindDef(pointer_type);
        assert(pointer_inst);
        assert(pointer_inst->opcode() == SpvOpTypePointer);
        const uint32_t data_type =
            pointer_inst->word(pointer_inst->operands()[2].offset);
        return GetMtfIdOfType(data_type);
      }
      break;
    }

    case SpvOpVariable: {
      if (operand_index_ == 0)
        return GetMtfIdGeneratedByOpcode(SpvOpTypePointer);
      break;
    }

    case SpvOpAccessChain: {
      if (operand_index_ == 0)
        return GetMtfIdGeneratedByOpcode(SpvOpTypePointer);
      if (operand_index_ == 2) return kMtfTypePointerToComposite;
      if (operand_index_ >= 3)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeInt);
      break;
    }

    case SpvOpCompositeConstruct: {
      if (operand_index_ == 0) return kMtfTypeComposite;
      if (operand_index_ >= 2) {
        const uint32_t composite_type = GetInstWords()[1];
        if (multi_mtf_.HasValue(kMtfTypeFloatScalarOrVector, composite_type))
          return kMtfFloatScalarOrVector;
        if (multi_mtf_.HasValue(kMtfTypeIntScalarOrVector, composite_type))
          return kMtfIntScalarOrVector;
        if (multi_mtf_.HasValue(kMtfTypeBoolScalarOrVector, composite_type))
          return kMtfBoolScalarOrVector;
      }
      break;
    }

    case SpvOpCompositeExtract: {
      if (operand_index_ == 2) return kMtfComposite;
      break;
    }

    case SpvOpConstantComposite: {
      if (operand_index_ == 0) return kMtfTypeComposite;
      if (operand_index_ >= 2) {
        const val::Instruction* composite_type_inst = FindDef(inst_.type_id);
        assert(composite_type_inst);
        if (composite_type_inst->opcode() == SpvOpTypeVector) {
          return GetMtfIdOfType(composite_type_inst->word(2));
        }
      }
      break;
    }

    case SpvOpExtInst: {
      if (operand_index_ == 2)
        return GetMtfIdGeneratedByOpcode(SpvOpExtInstImport);
      if (operand_index_ >= 4) {
        const uint32_t return_type = GetInstWords()[1];
        const uint32_t ext_inst_type = inst_.ext_inst_type;
        const uint32_t ext_inst_index = GetInstWords()[4];
        // TODO(atgoo@github.com) The list of extended instructions is
        // incomplete. Only common instructions and low-hanging fruits listed.
        if (ext_inst_type == SPV_EXT_INST_TYPE_GLSL_STD_450) {
          switch (ext_inst_index) {
            case GLSLstd450FAbs:
            case GLSLstd450FClamp:
            case GLSLstd450FMax:
            case GLSLstd450FMin:
            case GLSLstd450FMix:
            case GLSLstd450Step:
            case GLSLstd450SmoothStep:
            case GLSLstd450Fma:
            case GLSLstd450Pow:
            case GLSLstd450Exp:
            case GLSLstd450Exp2:
            case GLSLstd450Log:
            case GLSLstd450Log2:
            case GLSLstd450Sqrt:
            case GLSLstd450InverseSqrt:
            case GLSLstd450Fract:
            case GLSLstd450Floor:
            case GLSLstd450Ceil:
            case GLSLstd450Radians:
            case GLSLstd450Degrees:
            case GLSLstd450Sin:
            case GLSLstd450Cos:
            case GLSLstd450Tan:
            case GLSLstd450Sinh:
            case GLSLstd450Cosh:
            case GLSLstd450Tanh:
            case GLSLstd450Asin:
            case GLSLstd450Acos:
            case GLSLstd450Atan:
            case GLSLstd450Atan2:
            case GLSLstd450Asinh:
            case GLSLstd450Acosh:
            case GLSLstd450Atanh:
            case GLSLstd450MatrixInverse:
            case GLSLstd450Cross:
            case GLSLstd450Normalize:
            case GLSLstd450Reflect:
            case GLSLstd450FaceForward:
              return GetMtfIdOfType(return_type);
            case GLSLstd450Length:
            case GLSLstd450Distance:
            case GLSLstd450Refract:
              return kMtfFloatScalarOrVector;
            default:
              break;
          }
        } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_STD) {
          switch (ext_inst_index) {
            case OpenCLLIB::Fabs:
            case OpenCLLIB::FClamp:
            case OpenCLLIB::Fmax:
            case OpenCLLIB::Fmin:
            case OpenCLLIB::Step:
            case OpenCLLIB::Smoothstep:
            case OpenCLLIB::Fma:
            case OpenCLLIB::Pow:
            case OpenCLLIB::Exp:
            case OpenCLLIB::Exp2:
            case OpenCLLIB::Log:
            case OpenCLLIB::Log2:
            case OpenCLLIB::Sqrt:
            case OpenCLLIB::Rsqrt:
            case OpenCLLIB::Fract:
            case OpenCLLIB::Floor:
            case OpenCLLIB::Ceil:
            case OpenCLLIB::Radians:
            case OpenCLLIB::Degrees:
            case OpenCLLIB::Sin:
            case OpenCLLIB::Cos:
            case OpenCLLIB::Tan:
            case OpenCLLIB::Sinh:
            case OpenCLLIB::Cosh:
            case OpenCLLIB::Tanh:
            case OpenCLLIB::Asin:
            case OpenCLLIB::Acos:
            case OpenCLLIB::Atan:
            case OpenCLLIB::Atan2:
            case OpenCLLIB::Asinh:
            case OpenCLLIB::Acosh:
            case OpenCLLIB::Atanh:
            case OpenCLLIB::Cross:
            case OpenCLLIB::Normalize:
              return GetMtfIdOfType(return_type);
            case OpenCLLIB::Length:
            case OpenCLLIB::Distance:
              return kMtfFloatScalarOrVector;
            default:
              break;
          }
        }
      }
      break;
    }

    case SpvOpFunction: {
      if (operand_index_ == 0) return kMtfTypeReturnedByFunction;

      if (operand_index_ == 3) {
        const uint32_t return_type = GetInstWords()[1];
        return GetMtfFunctionTypeWithReturnType(return_type);
      }
      break;
    }

    case SpvOpFunctionCall: {
      if (operand_index_ == 0) return kMtfTypeReturnedByFunction;

      if (operand_index_ == 2) {
        const uint32_t return_type = GetInstWords()[1];
        return GetMtfFunctionWithReturnType(return_type);
      }

      if (operand_index_ >= 3) {
        const uint32_t function_id = GetInstWords()[3];
        const val::Instruction* function_inst = FindDef(function_id);
        if (!function_inst) return kMtfObject;

        assert(function_inst->opcode() == SpvOpFunction);

        const uint32_t function_type_id = function_inst->word(4);
        const val::Instruction* function_type_inst = FindDef(function_type_id);
        assert(function_type_inst);
        assert(function_type_inst->opcode() == SpvOpTypeFunction);

        const uint32_t argument_type = function_type_inst->word(operand_index_);
        return GetMtfIdOfType(argument_type);
      }
      break;
    }

    case SpvOpReturnValue: {
      if (operand_index_ == 0) return GetMtfIdOfType(cur_function_return_type_);
      break;
    }

    case SpvOpBranchConditional: {
      if (operand_index_ == 0)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeBool);
      break;
    }

    case SpvOpSampledImage: {
      if (operand_index_ == 0)
        return GetMtfIdGeneratedByOpcode(SpvOpTypeSampledImage);
      if (operand_index_ == 2)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeImage);
      if (operand_index_ == 3)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeSampler);
      break;
    }

    case SpvOpImageSampleImplicitLod: {
      if (operand_index_ == 0)
        return GetMtfIdGeneratedByOpcode(SpvOpTypeVector);
      if (operand_index_ == 2)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeSampledImage);
      if (operand_index_ == 3)
        return GetMtfIdWithTypeGeneratedByOpcode(SpvOpTypeVector);
      break;
    }

    default:
      break;
  }

  return kMtfNone;
}

}  // namespace comp
}  // namespace spvtools
