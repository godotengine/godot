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

#include "source/name_mapper.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "source/binary.h"
#include "source/latest_version_spirv_header.h"
#include "source/parsed_operand.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace {

// Converts a uint32_t to its string decimal representation.
std::string to_string(uint32_t id) {
  // Use stringstream, since some versions of Android compilers lack
  // std::to_string.
  std::stringstream os;
  os << id;
  return os.str();
}

}  // anonymous namespace

NameMapper GetTrivialNameMapper() { return to_string; }

FriendlyNameMapper::FriendlyNameMapper(const spv_const_context context,
                                       const uint32_t* code,
                                       const size_t wordCount)
    : grammar_(AssemblyGrammar(context)) {
  spv_diagnostic diag = nullptr;
  // We don't care if the parse fails.
  spvBinaryParse(context, this, code, wordCount, nullptr,
                 ParseInstructionForwarder, &diag);
  spvDiagnosticDestroy(diag);
}

std::string FriendlyNameMapper::NameForId(uint32_t id) {
  auto iter = name_for_id_.find(id);
  if (iter == name_for_id_.end()) {
    // It must have been an invalid module, so just return a trivial mapping.
    // We don't care about uniqueness.
    return to_string(id);
  } else {
    return iter->second;
  }
}

std::string FriendlyNameMapper::Sanitize(const std::string& suggested_name) {
  if (suggested_name.empty()) return "_";
  // Otherwise, replace invalid characters by '_'.
  std::string result;
  std::string valid =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "_0123456789";
  std::transform(suggested_name.begin(), suggested_name.end(),
                 std::back_inserter(result), [&valid](const char c) {
                   return (std::string::npos == valid.find(c)) ? '_' : c;
                 });
  return result;
}

void FriendlyNameMapper::SaveName(uint32_t id,
                                  const std::string& suggested_name) {
  if (name_for_id_.find(id) != name_for_id_.end()) return;

  const std::string sanitized_suggested_name = Sanitize(suggested_name);
  std::string name = sanitized_suggested_name;
  auto inserted = used_names_.insert(name);
  if (!inserted.second) {
    const std::string base_name = sanitized_suggested_name + "_";
    for (uint32_t index = 0; !inserted.second; ++index) {
      name = base_name + to_string(index);
      inserted = used_names_.insert(name);
    }
  }
  name_for_id_[id] = name;
}

void FriendlyNameMapper::SaveBuiltInName(uint32_t target_id,
                                         uint32_t built_in) {
#define GLCASE(name)                  \
  case spv::BuiltIn::name:            \
    SaveName(target_id, "gl_" #name); \
    return;
#define GLCASE2(name, suggested)           \
  case spv::BuiltIn::name:                 \
    SaveName(target_id, "gl_" #suggested); \
    return;
#define CASE(name)              \
  case spv::BuiltIn::name:      \
    SaveName(target_id, #name); \
    return;
  switch (spv::BuiltIn(built_in)) {
    GLCASE(Position)
    GLCASE(PointSize)
    GLCASE(ClipDistance)
    GLCASE(CullDistance)
    GLCASE2(VertexId, VertexID)
    GLCASE2(InstanceId, InstanceID)
    GLCASE2(PrimitiveId, PrimitiveID)
    GLCASE2(InvocationId, InvocationID)
    GLCASE(Layer)
    GLCASE(ViewportIndex)
    GLCASE(TessLevelOuter)
    GLCASE(TessLevelInner)
    GLCASE(TessCoord)
    GLCASE(PatchVertices)
    GLCASE(FragCoord)
    GLCASE(PointCoord)
    GLCASE(FrontFacing)
    GLCASE2(SampleId, SampleID)
    GLCASE(SamplePosition)
    GLCASE(SampleMask)
    GLCASE(FragDepth)
    GLCASE(HelperInvocation)
    GLCASE2(NumWorkgroups, NumWorkGroups)
    GLCASE2(WorkgroupSize, WorkGroupSize)
    GLCASE2(WorkgroupId, WorkGroupID)
    GLCASE2(LocalInvocationId, LocalInvocationID)
    GLCASE2(GlobalInvocationId, GlobalInvocationID)
    GLCASE(LocalInvocationIndex)
    CASE(WorkDim)
    CASE(GlobalSize)
    CASE(EnqueuedWorkgroupSize)
    CASE(GlobalOffset)
    CASE(GlobalLinearId)
    CASE(SubgroupSize)
    CASE(SubgroupMaxSize)
    CASE(NumSubgroups)
    CASE(NumEnqueuedSubgroups)
    CASE(SubgroupId)
    CASE(SubgroupLocalInvocationId)
    GLCASE(VertexIndex)
    GLCASE(InstanceIndex)
    GLCASE(BaseInstance)
    CASE(SubgroupEqMaskKHR)
    CASE(SubgroupGeMaskKHR)
    CASE(SubgroupGtMaskKHR)
    CASE(SubgroupLeMaskKHR)
    CASE(SubgroupLtMaskKHR)
    default:
      break;
  }
#undef GLCASE
#undef GLCASE2
#undef CASE
}

spv_result_t FriendlyNameMapper::ParseInstruction(
    const spv_parsed_instruction_t& inst) {
  const auto result_id = inst.result_id;
  switch (spv::Op(inst.opcode)) {
    case spv::Op::OpName:
      SaveName(inst.words[1], spvDecodeLiteralStringOperand(inst, 1));
      break;
    case spv::Op::OpDecorate:
      // Decorations come after OpName.  So OpName will take precedence over
      // decorations.
      //
      // In theory, we should also handle OpGroupDecorate.  But that's unlikely
      // to occur.
      if (spv::Decoration(inst.words[2]) == spv::Decoration::BuiltIn) {
        assert(inst.num_words > 3);
        SaveBuiltInName(inst.words[1], inst.words[3]);
      }
      break;
    case spv::Op::OpTypeVoid:
      SaveName(result_id, "void");
      break;
    case spv::Op::OpTypeBool:
      SaveName(result_id, "bool");
      break;
    case spv::Op::OpTypeInt: {
      std::string signedness;
      std::string root;
      const auto bit_width = inst.words[2];
      switch (bit_width) {
        case 8:
          root = "char";
          break;
        case 16:
          root = "short";
          break;
        case 32:
          root = "int";
          break;
        case 64:
          root = "long";
          break;
        default:
          root = to_string(bit_width);
          signedness = "i";
          break;
      }
      if (0 == inst.words[3]) signedness = "u";
      SaveName(result_id, signedness + root);
    } break;
    case spv::Op::OpTypeFloat: {
      const auto bit_width = inst.words[2];
      switch (bit_width) {
        case 16:
          SaveName(result_id, "half");
          break;
        case 32:
          SaveName(result_id, "float");
          break;
        case 64:
          SaveName(result_id, "double");
          break;
        default:
          SaveName(result_id, std::string("fp") + to_string(bit_width));
          break;
      }
    } break;
    case spv::Op::OpTypeVector:
      SaveName(result_id, std::string("v") + to_string(inst.words[3]) +
                              NameForId(inst.words[2]));
      break;
    case spv::Op::OpTypeMatrix:
      SaveName(result_id, std::string("mat") + to_string(inst.words[3]) +
                              NameForId(inst.words[2]));
      break;
    case spv::Op::OpTypeArray:
      SaveName(result_id, std::string("_arr_") + NameForId(inst.words[2]) +
                              "_" + NameForId(inst.words[3]));
      break;
    case spv::Op::OpTypeRuntimeArray:
      SaveName(result_id,
               std::string("_runtimearr_") + NameForId(inst.words[2]));
      break;
    case spv::Op::OpTypePointer:
      SaveName(result_id, std::string("_ptr_") +
                              NameForEnumOperand(SPV_OPERAND_TYPE_STORAGE_CLASS,
                                                 inst.words[2]) +
                              "_" + NameForId(inst.words[3]));
      break;
    case spv::Op::OpTypePipe:
      SaveName(result_id,
               std::string("Pipe") +
                   NameForEnumOperand(SPV_OPERAND_TYPE_ACCESS_QUALIFIER,
                                      inst.words[2]));
      break;
    case spv::Op::OpTypeEvent:
      SaveName(result_id, "Event");
      break;
    case spv::Op::OpTypeDeviceEvent:
      SaveName(result_id, "DeviceEvent");
      break;
    case spv::Op::OpTypeReserveId:
      SaveName(result_id, "ReserveId");
      break;
    case spv::Op::OpTypeQueue:
      SaveName(result_id, "Queue");
      break;
    case spv::Op::OpTypeOpaque:
      SaveName(result_id, std::string("Opaque_") +
                              Sanitize(spvDecodeLiteralStringOperand(inst, 1)));
      break;
    case spv::Op::OpTypePipeStorage:
      SaveName(result_id, "PipeStorage");
      break;
    case spv::Op::OpTypeNamedBarrier:
      SaveName(result_id, "NamedBarrier");
      break;
    case spv::Op::OpTypeStruct:
      // Structs are mapped rather simplisitically. Just indicate that they
      // are a struct and then give the raw Id number.
      SaveName(result_id, std::string("_struct_") + to_string(result_id));
      break;
    case spv::Op::OpConstantTrue:
      SaveName(result_id, "true");
      break;
    case spv::Op::OpConstantFalse:
      SaveName(result_id, "false");
      break;
    case spv::Op::OpConstant: {
      std::ostringstream value;
      EmitNumericLiteral(&value, inst, inst.operands[2]);
      auto value_str = value.str();
      // Use 'n' to signify negative. Other invalid characters will be mapped
      // to underscore.
      for (auto& c : value_str)
        if (c == '-') c = 'n';
      SaveName(result_id, NameForId(inst.type_id) + "_" + value_str);
    } break;
    default:
      // If this instruction otherwise defines an Id, then save a mapping for
      // it.  This is needed to ensure uniqueness in there is an OpName with
      // string something like "1" that might collide with this result_id.
      // We should only do this if a name hasn't already been registered by some
      // previous forward reference.
      if (result_id && name_for_id_.find(result_id) == name_for_id_.end())
        SaveName(result_id, to_string(result_id));
      break;
  }
  return SPV_SUCCESS;
}

std::string FriendlyNameMapper::NameForEnumOperand(spv_operand_type_t type,
                                                   uint32_t word) {
  spv_operand_desc desc = nullptr;
  if (SPV_SUCCESS == grammar_.lookupOperand(type, word, &desc)) {
    return desc->name;
  } else {
    // Invalid input.  Just give something.
    return std::string("StorageClass") + to_string(word);
  }
}

}  // namespace spvtools
