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

#include "source/val/validate.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/spirv_validator_options.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Distinguish between row and column major matrix layouts.
enum MatrixLayout { kRowMajor, kColumnMajor };

// A functor for hashing a pair of integers.
struct PairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t> pair) const {
    const uint32_t a = pair.first;
    const uint32_t b = pair.second;
    const uint32_t rotated_b = (b >> 2) | ((b & 3) << 30);
    return a ^ rotated_b;
  }
};

// A functor for hashing decoration types.
struct SpvDecorationHash {
  std::size_t operator()(SpvDecoration dec) const {
    return static_cast<std::size_t>(dec);
  }
};

// Struct member layout attributes that are inherited through arrays.
struct LayoutConstraints {
  explicit LayoutConstraints(
      MatrixLayout the_majorness = MatrixLayout::kColumnMajor,
      uint32_t stride = 0)
      : majorness(the_majorness), matrix_stride(stride) {}
  MatrixLayout majorness;
  uint32_t matrix_stride;
};

// A type for mapping (struct id, member id) to layout constraints.
using MemberConstraints = std::unordered_map<std::pair<uint32_t, uint32_t>,
                                             LayoutConstraints, PairHash>;

// Returns the array stride of the given array type.
uint32_t GetArrayStride(uint32_t array_id, ValidationState_t& vstate) {
  for (auto& decoration : vstate.id_decorations(array_id)) {
    if (SpvDecorationArrayStride == decoration.dec_type()) {
      return decoration.params()[0];
    }
  }
  return 0;
}

// Returns true if the given variable has a BuiltIn decoration.
bool isBuiltInVar(uint32_t var_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(var_id);
  return std::any_of(
      decorations.begin(), decorations.end(),
      [](const Decoration& d) { return SpvDecorationBuiltIn == d.dec_type(); });
}

// Returns true if the given structure type has any members with BuiltIn
// decoration.
bool isBuiltInStruct(uint32_t struct_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(struct_id);
  return std::any_of(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type() &&
               Decoration::kInvalidMember != d.struct_member_index();
      });
}

// Returns true if the given ID has the Import LinkageAttributes decoration.
bool hasImportLinkageAttribute(uint32_t id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(id);
  return std::any_of(decorations.begin(), decorations.end(),
                     [](const Decoration& d) {
                       return SpvDecorationLinkageAttributes == d.dec_type() &&
                              d.params().size() >= 2u &&
                              d.params().back() == SpvLinkageTypeImport;
                     });
}

// Returns a vector of all members of a structure.
std::vector<uint32_t> getStructMembers(uint32_t struct_id,
                                       ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(struct_id);
  return std::vector<uint32_t>(inst->words().begin() + 2, inst->words().end());
}

// Returns a vector of all members of a structure that have specific type.
std::vector<uint32_t> getStructMembers(uint32_t struct_id, SpvOp type,
                                       ValidationState_t& vstate) {
  std::vector<uint32_t> members;
  for (auto id : getStructMembers(struct_id, vstate)) {
    if (type == vstate.FindDef(id)->opcode()) {
      members.push_back(id);
    }
  }
  return members;
}

// Returns whether the given structure is missing Offset decoration for any
// member. Handles also nested structures.
bool isMissingOffsetInStruct(uint32_t struct_id, ValidationState_t& vstate) {
  std::vector<bool> hasOffset(getStructMembers(struct_id, vstate).size(),
                              false);
  // Check offsets of member decorations
  for (auto& decoration : vstate.id_decorations(struct_id)) {
    if (SpvDecorationOffset == decoration.dec_type() &&
        Decoration::kInvalidMember != decoration.struct_member_index()) {
      hasOffset[decoration.struct_member_index()] = true;
    }
  }
  // Check also nested structures
  bool nestedStructsMissingOffset = false;
  for (auto id : getStructMembers(struct_id, SpvOpTypeStruct, vstate)) {
    if (isMissingOffsetInStruct(id, vstate)) {
      nestedStructsMissingOffset = true;
      break;
    }
  }
  return nestedStructsMissingOffset ||
         !std::all_of(hasOffset.begin(), hasOffset.end(),
                      [](const bool b) { return b; });
}

// Rounds x up to the next alignment. Assumes alignment is a power of two.
uint32_t align(uint32_t x, uint32_t alignment) {
  return (x + alignment - 1) & ~(alignment - 1);
}

// Returns base alignment of struct member. If |roundUp| is true, also
// ensure that structs and arrays are aligned at least to a multiple of 16
// bytes.
uint32_t getBaseAlignment(uint32_t member_id, bool roundUp,
                          const LayoutConstraints& inherited,
                          MemberConstraints& constraints,
                          ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(member_id);
  const auto& words = inst->words();
  // Minimal alignment is byte-aligned.
  uint32_t baseAlignment = 1;
  switch (inst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      baseAlignment = words[2] / 8;
      break;
    case SpvOpTypeVector: {
      const auto componentId = words[2];
      const auto numComponents = words[3];
      const auto componentAlignment = getBaseAlignment(
          componentId, roundUp, inherited, constraints, vstate);
      baseAlignment =
          componentAlignment * (numComponents == 3 ? 4 : numComponents);
      break;
    }
    case SpvOpTypeMatrix: {
      const auto column_type = words[2];
      if (inherited.majorness == kColumnMajor) {
        baseAlignment = getBaseAlignment(column_type, roundUp, inherited,
                                         constraints, vstate);
      } else {
        // A row-major matrix of C columns has a base alignment equal to the
        // base alignment of a vector of C matrix components.
        const auto num_columns = words[3];
        const auto component_inst = vstate.FindDef(column_type);
        const auto component_id = component_inst->words()[2];
        const auto componentAlignment = getBaseAlignment(
            component_id, roundUp, inherited, constraints, vstate);
        baseAlignment =
            componentAlignment * (num_columns == 3 ? 4 : num_columns);
      }
    } break;
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
      baseAlignment =
          getBaseAlignment(words[2], roundUp, inherited, constraints, vstate);
      if (roundUp) baseAlignment = align(baseAlignment, 16u);
      break;
    case SpvOpTypeStruct: {
      const auto members = getStructMembers(member_id, vstate);
      for (uint32_t memberIdx = 0, numMembers = uint32_t(members.size());
           memberIdx < numMembers; ++memberIdx) {
        const auto id = members[memberIdx];
        const auto& constraint =
            constraints[std::make_pair(member_id, memberIdx)];
        baseAlignment = std::max(
            baseAlignment,
            getBaseAlignment(id, roundUp, constraint, constraints, vstate));
      }
      if (roundUp) baseAlignment = align(baseAlignment, 16u);
      break;
    }
    case SpvOpTypePointer:
      baseAlignment = vstate.pointer_size_and_alignment();
      break;
    default:
      assert(0);
      break;
  }

  return baseAlignment;
}

// Returns scalar alignment of a type.
uint32_t getScalarAlignment(uint32_t type_id, ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(type_id);
  const auto& words = inst->words();
  switch (inst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      return words[2] / 8;
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray: {
      const auto compositeMemberTypeId = words[2];
      return getScalarAlignment(compositeMemberTypeId, vstate);
    }
    case SpvOpTypeStruct: {
      const auto members = getStructMembers(type_id, vstate);
      uint32_t max_member_alignment = 1;
      for (uint32_t memberIdx = 0, numMembers = uint32_t(members.size());
           memberIdx < numMembers; ++memberIdx) {
        const auto id = members[memberIdx];
        uint32_t member_alignment = getScalarAlignment(id, vstate);
        if (member_alignment > max_member_alignment) {
          max_member_alignment = member_alignment;
        }
      }
      return max_member_alignment;
    } break;
    case SpvOpTypePointer:
      return vstate.pointer_size_and_alignment();
    default:
      assert(0);
      break;
  }

  return 1;
}

// Returns size of a struct member. Doesn't include padding at the end of struct
// or array.  Assumes that in the struct case, all members have offsets.
uint32_t getSize(uint32_t member_id, const LayoutConstraints& inherited,
                 MemberConstraints& constraints, ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(member_id);
  const auto& words = inst->words();
  switch (inst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      return words[2] / 8;
    case SpvOpTypeVector: {
      const auto componentId = words[2];
      const auto numComponents = words[3];
      const auto componentSize =
          getSize(componentId, inherited, constraints, vstate);
      const auto size = componentSize * numComponents;
      return size;
    }
    case SpvOpTypeArray: {
      const auto sizeInst = vstate.FindDef(words[3]);
      if (spvOpcodeIsSpecConstant(sizeInst->opcode())) return 0;
      assert(SpvOpConstant == sizeInst->opcode());
      const uint32_t num_elem = sizeInst->words()[3];
      const uint32_t elem_type = words[2];
      const uint32_t elem_size =
          getSize(elem_type, inherited, constraints, vstate);
      // Account for gaps due to alignments in the first N-1 elements,
      // then add the size of the last element.
      const auto size =
          (num_elem - 1) * GetArrayStride(member_id, vstate) + elem_size;
      return size;
    }
    case SpvOpTypeRuntimeArray:
      return 0;
    case SpvOpTypeMatrix: {
      const auto num_columns = words[3];
      if (inherited.majorness == kColumnMajor) {
        return num_columns * inherited.matrix_stride;
      } else {
        // Row major case.
        const auto column_type = words[2];
        const auto component_inst = vstate.FindDef(column_type);
        const auto num_rows = component_inst->words()[3];
        const auto scalar_elem_type = component_inst->words()[2];
        const uint32_t scalar_elem_size =
            getSize(scalar_elem_type, inherited, constraints, vstate);
        return (num_rows - 1) * inherited.matrix_stride +
               num_columns * scalar_elem_size;
      }
    }
    case SpvOpTypeStruct: {
      const auto& members = getStructMembers(member_id, vstate);
      if (members.empty()) return 0;
      const auto lastIdx = uint32_t(members.size() - 1);
      const auto& lastMember = members.back();
      uint32_t offset = 0xffffffff;
      // Find the offset of the last element and add the size.
      for (auto& decoration : vstate.id_decorations(member_id)) {
        if (SpvDecorationOffset == decoration.dec_type() &&
            decoration.struct_member_index() == (int)lastIdx) {
          offset = decoration.params()[0];
        }
      }
      // This check depends on the fact that all members have offsets.  This
      // has been checked earlier in the flow.
      assert(offset != 0xffffffff);
      const auto& constraint = constraints[std::make_pair(lastMember, lastIdx)];
      return offset + getSize(lastMember, constraint, constraints, vstate);
    }
    case SpvOpTypePointer:
      return vstate.pointer_size_and_alignment();
    default:
      assert(0);
      return 0;
  }
}

// A member is defined to improperly straddle if either of the following are
// true:
// - It is a vector with total size less than or equal to 16 bytes, and has
// Offset decorations placing its first byte at F and its last byte at L, where
// floor(F / 16) != floor(L / 16).
// - It is a vector with total size greater than 16 bytes and has its Offset
// decorations placing its first byte at a non-integer multiple of 16.
bool hasImproperStraddle(uint32_t id, uint32_t offset,
                         const LayoutConstraints& inherited,
                         MemberConstraints& constraints,
                         ValidationState_t& vstate) {
  const auto size = getSize(id, inherited, constraints, vstate);
  const auto F = offset;
  const auto L = offset + size - 1;
  if (size <= 16) {
    if ((F >> 4) != (L >> 4)) return true;
  } else {
    if (F % 16 != 0) return true;
  }
  return false;
}

// Returns true if |offset| satsifies an alignment to |alignment|.  In the case
// of |alignment| of zero, the |offset| must also be zero.
bool IsAlignedTo(uint32_t offset, uint32_t alignment) {
  if (alignment == 0) return offset == 0;
  return 0 == (offset % alignment);
}

// Returns SPV_SUCCESS if the given struct satisfies standard layout rules for
// Block or BufferBlocks in Vulkan.  Otherwise emits a diagnostic and returns
// something other than SPV_SUCCESS.  Matrices inherit the specified column
// or row major-ness.
spv_result_t checkLayout(uint32_t struct_id, const char* storage_class_str,
                         const char* decoration_str, bool blockRules,
                         uint32_t incoming_offset,
                         MemberConstraints& constraints,
                         ValidationState_t& vstate) {
  if (vstate.options()->skip_block_layout) return SPV_SUCCESS;

  // Relaxed layout and scalar layout can both be in effect at the same time.
  // For example, relaxed layout is implied by Vulkan 1.1.  But scalar layout
  // is more permissive than relaxed layout.
  const bool relaxed_block_layout = vstate.IsRelaxedBlockLayout();
  const bool scalar_block_layout = vstate.options()->scalar_block_layout;

  auto fail = [&vstate, struct_id, storage_class_str, decoration_str,
               blockRules, relaxed_block_layout,
               scalar_block_layout](uint32_t member_idx) -> DiagnosticStream {
    DiagnosticStream ds =
        std::move(vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(struct_id))
                  << "Structure id " << struct_id << " decorated as "
                  << decoration_str << " for variable in " << storage_class_str
                  << " storage class must follow "
                  << (scalar_block_layout
                          ? "scalar "
                          : (relaxed_block_layout ? "relaxed " : "standard "))
                  << (blockRules ? "uniform buffer" : "storage buffer")
                  << " layout rules: member " << member_idx << " ");
    return ds;
  };

  const auto& members = getStructMembers(struct_id, vstate);

  // To check for member overlaps, we want to traverse the members in
  // offset order.
  struct MemberOffsetPair {
    uint32_t member;
    uint32_t offset;
  };
  std::vector<MemberOffsetPair> member_offsets;
  member_offsets.reserve(members.size());
  for (uint32_t memberIdx = 0, numMembers = uint32_t(members.size());
       memberIdx < numMembers; memberIdx++) {
    uint32_t offset = 0xffffffff;
    for (auto& decoration : vstate.id_decorations(struct_id)) {
      if (decoration.struct_member_index() == (int)memberIdx) {
        switch (decoration.dec_type()) {
          case SpvDecorationOffset:
            offset = decoration.params()[0];
            break;
          default:
            break;
        }
      }
    }
    member_offsets.push_back(
        MemberOffsetPair{memberIdx, incoming_offset + offset});
  }
  std::stable_sort(
      member_offsets.begin(), member_offsets.end(),
      [](const MemberOffsetPair& lhs, const MemberOffsetPair& rhs) {
        return lhs.offset < rhs.offset;
      });

  // Now scan from lowest offest to highest offset.
  uint32_t nextValidOffset = 0;
  for (size_t ordered_member_idx = 0;
       ordered_member_idx < member_offsets.size(); ordered_member_idx++) {
    const auto& member_offset = member_offsets[ordered_member_idx];
    const auto memberIdx = member_offset.member;
    const auto offset = member_offset.offset;
    auto id = members[member_offset.member];
    const LayoutConstraints& constraint =
        constraints[std::make_pair(struct_id, uint32_t(memberIdx))];
    // Scalar layout takes precedence because it's more permissive, and implying
    // an alignment that divides evenly into the alignment that would otherwise
    // be used.
    const auto alignment =
        scalar_block_layout
            ? getScalarAlignment(id, vstate)
            : getBaseAlignment(id, blockRules, constraint, constraints, vstate);
    const auto inst = vstate.FindDef(id);
    const auto opcode = inst->opcode();
    const auto size = getSize(id, constraint, constraints, vstate);
    // Check offset.
    if (offset == 0xffffffff)
      return fail(memberIdx) << "is missing an Offset decoration";
    if (!scalar_block_layout && relaxed_block_layout &&
        opcode == SpvOpTypeVector) {
      // In relaxed block layout, the vector offset must be aligned to the
      // vector's scalar element type.
      const auto componentId = inst->words()[2];
      const auto scalar_alignment = getScalarAlignment(componentId, vstate);
      if (!IsAlignedTo(offset, scalar_alignment)) {
        return fail(memberIdx)
               << "at offset " << offset
               << " is not aligned to scalar element size " << scalar_alignment;
      }
    } else {
      // Without relaxed block layout, the offset must be divisible by the
      // alignment requirement.
      if (!IsAlignedTo(offset, alignment)) {
        return fail(memberIdx)
               << "at offset " << offset << " is not aligned to " << alignment;
      }
    }
    if (offset < nextValidOffset)
      return fail(memberIdx) << "at offset " << offset
                             << " overlaps previous member ending at offset "
                             << nextValidOffset - 1;
    if (!scalar_block_layout && relaxed_block_layout) {
      // Check improper straddle of vectors.
      if (SpvOpTypeVector == opcode &&
          hasImproperStraddle(id, offset, constraint, constraints, vstate))
        return fail(memberIdx)
               << "is an improperly straddling vector at offset " << offset;
    }
    // Check struct members recursively.
    spv_result_t recursive_status = SPV_SUCCESS;
    if (SpvOpTypeStruct == opcode &&
        SPV_SUCCESS != (recursive_status = checkLayout(
                            id, storage_class_str, decoration_str, blockRules,
                            offset, constraints, vstate)))
      return recursive_status;
    // Check matrix stride.
    if (SpvOpTypeMatrix == opcode) {
      for (auto& decoration : vstate.id_decorations(id)) {
        if (SpvDecorationMatrixStride == decoration.dec_type() &&
            !IsAlignedTo(decoration.params()[0], alignment))
          return fail(memberIdx)
                 << "is a matrix with stride " << decoration.params()[0]
                 << " not satisfying alignment to " << alignment;
      }
    }

    // Check arrays and runtime arrays recursively.
    auto array_inst = inst;
    auto array_alignment = alignment;
    while (array_inst->opcode() == SpvOpTypeArray ||
           array_inst->opcode() == SpvOpTypeRuntimeArray) {
      const auto typeId = array_inst->word(2);
      const auto element_inst = vstate.FindDef(typeId);
      // Check array stride.
      auto array_stride = 0;
      for (auto& decoration : vstate.id_decorations(array_inst->id())) {
        if (SpvDecorationArrayStride == decoration.dec_type()) {
          array_stride = decoration.params()[0];
          if (!IsAlignedTo(array_stride, array_alignment))
            return fail(memberIdx)
                   << "contains an array with stride " << decoration.params()[0]
                   << " not satisfying alignment to " << alignment;
        }
      }

      bool is_int32 = false;
      bool is_const = false;
      uint32_t num_elements = 0;
      if (array_inst->opcode() == SpvOpTypeArray) {
        std::tie(is_int32, is_const, num_elements) =
            vstate.EvalInt32IfConst(array_inst->word(3));
      }
      num_elements = std::max(1u, num_elements);
      // Check each element recursively if it is a struct. There is a
      // limitation to this check if the array size is a spec constant or is a
      // runtime array then we will only check a single element. This means
      // some improper straddles might be missed.
      for (uint32_t i = 0; i < num_elements; ++i) {
        uint32_t next_offset = i * array_stride + offset;
        if (SpvOpTypeStruct == element_inst->opcode() &&
            SPV_SUCCESS != (recursive_status = checkLayout(
                                typeId, storage_class_str, decoration_str,
                                blockRules, next_offset, constraints, vstate)))
          return recursive_status;
      }

      // Proceed to the element in case it is an array.
      array_inst = element_inst;
      array_alignment = scalar_block_layout
                            ? getScalarAlignment(array_inst->id(), vstate)
                            : getBaseAlignment(array_inst->id(), blockRules,
                                               constraint, constraints, vstate);
    }
    nextValidOffset = offset + size;
    if (!scalar_block_layout && blockRules &&
        (SpvOpTypeArray == opcode || SpvOpTypeStruct == opcode)) {
      // Uniform block rules don't permit anything in the padding of a struct
      // or array.
      nextValidOffset = align(nextValidOffset, alignment);
    }
  }
  return SPV_SUCCESS;
}

// Returns true if variable or structure id has given decoration. Handles also
// nested structures.
bool hasDecoration(uint32_t id, SpvDecoration decoration,
                   ValidationState_t& vstate) {
  for (auto& dec : vstate.id_decorations(id)) {
    if (decoration == dec.dec_type()) return true;
  }
  if (SpvOpTypeStruct != vstate.FindDef(id)->opcode()) {
    return false;
  }
  for (auto member_id : getStructMembers(id, SpvOpTypeStruct, vstate)) {
    if (hasDecoration(member_id, decoration, vstate)) {
      return true;
    }
  }
  return false;
}

// Returns true if all ids of given type have a specified decoration.
bool checkForRequiredDecoration(uint32_t struct_id, SpvDecoration decoration,
                                SpvOp type, ValidationState_t& vstate) {
  const auto& members = getStructMembers(struct_id, vstate);
  for (size_t memberIdx = 0; memberIdx < members.size(); memberIdx++) {
    const auto id = members[memberIdx];
    if (type != vstate.FindDef(id)->opcode()) continue;
    bool found = false;
    for (auto& dec : vstate.id_decorations(id)) {
      if (decoration == dec.dec_type()) found = true;
    }
    for (auto& dec : vstate.id_decorations(struct_id)) {
      if (decoration == dec.dec_type() &&
          (int)memberIdx == dec.struct_member_index()) {
        found = true;
      }
    }
    if (!found) {
      return false;
    }
  }
  for (auto id : getStructMembers(struct_id, SpvOpTypeStruct, vstate)) {
    if (!checkForRequiredDecoration(id, decoration, type, vstate)) {
      return false;
    }
  }
  return true;
}

spv_result_t CheckLinkageAttrOfFunctions(ValidationState_t& vstate) {
  for (const auto& function : vstate.functions()) {
    if (function.block_count() == 0u) {
      // A function declaration (an OpFunction with no basic blocks), must have
      // a Linkage Attributes Decoration with the Import Linkage Type.
      if (!hasImportLinkageAttribute(function.id(), vstate)) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY,
                           vstate.FindDef(function.id()))
               << "Function declaration (id " << function.id()
               << ") must have a LinkageAttributes decoration with the Import "
                  "Linkage type.";
      }
    } else {
      if (hasImportLinkageAttribute(function.id(), vstate)) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY,
                           vstate.FindDef(function.id()))
               << "Function definition (id " << function.id()
               << ") may not be decorated with Import Linkage type.";
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks whether an imported variable is initialized by this module.
spv_result_t CheckImportedVariableInitialization(ValidationState_t& vstate) {
  // According the SPIR-V Spec 2.16.1, it is illegal to initialize an imported
  // variable. This means that a module-scope OpVariable with initialization
  // value cannot be marked with the Import Linkage Type (import type id = 1).
  for (auto global_var_id : vstate.global_vars()) {
    // Initializer <id> is an optional argument for OpVariable. If initializer
    // <id> is present, the instruction will have 5 words.
    auto variable_instr = vstate.FindDef(global_var_id);
    if (variable_instr->words().size() == 5u &&
        hasImportLinkageAttribute(global_var_id, vstate)) {
      return vstate.diag(SPV_ERROR_INVALID_ID, variable_instr)
             << "A module-scope OpVariable with initialization value "
                "cannot be marked with the Import Linkage Type.";
    }
  }
  return SPV_SUCCESS;
}

// Checks whether a builtin variable is valid.
spv_result_t CheckBuiltInVariable(uint32_t var_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(var_id);
  for (const auto& d : decorations) {
    if (spvIsVulkanEnv(vstate.context()->target_env)) {
      if (d.dec_type() == SpvDecorationLocation ||
          d.dec_type() == SpvDecorationComponent) {
        return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
               << "A BuiltIn variable (id " << var_id
               << ") cannot have any Location or Component decorations";
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks whether proper decorations have been appied to the entry points.
spv_result_t CheckDecorationsOfEntryPoints(ValidationState_t& vstate) {
  for (uint32_t entry_point : vstate.entry_points()) {
    const auto& descs = vstate.entry_point_descriptions(entry_point);
    int num_builtin_inputs = 0;
    int num_builtin_outputs = 0;
    for (const auto& desc : descs) {
      for (auto interface : desc.interfaces) {
        Instruction* var_instr = vstate.FindDef(interface);
        if (!var_instr || SpvOpVariable != var_instr->opcode()) {
          return vstate.diag(SPV_ERROR_INVALID_ID, var_instr)
                 << "Interfaces passed to OpEntryPoint must be of type "
                    "OpTypeVariable. Found Op"
                 << spvOpcodeString(var_instr->opcode()) << ".";
        }
        const SpvStorageClass storage_class =
            var_instr->GetOperandAs<SpvStorageClass>(2);
        if (storage_class != SpvStorageClassInput &&
            storage_class != SpvStorageClassOutput) {
          return vstate.diag(SPV_ERROR_INVALID_ID, var_instr)
                 << "OpEntryPoint interfaces must be OpVariables with "
                    "Storage Class of Input(1) or Output(3). Found Storage "
                    "Class "
                 << storage_class << " for Entry Point id " << entry_point
                 << ".";
        }

        const uint32_t ptr_id = var_instr->word(1);
        Instruction* ptr_instr = vstate.FindDef(ptr_id);
        // It is guaranteed (by validator ID checks) that ptr_instr is
        // OpTypePointer. Word 3 of this instruction is the type being pointed
        // to.
        const uint32_t type_id = ptr_instr->word(3);
        Instruction* type_instr = vstate.FindDef(type_id);
        if (type_instr && SpvOpTypeStruct == type_instr->opcode() &&
            isBuiltInStruct(type_id, vstate)) {
          if (storage_class == SpvStorageClassInput) ++num_builtin_inputs;
          if (storage_class == SpvStorageClassOutput) ++num_builtin_outputs;
          if (num_builtin_inputs > 1 || num_builtin_outputs > 1) break;
          if (auto error = CheckBuiltInVariable(interface, vstate))
            return error;
        } else if (isBuiltInVar(interface, vstate)) {
          if (auto error = CheckBuiltInVariable(interface, vstate))
            return error;
        }
      }
      if (num_builtin_inputs > 1 || num_builtin_outputs > 1) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY,
                           vstate.FindDef(entry_point))
               << "There must be at most one object per Storage Class that can "
                  "contain a structure type containing members decorated with "
                  "BuiltIn, consumed per entry-point. Entry Point id "
               << entry_point << " does not meet this requirement.";
      }
      // The LinkageAttributes Decoration cannot be applied to functions
      // targeted by an OpEntryPoint instruction
      for (auto& decoration : vstate.id_decorations(entry_point)) {
        if (SpvDecorationLinkageAttributes == decoration.dec_type()) {
          const char* linkage_name =
              reinterpret_cast<const char*>(&decoration.params()[0]);
          return vstate.diag(SPV_ERROR_INVALID_BINARY,
                             vstate.FindDef(entry_point))
                 << "The LinkageAttributes Decoration (Linkage name: "
                 << linkage_name << ") cannot be applied to function id "
                 << entry_point
                 << " because it is targeted by an OpEntryPoint instruction.";
        }
      }
    }
  }
  return SPV_SUCCESS;
}

// Load |constraints| with all the member constraints for structs contained
// within the given array type.
void ComputeMemberConstraintsForArray(MemberConstraints* constraints,
                                      uint32_t array_id,
                                      const LayoutConstraints& inherited,
                                      ValidationState_t& vstate);

// Load |constraints| with all the member constraints for the given struct,
// and all its contained structs.
void ComputeMemberConstraintsForStruct(MemberConstraints* constraints,
                                       uint32_t struct_id,
                                       const LayoutConstraints& inherited,
                                       ValidationState_t& vstate) {
  assert(constraints);
  const auto& members = getStructMembers(struct_id, vstate);
  for (uint32_t memberIdx = 0, numMembers = uint32_t(members.size());
       memberIdx < numMembers; memberIdx++) {
    LayoutConstraints& constraint =
        (*constraints)[std::make_pair(struct_id, memberIdx)];
    constraint = inherited;
    for (auto& decoration : vstate.id_decorations(struct_id)) {
      if (decoration.struct_member_index() == (int)memberIdx) {
        switch (decoration.dec_type()) {
          case SpvDecorationRowMajor:
            constraint.majorness = kRowMajor;
            break;
          case SpvDecorationColMajor:
            constraint.majorness = kColumnMajor;
            break;
          case SpvDecorationMatrixStride:
            constraint.matrix_stride = decoration.params()[0];
            break;
          default:
            break;
        }
      }
    }

    // Now recurse
    auto member_type_id = members[memberIdx];
    const auto member_type_inst = vstate.FindDef(member_type_id);
    const auto opcode = member_type_inst->opcode();
    switch (opcode) {
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray:
        ComputeMemberConstraintsForArray(constraints, member_type_id, inherited,
                                         vstate);
        break;
      case SpvOpTypeStruct:
        ComputeMemberConstraintsForStruct(constraints, member_type_id,
                                          inherited, vstate);
        break;
      default:
        break;
    }
  }
}

void ComputeMemberConstraintsForArray(MemberConstraints* constraints,
                                      uint32_t array_id,
                                      const LayoutConstraints& inherited,
                                      ValidationState_t& vstate) {
  assert(constraints);
  auto elem_type_id = vstate.FindDef(array_id)->words()[2];
  const auto elem_type_inst = vstate.FindDef(elem_type_id);
  const auto opcode = elem_type_inst->opcode();
  switch (opcode) {
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
      ComputeMemberConstraintsForArray(constraints, elem_type_id, inherited,
                                       vstate);
      break;
    case SpvOpTypeStruct:
      ComputeMemberConstraintsForStruct(constraints, elem_type_id, inherited,
                                        vstate);
      break;
    default:
      break;
  }
}

spv_result_t CheckDecorationsOfBuffers(ValidationState_t& vstate) {
  // Set of entry points that are known to use a push constant.
  std::unordered_set<uint32_t> uses_push_constant;
  for (const auto& inst : vstate.ordered_instructions()) {
    const auto& words = inst.words();
    if (SpvOpVariable == inst.opcode()) {
      const auto var_id = inst.id();
      // For storage class / decoration combinations, see Vulkan 14.5.4 "Offset
      // and Stride Assignment".
      const auto storageClass = words[3];
      const bool uniform = storageClass == SpvStorageClassUniform;
      const bool uniform_constant =
          storageClass == SpvStorageClassUniformConstant;
      const bool push_constant = storageClass == SpvStorageClassPushConstant;
      const bool storage_buffer = storageClass == SpvStorageClassStorageBuffer;

      if (spvIsVulkanEnv(vstate.context()->target_env)) {
        // Vulkan 14.5.1: There must be no more than one PushConstant block
        // per entry point.
        if (push_constant) {
          auto entry_points = vstate.EntryPointReferences(var_id);
          for (auto ep_id : entry_points) {
            const bool already_used = !uses_push_constant.insert(ep_id).second;
            if (already_used) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                     << "Entry point id '" << ep_id
                     << "' uses more than one PushConstant interface.\n"
                     << "From Vulkan spec, section 14.5.1:\n"
                     << "There must be no more than one push constant block "
                     << "statically used per shader entry point.";
            }
          }
        }
        // Vulkan 14.5.2: Check DescriptorSet and Binding decoration for
        // UniformConstant which cannot be a struct.
        if (uniform_constant) {
          auto entry_points = vstate.EntryPointReferences(var_id);
          if (!entry_points.empty() &&
              !hasDecoration(var_id, SpvDecorationDescriptorSet, vstate)) {
            return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                   << "UniformConstant id '" << var_id
                   << "' is missing DescriptorSet decoration.\n"
                   << "From Vulkan spec, section 14.5.2:\n"
                   << "These variables must have DescriptorSet and Binding "
                      "decorations specified";
          }
          if (!entry_points.empty() &&
              !hasDecoration(var_id, SpvDecorationBinding, vstate)) {
            return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                   << "UniformConstant id '" << var_id
                   << "' is missing Binding decoration.\n"
                   << "From Vulkan spec, section 14.5.2:\n"
                   << "These variables must have DescriptorSet and Binding "
                      "decorations specified";
          }
        }
      }

      const bool phys_storage_buffer =
          storageClass == SpvStorageClassPhysicalStorageBufferEXT;
      if (uniform || push_constant || storage_buffer || phys_storage_buffer) {
        const auto ptrInst = vstate.FindDef(words[1]);
        assert(SpvOpTypePointer == ptrInst->opcode());
        auto id = ptrInst->words()[3];
        auto id_inst = vstate.FindDef(id);
        // Jump through one level of arraying.
        if (id_inst->opcode() == SpvOpTypeArray ||
            id_inst->opcode() == SpvOpTypeRuntimeArray) {
          id = id_inst->GetOperandAs<uint32_t>(1u);
          id_inst = vstate.FindDef(id);
        }
        if (SpvOpTypeStruct != id_inst->opcode()) continue;
        MemberConstraints constraints;
        ComputeMemberConstraintsForStruct(&constraints, id, LayoutConstraints(),
                                          vstate);
        // Prepare for messages
        const char* sc_str =
            uniform ? "Uniform"
                    : (push_constant ? "PushConstant" : "StorageBuffer");

        if (spvIsVulkanEnv(vstate.context()->target_env)) {
          if (storage_buffer &&
              hasDecoration(id, SpvDecorationBufferBlock, vstate)) {
            return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                   << "Storage buffer id '" << var_id
                   << " In Vulkan, BufferBlock is disallowed on variables in "
                      "the StorageBuffer storage class";
          }
          // Vulkan 14.5.1: Check Block decoration for PushConstant variables.
          if (push_constant && !hasDecoration(id, SpvDecorationBlock, vstate)) {
            return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                   << "PushConstant id '" << id
                   << "' is missing Block decoration.\n"
                   << "From Vulkan spec, section 14.5.1:\n"
                   << "Such variables must be identified with a Block "
                      "decoration";
          }
          // Vulkan 14.5.2: Check DescriptorSet and Binding decoration for
          // Uniform and StorageBuffer variables.
          if (uniform || storage_buffer) {
            auto entry_points = vstate.EntryPointReferences(var_id);
            if (!entry_points.empty() &&
                !hasDecoration(var_id, SpvDecorationDescriptorSet, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                     << sc_str << " id '" << var_id
                     << "' is missing DescriptorSet decoration.\n"
                     << "From Vulkan spec, section 14.5.2:\n"
                     << "These variables must have DescriptorSet and Binding "
                        "decorations specified";
            }
            if (!entry_points.empty() &&
                !hasDecoration(var_id, SpvDecorationBinding, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(var_id))
                     << sc_str << " id '" << var_id
                     << "' is missing Binding decoration.\n"
                     << "From Vulkan spec, section 14.5.2:\n"
                     << "These variables must have DescriptorSet and Binding "
                        "decorations specified";
            }
          }
        }

        for (const auto& dec : vstate.id_decorations(id)) {
          const bool blockDeco = SpvDecorationBlock == dec.dec_type();
          const bool bufferDeco = SpvDecorationBufferBlock == dec.dec_type();
          const bool blockRules = uniform && blockDeco;
          const bool bufferRules =
              (uniform && bufferDeco) || (push_constant && blockDeco) ||
              ((storage_buffer || phys_storage_buffer) && blockDeco);
          if (uniform && blockDeco) {
            vstate.RegisterPointerToUniformBlock(ptrInst->id());
            vstate.RegisterStructForUniformBlock(id);
          }
          if ((uniform && bufferDeco) || (storage_buffer && blockDeco)) {
            vstate.RegisterPointerToStorageBuffer(ptrInst->id());
            vstate.RegisterStructForStorageBuffer(id);
          }
          if (blockRules || bufferRules) {
            const char* deco_str = blockDeco ? "Block" : "BufferBlock";
            spv_result_t recursive_status = SPV_SUCCESS;
            if (isMissingOffsetInStruct(id, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                     << "Structure id " << id << " decorated as " << deco_str
                     << " must be explicitly laid out with Offset "
                        "decorations.";
            } else if (hasDecoration(id, SpvDecorationGLSLShared, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                     << "Structure id " << id << " decorated as " << deco_str
                     << " must not use GLSLShared decoration.";
            } else if (hasDecoration(id, SpvDecorationGLSLPacked, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                     << "Structure id " << id << " decorated as " << deco_str
                     << " must not use GLSLPacked decoration.";
            } else if (!checkForRequiredDecoration(id, SpvDecorationArrayStride,
                                                   SpvOpTypeArray, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                     << "Structure id " << id << " decorated as " << deco_str
                     << " must be explicitly laid out with ArrayStride "
                        "decorations.";
            } else if (!checkForRequiredDecoration(id,
                                                   SpvDecorationMatrixStride,
                                                   SpvOpTypeMatrix, vstate)) {
              return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                     << "Structure id " << id << " decorated as " << deco_str
                     << " must be explicitly laid out with MatrixStride "
                        "decorations.";
            } else if (blockRules &&
                       (SPV_SUCCESS != (recursive_status = checkLayout(
                                            id, sc_str, deco_str, true, 0,
                                            constraints, vstate)))) {
              return recursive_status;
            } else if (bufferRules &&
                       (SPV_SUCCESS != (recursive_status = checkLayout(
                                            id, sc_str, deco_str, false, 0,
                                            constraints, vstate)))) {
              return recursive_status;
            }
          }
        }
      }
    }
  }
  return SPV_SUCCESS;
}

// Returns true if |decoration| cannot be applied to the same id more than once.
bool AtMostOncePerId(SpvDecoration decoration) {
  return decoration == SpvDecorationArrayStride;
}

// Returns true if |decoration| cannot be applied to the same member more than
// once.
bool AtMostOncePerMember(SpvDecoration decoration) {
  switch (decoration) {
    case SpvDecorationOffset:
    case SpvDecorationMatrixStride:
    case SpvDecorationRowMajor:
    case SpvDecorationColMajor:
      return true;
    default:
      return false;
  }
}

// Returns the string name for |decoration|.
const char* GetDecorationName(SpvDecoration decoration) {
  switch (decoration) {
    case SpvDecorationAliased:
      return "Aliased";
    case SpvDecorationRestrict:
      return "Restrict";
    case SpvDecorationArrayStride:
      return "ArrayStride";
    case SpvDecorationOffset:
      return "Offset";
    case SpvDecorationMatrixStride:
      return "MatrixStride";
    case SpvDecorationRowMajor:
      return "RowMajor";
    case SpvDecorationColMajor:
      return "ColMajor";
    case SpvDecorationBlock:
      return "Block";
    case SpvDecorationBufferBlock:
      return "BufferBlock";
    default:
      return "";
  }
}

spv_result_t CheckDecorationsCompatibility(ValidationState_t& vstate) {
  using PerIDKey = std::tuple<SpvDecoration, uint32_t>;
  using PerMemberKey = std::tuple<SpvDecoration, uint32_t, uint32_t>;

  // An Array of pairs where the decorations in the pair cannot both be applied
  // to the same id.
  static const SpvDecoration mutually_exclusive_per_id[][2] = {
      {SpvDecorationBlock, SpvDecorationBufferBlock},
      {SpvDecorationRestrict, SpvDecorationAliased}};
  static const auto num_mutually_exclusive_per_id_pairs =
      sizeof(mutually_exclusive_per_id) / (2 * sizeof(SpvDecoration));

  // An Array of pairs where the decorations in the pair cannot both be applied
  // to the same member.
  static const SpvDecoration mutually_exclusive_per_member[][2] = {
      {SpvDecorationRowMajor, SpvDecorationColMajor}};
  static const auto num_mutually_exclusive_per_mem_pairs =
      sizeof(mutually_exclusive_per_member) / (2 * sizeof(SpvDecoration));

  std::set<PerIDKey> seen_per_id;
  std::set<PerMemberKey> seen_per_member;

  for (const auto& inst : vstate.ordered_instructions()) {
    const auto& words = inst.words();
    if (SpvOpDecorate == inst.opcode()) {
      const auto id = words[1];
      const auto dec_type = static_cast<SpvDecoration>(words[2]);
      const auto k = PerIDKey(dec_type, id);
      const auto already_used = !seen_per_id.insert(k).second;
      if (already_used && AtMostOncePerId(dec_type)) {
        return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
               << "ID '" << id << "' decorated with "
               << GetDecorationName(dec_type)
               << " multiple times is not allowed.";
      }
      // Verify certain mutually exclusive decorations are not both applied on
      // an ID.
      for (uint32_t pair_idx = 0;
           pair_idx < num_mutually_exclusive_per_id_pairs; ++pair_idx) {
        SpvDecoration excl_dec_type = SpvDecorationMax;
        if (mutually_exclusive_per_id[pair_idx][0] == dec_type) {
          excl_dec_type = mutually_exclusive_per_id[pair_idx][1];
        } else if (mutually_exclusive_per_id[pair_idx][1] == dec_type) {
          excl_dec_type = mutually_exclusive_per_id[pair_idx][0];
        } else {
          continue;
        }

        const auto excl_k = PerIDKey(excl_dec_type, id);
        if (seen_per_id.find(excl_k) != seen_per_id.end()) {
          return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                 << "ID '" << id << "' decorated with both "
                 << GetDecorationName(dec_type) << " and "
                 << GetDecorationName(excl_dec_type) << " is not allowed.";
        }
      }
    } else if (SpvOpMemberDecorate == inst.opcode()) {
      const auto id = words[1];
      const auto member_id = words[2];
      const auto dec_type = static_cast<SpvDecoration>(words[3]);
      const auto k = PerMemberKey(dec_type, id, member_id);
      const auto already_used = !seen_per_member.insert(k).second;
      if (already_used && AtMostOncePerMember(dec_type)) {
        return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
               << "ID '" << id << "', member '" << member_id
               << "' decorated with " << GetDecorationName(dec_type)
               << " multiple times is not allowed.";
      }
      // Verify certain mutually exclusive decorations are not both applied on
      // a (ID, member) tuple.
      for (uint32_t pair_idx = 0;
           pair_idx < num_mutually_exclusive_per_mem_pairs; ++pair_idx) {
        SpvDecoration excl_dec_type = SpvDecorationMax;
        if (mutually_exclusive_per_member[pair_idx][0] == dec_type) {
          excl_dec_type = mutually_exclusive_per_member[pair_idx][1];
        } else if (mutually_exclusive_per_member[pair_idx][1] == dec_type) {
          excl_dec_type = mutually_exclusive_per_member[pair_idx][0];
        } else {
          continue;
        }

        const auto excl_k = PerMemberKey(excl_dec_type, id, member_id);
        if (seen_per_member.find(excl_k) != seen_per_member.end()) {
          return vstate.diag(SPV_ERROR_INVALID_ID, vstate.FindDef(id))
                 << "ID '" << id << "', member '" << member_id
                 << "' decorated with both " << GetDecorationName(dec_type)
                 << " and " << GetDecorationName(excl_dec_type)
                 << " is not allowed.";
        }
      }
    }
  }
  return SPV_SUCCESS;
}

spv_result_t CheckVulkanMemoryModelDeprecatedDecorations(
    ValidationState_t& vstate) {
  if (vstate.memory_model() != SpvMemoryModelVulkanKHR) return SPV_SUCCESS;

  std::string msg;
  std::ostringstream str(msg);
  for (const auto& def : vstate.all_definitions()) {
    const auto inst = def.second;
    const auto id = inst->id();
    for (const auto& dec : vstate.id_decorations(id)) {
      const auto member = dec.struct_member_index();
      if (dec.dec_type() == SpvDecorationCoherent ||
          dec.dec_type() == SpvDecorationVolatile) {
        str << (dec.dec_type() == SpvDecorationCoherent ? "Coherent"
                                                        : "Volatile");
        str << " decoration targeting " << vstate.getIdName(id);
        if (member != Decoration::kInvalidMember) {
          str << " (member index " << member << ")";
        }
        str << " is banned when using the Vulkan memory model.";
        return vstate.diag(SPV_ERROR_INVALID_ID, inst) << str.str();
      }
    }
  }
  return SPV_SUCCESS;
}

// Returns SPV_SUCCESS if validation rules are satisfied for FPRoundingMode
// decorations.  Otherwise emits a diagnostic and returns something other than
// SPV_SUCCESS.
spv_result_t CheckFPRoundingModeForShaders(ValidationState_t& vstate,
                                           const Instruction& inst) {
  // Validates width-only conversion instruction for floating-point object
  // i.e., OpFConvert
  if (inst.opcode() != SpvOpFConvert) {
    return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
           << "FPRoundingMode decoration can be applied only to a "
              "width-only conversion instruction for floating-point "
              "object.";
  }

  // Validates Object operand of an OpStore
  for (const auto& use : inst.uses()) {
    const auto store = use.first;
    if (store->opcode() == SpvOpFConvert) continue;
    if (spvOpcodeIsDebug(store->opcode())) continue;
    if (spvOpcodeIsDecoration(store->opcode())) continue;
    if (store->opcode() != SpvOpStore) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "FPRoundingMode decoration can be applied only to the "
                "Object operand of an OpStore.";
    }

    if (use.second != 2) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "FPRoundingMode decoration can be applied only to the "
                "Object operand of an OpStore.";
    }

    const auto ptr_inst = vstate.FindDef(store->GetOperandAs<uint32_t>(0));
    const auto ptr_type = vstate.FindDef(ptr_inst->GetOperandAs<uint32_t>(0));

    const auto half_float_id = ptr_type->GetOperandAs<uint32_t>(2);
    if (!vstate.IsFloatScalarOrVectorType(half_float_id) ||
        vstate.GetBitWidth(half_float_id) != 16) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "FPRoundingMode decoration can be applied only to the "
                "Object operand of an OpStore storing through a pointer "
                "to "
                "a 16-bit floating-point scalar or vector object.";
    }

    // Validates storage class of the pointer to the OpStore
    const auto storage = ptr_type->GetOperandAs<uint32_t>(1);
    if (storage != SpvStorageClassStorageBuffer &&
        storage != SpvStorageClassUniform &&
        storage != SpvStorageClassPushConstant &&
        storage != SpvStorageClassInput && storage != SpvStorageClassOutput &&
        storage != SpvStorageClassPhysicalStorageBufferEXT) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "FPRoundingMode decoration can be applied only to the "
                "Object operand of an OpStore in the StorageBuffer, "
                "PhysicalStorageBufferEXT, Uniform, PushConstant, Input, or "
                "Output Storage Classes.";
    }
  }
  return SPV_SUCCESS;
}

// Returns SPV_SUCCESS if validation rules are satisfied for the NonWritable
// decoration.  Otherwise emits a diagnostic and returns something other than
// SPV_SUCCESS.  The |inst| parameter is the object being decorated.  This must
// be called after TypePass and AnnotateCheckDecorationsOfBuffers are called.
spv_result_t CheckNonWritableDecoration(ValidationState_t& vstate,
                                        const Instruction& inst,
                                        const Decoration& decoration) {
  assert(inst.id() && "Parser ensures the target of the decoration has an ID");

  if (decoration.struct_member_index() == Decoration::kInvalidMember) {
    // The target must be a memory object declaration.
    // First, it must be a variable or function parameter.
    if (inst.opcode() != SpvOpVariable &&
        inst.opcode() != SpvOpFunctionParameter) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "Target of NonWritable decoration must be a memory object "
                "declaration (a variable or a function parameter)";
    }
    // Second, it must point to a UBO, SSBO, or storage image.
    const auto type_id = inst.type_id();
    if (!vstate.IsPointerToUniformBlock(type_id) &&
        !vstate.IsPointerToStorageBuffer(type_id) &&
        !vstate.IsPointerToStorageImage(type_id)) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "Target of NonWritable decoration is invalid: must point to a "
                "storage image, uniform block, or storage buffer";
    }
  }

  return SPV_SUCCESS;
}

// Returns SPV_SUCCESS if validation rules are satisfied for Uniform
// decorations. Otherwise emits a diagnostic and returns something other than
// SPV_SUCCESS. Assumes each decoration on a group has been propagated down to
// the group members.
spv_result_t CheckUniformDecoration(ValidationState_t& vstate,
                                    const Instruction& inst,
                                    const Decoration&) {
  // Uniform must decorate an "object"
  //  - has a result ID
  //  - is an instantiation of a non-void type.  So it has a type ID, and that
  //  type is not void.

  // We already know the result ID is non-zero.

  if (inst.type_id() == 0) {
    return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
           << "Uniform decoration applied to a non-object";
  }
  if (Instruction* type_inst = vstate.FindDef(inst.type_id())) {
    if (type_inst->opcode() == SpvOpTypeVoid) {
      return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
             << "Uniform decoration applied to a value with void type";
    }
  } else {
    // We might never get here because this would have been rejected earlier in
    // the flow.
    return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
           << "Uniform decoration applied to an object with invalid type";
  }
  return SPV_SUCCESS;
}

// Returns SPV_SUCCESS if validation rules are satisfied for NoSignedWrap or
// NoUnsignedWrap decorations. Otherwise emits a diagnostic and returns
// something other than SPV_SUCCESS. Assumes each decoration on a group has been
// propagated down to the group members.
spv_result_t CheckIntegerWrapDecoration(ValidationState_t& vstate,
                                        const Instruction& inst,
                                        const Decoration& decoration) {
  switch (inst.opcode()) {
    case SpvOpIAdd:
    case SpvOpISub:
    case SpvOpIMul:
    case SpvOpShiftLeftLogical:
    case SpvOpSNegate:
      return SPV_SUCCESS;
    case SpvOpExtInst:
      // TODO(dneto): Only certain extended instructions allow these
      // decorations.  For now allow anything.
      return SPV_SUCCESS;
    default:
      break;
  }

  return vstate.diag(SPV_ERROR_INVALID_ID, &inst)
         << (decoration.dec_type() == SpvDecorationNoSignedWrap
                 ? "NoSignedWrap"
                 : "NoUnsignedWrap")
         << " decoration may not be applied to "
         << spvOpcodeString(inst.opcode());
}

#define PASS_OR_BAIL_AT_LINE(X, LINE)           \
  {                                             \
    spv_result_t e##LINE = (X);                 \
    if (e##LINE != SPV_SUCCESS) return e##LINE; \
  }
#define PASS_OR_BAIL(X) PASS_OR_BAIL_AT_LINE(X, __LINE__)

// Check rules for decorations where we start from the decoration rather
// than the decorated object.  Assumes each decoration on a group have been
// propagated down to the group members.
spv_result_t CheckDecorationsFromDecoration(ValidationState_t& vstate) {
  // Some rules are only checked for shaders.
  const bool is_shader = vstate.HasCapability(SpvCapabilityShader);

  for (const auto& kv : vstate.id_decorations()) {
    const uint32_t id = kv.first;
    const auto& decorations = kv.second;
    if (decorations.empty()) continue;

    const Instruction* inst = vstate.FindDef(id);
    assert(inst);

    // We assume the decorations applied to a decoration group have already
    // been propagated down to the group members.
    if (inst->opcode() == SpvOpDecorationGroup) continue;

    // Validates FPRoundingMode decoration
    for (const auto& decoration : decorations) {
      switch (decoration.dec_type()) {
        case SpvDecorationFPRoundingMode:
          if (is_shader)
            PASS_OR_BAIL(CheckFPRoundingModeForShaders(vstate, *inst));
          break;
        case SpvDecorationNonWritable:
          PASS_OR_BAIL(CheckNonWritableDecoration(vstate, *inst, decoration));
          break;
        case SpvDecorationUniform:
          PASS_OR_BAIL(CheckUniformDecoration(vstate, *inst, decoration));
          break;
        case SpvDecorationNoSignedWrap:
        case SpvDecorationNoUnsignedWrap:
          PASS_OR_BAIL(CheckIntegerWrapDecoration(vstate, *inst, decoration));
          break;
        default:
          break;
      }
    }
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ValidateDecorations(ValidationState_t& vstate) {
  if (auto error = CheckImportedVariableInitialization(vstate)) return error;
  if (auto error = CheckDecorationsOfEntryPoints(vstate)) return error;
  if (auto error = CheckDecorationsOfBuffers(vstate)) return error;
  if (auto error = CheckDecorationsCompatibility(vstate)) return error;
  if (auto error = CheckLinkageAttrOfFunctions(vstate)) return error;
  if (auto error = CheckVulkanMemoryModelDeprecatedDecorations(vstate))
    return error;
  if (auto error = CheckDecorationsFromDecoration(vstate)) return error;
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
