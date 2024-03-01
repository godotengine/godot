// Copyright (c) 2019 Google LLC
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

// This pass injects code in a graphics shader to implement guarantees
// satisfying Vulkan's robustBufferAccess rules.  Robust access rules permit
// an out-of-bounds access to be redirected to an access of the same type
// (load, store, etc.) but within the same root object.
//
// We assume baseline functionality in Vulkan, i.e. the module uses
// logical addressing mode, without VK_KHR_variable_pointers.
//
//    - Logical addressing mode implies:
//      - Each root pointer (a pointer that exists other than by the
//        execution of a shader instruction) is the result of an OpVariable.
//
//      - Instructions that result in pointers are:
//          OpVariable
//          OpAccessChain
//          OpInBoundsAccessChain
//          OpFunctionParameter
//          OpImageTexelPointer
//          OpCopyObject
//
//      - Instructions that use a pointer are:
//          OpLoad
//          OpStore
//          OpAccessChain
//          OpInBoundsAccessChain
//          OpFunctionCall
//          OpImageTexelPointer
//          OpCopyMemory
//          OpCopyObject
//          all OpAtomic* instructions
//
// We classify pointer-users into:
//  - Accesses:
//    - OpLoad
//    - OpStore
//    - OpAtomic*
//    - OpCopyMemory
//
//  - Address calculations:
//    - OpAccessChain
//    - OpInBoundsAccessChain
//
//  - Pass-through:
//    - OpFunctionCall
//    - OpFunctionParameter
//    - OpCopyObject
//
// The strategy is:
//
//  - Handle only logical addressing mode. In particular, don't handle a module
//    if it uses one of the variable-pointers capabilities.
//
//  - Don't handle modules using capability RuntimeDescriptorArrayEXT.  So the
//    only runtime arrays are those that are the last member in a
//    Block-decorated struct.  This allows us to feasibly/easily compute the
//    length of the runtime array. See below.
//
//  - The memory locations accessed by OpLoad, OpStore, OpCopyMemory, and
//    OpAtomic* are determined by their pointer parameter or parameters.
//    Pointers are always (correctly) typed and so the address and number of
//    consecutive locations are fully determined by the pointer.
//
//  - A pointer value originates as one of few cases:
//
//    - OpVariable for an interface object or an array of them: image,
//      buffer (UBO or SSBO), sampler, sampled-image, push-constant, input
//      variable, output variable. The execution environment is responsible for
//      allocating the correct amount of storage for these, and for ensuring
//      each resource bound to such a variable is big enough to contain the
//      SPIR-V pointee type of the variable.
//
//    - OpVariable for a non-interface object.  These are variables in
//      Workgroup, Private, and Function storage classes.  The compiler ensures
//      the underlying allocation is big enough to store the entire SPIR-V
//      pointee type of the variable.
//
//    - An OpFunctionParameter. This always maps to a pointer parameter to an
//      OpFunctionCall.
//
//      - In logical addressing mode, these are severely limited:
//        "Any pointer operand to an OpFunctionCall must be:
//          - a memory object declaration, or
//          - a pointer to an element in an array that is a memory object
//          declaration, where the element type is OpTypeSampler or OpTypeImage"
//
//      - This has an important simplifying consequence:
//
//        - When looking for a pointer to the structure containing a runtime
//          array, you begin with a pointer to the runtime array and trace
//          backward in the function.  You never have to trace back beyond
//          your function call boundary.  So you can't take a partial access
//          chain into an SSBO, then pass that pointer into a function.  So
//          we don't resort to using fat pointers to compute array length.
//          We can trace back to a pointer to the containing structure,
//          and use that in an OpArrayLength instruction. (The structure type
//          gives us the member index of the runtime array.)
//
//        - Otherwise, the pointer type fully encodes the range of valid
//          addresses. In particular, the type of a pointer to an aggregate
//          value fully encodes the range of indices when indexing into
//          that aggregate.
//
//    - The pointer is the result of an access chain instruction.  We clamp
//      indices contributing to address calculations.  As noted above, the
//      valid ranges are either bound by the length of a runtime array, or
//      by the type of the base pointer.  The length of a runtime array is
//      the result of an OpArrayLength instruction acting on the pointer of
//      the containing structure as noted above.
//
//      - Access chain indices are always treated as signed, so:
//        - Clamp the upper bound at the signed integer maximum.
//        - Use SClamp for all clamping.
//
//    - TODO(dneto): OpImageTexelPointer:
//      - Clamp coordinate to the image size returned by OpImageQuerySize
//      - If multi-sampled, clamp the sample index to the count returned by
//        OpImageQuerySamples.
//      - If not multi-sampled, set the sample index to 0.
//
//  - Rely on the external validator to check that pointers are only
//    used by the instructions as above.
//
//  - Handles OpTypeRuntimeArray
//    Track pointer back to original resource (pointer to struct), so we can
//    query the runtime array size.
//

#include "graphics_robust_access_pass.h"

#include <functional>
#include <initializer_list>
#include <utility>

#include "function.h"
#include "ir_context.h"
#include "pass.h"
#include "source/diagnostic.h"
#include "source/util/make_unique.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/GLSL.std.450.h"
#include "type_manager.h"
#include "types.h"

namespace spvtools {
namespace opt {

using opt::Instruction;
using opt::Operand;
using spvtools::MakeUnique;

GraphicsRobustAccessPass::GraphicsRobustAccessPass() : module_status_() {}

Pass::Status GraphicsRobustAccessPass::Process() {
  module_status_ = PerModuleState();

  ProcessCurrentModule();

  auto result = module_status_.failed
                    ? Status::Failure
                    : (module_status_.modified ? Status::SuccessWithChange
                                               : Status::SuccessWithoutChange);

  return result;
}

spvtools::DiagnosticStream GraphicsRobustAccessPass::Fail() {
  module_status_.failed = true;
  // We don't really have a position, and we'll ignore the result.
  return std::move(
      spvtools::DiagnosticStream({}, consumer(), "", SPV_ERROR_INVALID_BINARY)
      << name() << ": ");
}

spv_result_t GraphicsRobustAccessPass::IsCompatibleModule() {
  auto* feature_mgr = context()->get_feature_mgr();
  if (!feature_mgr->HasCapability(spv::Capability::Shader))
    return Fail() << "Can only process Shader modules";
  if (feature_mgr->HasCapability(spv::Capability::VariablePointers))
    return Fail() << "Can't process modules with VariablePointers capability";
  if (feature_mgr->HasCapability(
          spv::Capability::VariablePointersStorageBuffer))
    return Fail() << "Can't process modules with VariablePointersStorageBuffer "
                     "capability";
  if (feature_mgr->HasCapability(spv::Capability::RuntimeDescriptorArrayEXT)) {
    // These have a RuntimeArray outside of Block-decorated struct.  There
    // is no way to compute the array length from within SPIR-V.
    return Fail() << "Can't process modules with RuntimeDescriptorArrayEXT "
                     "capability";
  }

  {
    auto* inst = context()->module()->GetMemoryModel();
    const auto addressing_model =
        spv::AddressingModel(inst->GetSingleWordOperand(0));
    if (addressing_model != spv::AddressingModel::Logical)
      return Fail() << "Addressing model must be Logical.  Found "
                    << inst->PrettyPrint();
  }
  return SPV_SUCCESS;
}

spv_result_t GraphicsRobustAccessPass::ProcessCurrentModule() {
  auto err = IsCompatibleModule();
  if (err != SPV_SUCCESS) return err;

  ProcessFunction fn = [this](opt::Function* f) { return ProcessAFunction(f); };
  module_status_.modified |= context()->ProcessReachableCallTree(fn);

  // Need something here.  It's the price we pay for easier failure paths.
  return SPV_SUCCESS;
}

bool GraphicsRobustAccessPass::ProcessAFunction(opt::Function* function) {
  // Ensure that all pointers computed inside a function are within bounds.
  // Find the access chains in this block before trying to modify them.
  std::vector<Instruction*> access_chains;
  std::vector<Instruction*> image_texel_pointers;
  for (auto& block : *function) {
    for (auto& inst : block) {
      switch (inst.opcode()) {
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
          access_chains.push_back(&inst);
          break;
        case spv::Op::OpImageTexelPointer:
          image_texel_pointers.push_back(&inst);
          break;
        default:
          break;
      }
    }
  }
  for (auto* inst : access_chains) {
    ClampIndicesForAccessChain(inst);
    if (module_status_.failed) return module_status_.modified;
  }

  for (auto* inst : image_texel_pointers) {
    if (SPV_SUCCESS != ClampCoordinateForImageTexelPointer(inst)) break;
  }
  return module_status_.modified;
}

void GraphicsRobustAccessPass::ClampIndicesForAccessChain(
    Instruction* access_chain) {
  Instruction& inst = *access_chain;

  auto* constant_mgr = context()->get_constant_mgr();
  auto* def_use_mgr = context()->get_def_use_mgr();
  auto* type_mgr = context()->get_type_mgr();
  const bool have_int64_cap =
      context()->get_feature_mgr()->HasCapability(spv::Capability::Int64);

  // Replaces one of the OpAccessChain index operands with a new value.
  // Updates def-use analysis.
  auto replace_index = [this, &inst, def_use_mgr](uint32_t operand_index,
                                                  Instruction* new_value) {
    inst.SetOperand(operand_index, {new_value->result_id()});
    def_use_mgr->AnalyzeInstUse(&inst);
    module_status_.modified = true;
    return SPV_SUCCESS;
  };

  // Replaces one of the OpAccesssChain index operands with a clamped value.
  // Replace the operand at |operand_index| with the value computed from
  // signed_clamp(%old_value, %min_value, %max_value).  It also analyzes
  // the new instruction and records that them module is modified.
  // Assumes %min_value is signed-less-or-equal than %max_value. (All callees
  // use 0 for %min_value).
  auto clamp_index = [&inst, type_mgr, this, &replace_index](
                         uint32_t operand_index, Instruction* old_value,
                         Instruction* min_value, Instruction* max_value) {
    auto* clamp_inst =
        MakeSClampInst(*type_mgr, old_value, min_value, max_value, &inst);
    return replace_index(operand_index, clamp_inst);
  };

  // Ensures the specified index of access chain |inst| has a value that is
  // at most |count| - 1.  If the index is already a constant value less than
  // |count| then no change is made.
  auto clamp_to_literal_count =
      [&inst, this, &constant_mgr, &type_mgr, have_int64_cap, &replace_index,
       &clamp_index](uint32_t operand_index, uint64_t count) -> spv_result_t {
    Instruction* index_inst =
        this->GetDef(inst.GetSingleWordOperand(operand_index));
    const auto* index_type =
        type_mgr->GetType(index_inst->type_id())->AsInteger();
    assert(index_type);
    const auto index_width = index_type->width();

    if (count <= 1) {
      // Replace the index with 0.
      return replace_index(operand_index, GetValueForType(0, index_type));
    }

    uint64_t maxval = count - 1;

    // Compute the bit width of a viable type to hold |maxval|.
    // Look for a bit width, up to 64 bits wide, to fit maxval.
    uint32_t maxval_width = index_width;
    while ((maxval_width < 64) && (0 != (maxval >> maxval_width))) {
      maxval_width *= 2;
    }
    // Determine the type for |maxval|.
    uint32_t next_id = context()->module()->IdBound();
    analysis::Integer signed_type_for_query(maxval_width, true);
    auto* maxval_type =
        type_mgr->GetRegisteredType(&signed_type_for_query)->AsInteger();
    if (next_id != context()->module()->IdBound()) {
      module_status_.modified = true;
    }
    // Access chain indices are treated as signed, so limit the maximum value
    // of the index so it will always be positive for a signed clamp operation.
    maxval = std::min(maxval, ((uint64_t(1) << (maxval_width - 1)) - 1));

    if (index_width > 64) {
      return this->Fail() << "Can't handle indices wider than 64 bits, found "
                             "constant index with "
                          << index_width << " bits as index number "
                          << operand_index << " of access chain "
                          << inst.PrettyPrint();
    }

    // Split into two cases: the current index is a constant, or not.

    // If the index is a constant then |index_constant| will not be a null
    // pointer.  (If index is an |OpConstantNull| then it |index_constant| will
    // not be a null pointer.)  Since access chain indices must be scalar
    // integers, this can't be a spec constant.
    if (auto* index_constant = constant_mgr->GetConstantFromInst(index_inst)) {
      auto* int_index_constant = index_constant->AsIntConstant();
      int64_t value = 0;
      // OpAccessChain indices are treated as signed.  So get the signed
      // constant value here.
      if (index_width <= 32) {
        value = int64_t(int_index_constant->GetS32BitValue());
      } else if (index_width <= 64) {
        value = int_index_constant->GetS64BitValue();
      }
      if (value < 0) {
        return replace_index(operand_index, GetValueForType(0, index_type));
      } else if (uint64_t(value) <= maxval) {
        // Nothing to do.
        return SPV_SUCCESS;
      } else {
        // Replace with maxval.
        assert(count > 0);  // Already took care of this case above.
        return replace_index(operand_index,
                             GetValueForType(maxval, maxval_type));
      }
    } else {
      // Generate a clamp instruction.
      assert(maxval >= 1);
      assert(index_width <= 64);  // Otherwise, already returned above.
      if (index_width >= 64 && !have_int64_cap) {
        // An inconsistent module.
        return Fail() << "Access chain index is wider than 64 bits, but Int64 "
                         "is not declared: "
                      << index_inst->PrettyPrint();
      }
      // Widen the index value if necessary
      if (maxval_width > index_width) {
        // Find the wider type.  We only need this case if a constant array
        // bound is too big.

        // From how we calculated maxval_width, widening won't require adding
        // the Int64 capability.
        assert(have_int64_cap || maxval_width <= 32);
        if (!have_int64_cap && maxval_width >= 64) {
          // Be defensive, but this shouldn't happen.
          return this->Fail()
                 << "Clamping index would require adding Int64 capability. "
                 << "Can't clamp 32-bit index " << operand_index
                 << " of access chain " << inst.PrettyPrint();
        }
        index_inst = WidenInteger(index_type->IsSigned(), maxval_width,
                                  index_inst, &inst);
      }

      // Finally, clamp the index.
      return clamp_index(operand_index, index_inst,
                         GetValueForType(0, maxval_type),
                         GetValueForType(maxval, maxval_type));
    }
    return SPV_SUCCESS;
  };

  // Ensures the specified index of access chain |inst| has a value that is at
  // most the value of |count_inst| minus 1, where |count_inst| is treated as an
  // unsigned integer. This can log a failure.
  auto clamp_to_count = [&inst, this, &constant_mgr, &clamp_to_literal_count,
                         &clamp_index,
                         &type_mgr](uint32_t operand_index,
                                    Instruction* count_inst) -> spv_result_t {
    Instruction* index_inst =
        this->GetDef(inst.GetSingleWordOperand(operand_index));
    const auto* index_type =
        type_mgr->GetType(index_inst->type_id())->AsInteger();
    const auto* count_type =
        type_mgr->GetType(count_inst->type_id())->AsInteger();
    assert(index_type);
    if (const auto* count_constant =
            constant_mgr->GetConstantFromInst(count_inst)) {
      uint64_t value = 0;
      const auto width = count_constant->type()->AsInteger()->width();
      if (width <= 32) {
        value = count_constant->AsIntConstant()->GetU32BitValue();
      } else if (width <= 64) {
        value = count_constant->AsIntConstant()->GetU64BitValue();
      } else {
        return this->Fail() << "Can't handle indices wider than 64 bits, found "
                               "constant index with "
                            << index_type->width() << "bits";
      }
      return clamp_to_literal_count(operand_index, value);
    } else {
      // Widen them to the same width.
      const auto index_width = index_type->width();
      const auto count_width = count_type->width();
      const auto target_width = std::max(index_width, count_width);
      // UConvert requires the result type to have 0 signedness.  So enforce
      // that here.
      auto* wider_type = index_width < count_width ? count_type : index_type;
      if (index_type->width() < target_width) {
        // Access chain indices are treated as signed integers.
        index_inst = WidenInteger(true, target_width, index_inst, &inst);
      } else if (count_type->width() < target_width) {
        // Assume type sizes are treated as unsigned.
        count_inst = WidenInteger(false, target_width, count_inst, &inst);
      }
      // Compute count - 1.
      // It doesn't matter if 1 is signed or unsigned.
      auto* one = GetValueForType(1, wider_type);
      auto* count_minus_1 = InsertInst(
          &inst, spv::Op::OpISub, type_mgr->GetId(wider_type), TakeNextId(),
          {{SPV_OPERAND_TYPE_ID, {count_inst->result_id()}},
           {SPV_OPERAND_TYPE_ID, {one->result_id()}}});
      auto* zero = GetValueForType(0, wider_type);
      // Make sure we clamp to an upper bound that is at most the signed max
      // for the target type.
      const uint64_t max_signed_value =
          ((uint64_t(1) << (target_width - 1)) - 1);
      // Use unsigned-min to ensure that the result is always non-negative.
      // That ensures we satisfy the invariant for SClamp, where the "min"
      // argument we give it (zero), is no larger than the third argument.
      auto* upper_bound =
          MakeUMinInst(*type_mgr, count_minus_1,
                       GetValueForType(max_signed_value, wider_type), &inst);
      // Now clamp the index to this upper bound.
      return clamp_index(operand_index, index_inst, zero, upper_bound);
    }
    return SPV_SUCCESS;
  };

  const Instruction* base_inst = GetDef(inst.GetSingleWordInOperand(0));
  const Instruction* base_type = GetDef(base_inst->type_id());
  Instruction* pointee_type = GetDef(base_type->GetSingleWordInOperand(1));

  // Walk the indices from earliest to latest, replacing indices with a
  // clamped value, and updating the pointee_type.  The order matters for
  // the case when we have to compute the length of a runtime array.  In
  // that the algorithm relies on the fact that that the earlier indices
  // have already been clamped.
  const uint32_t num_operands = inst.NumOperands();
  for (uint32_t idx = 3; !module_status_.failed && idx < num_operands; ++idx) {
    const uint32_t index_id = inst.GetSingleWordOperand(idx);
    Instruction* index_inst = GetDef(index_id);

    switch (pointee_type->opcode()) {
      case spv::Op::OpTypeMatrix:  // Use column count
      case spv::Op::OpTypeVector:  // Use component count
      {
        const uint32_t count = pointee_type->GetSingleWordOperand(2);
        clamp_to_literal_count(idx, count);
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
      } break;

      case spv::Op::OpTypeArray: {
        // The array length can be a spec constant, so go through the general
        // case.
        Instruction* array_len = GetDef(pointee_type->GetSingleWordOperand(2));
        clamp_to_count(idx, array_len);
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
      } break;

      case spv::Op::OpTypeStruct: {
        // SPIR-V requires the index to be an OpConstant.
        // We need to know the index literal value so we can compute the next
        // pointee type.
        if (index_inst->opcode() != spv::Op::OpConstant ||
            !constant_mgr->GetConstantFromInst(index_inst)
                 ->type()
                 ->AsInteger()) {
          Fail() << "Member index into struct is not a constant integer: "
                 << index_inst->PrettyPrint(
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
                 << "\nin access chain: "
                 << inst.PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
          return;
        }
        const auto num_members = pointee_type->NumInOperands();
        const auto* index_constant =
            constant_mgr->GetConstantFromInst(index_inst);
        // Get the sign-extended value, since access index is always treated as
        // signed.
        const auto index_value = index_constant->GetSignExtendedValue();
        if (index_value < 0 || index_value >= num_members) {
          Fail() << "Member index " << index_value
                 << " is out of bounds for struct type: "
                 << pointee_type->PrettyPrint(
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
                 << "\nin access chain: "
                 << inst.PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
          return;
        }
        pointee_type = GetDef(pointee_type->GetSingleWordInOperand(
            static_cast<uint32_t>(index_value)));
        // No need to clamp this index.  We just checked that it's valid.
      } break;

      case spv::Op::OpTypeRuntimeArray: {
        auto* array_len = MakeRuntimeArrayLengthInst(&inst, idx);
        if (!array_len) {  // We've already signaled an error.
          return;
        }
        clamp_to_count(idx, array_len);
        if (module_status_.failed) return;
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
      } break;

      default:
        Fail() << " Unhandled pointee type for access chain "
               << pointee_type->PrettyPrint(
                      SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
    }
  }
}

uint32_t GraphicsRobustAccessPass::GetGlslInsts() {
  if (module_status_.glsl_insts_id == 0) {
    // This string serves double-duty as raw data for a string and for a vector
    // of 32-bit words
    const char glsl[] = "GLSL.std.450";
    // Use an existing import if we can.
    for (auto& inst : context()->module()->ext_inst_imports()) {
      if (inst.GetInOperand(0).AsString() == glsl) {
        module_status_.glsl_insts_id = inst.result_id();
      }
    }
    if (module_status_.glsl_insts_id == 0) {
      // Make a new import instruction.
      module_status_.glsl_insts_id = TakeNextId();
      std::vector<uint32_t> words = spvtools::utils::MakeVector(glsl);
      auto import_inst = MakeUnique<Instruction>(
          context(), spv::Op::OpExtInstImport, 0, module_status_.glsl_insts_id,
          std::initializer_list<Operand>{
              Operand{SPV_OPERAND_TYPE_LITERAL_STRING, std::move(words)}});
      Instruction* inst = import_inst.get();
      context()->module()->AddExtInstImport(std::move(import_inst));
      module_status_.modified = true;
      context()->AnalyzeDefUse(inst);
      // Invalidates the feature manager, since we added an extended instruction
      // set import.
      context()->ResetFeatureManager();
    }
  }
  return module_status_.glsl_insts_id;
}

opt::Instruction* opt::GraphicsRobustAccessPass::GetValueForType(
    uint64_t value, const analysis::Integer* type) {
  auto* mgr = context()->get_constant_mgr();
  assert(type->width() <= 64);
  std::vector<uint32_t> words;
  words.push_back(uint32_t(value));
  if (type->width() > 32) {
    words.push_back(uint32_t(value >> 32u));
  }
  const auto* constant = mgr->GetConstant(type, words);
  return mgr->GetDefiningInstruction(
      constant, context()->get_type_mgr()->GetTypeInstruction(type));
}

opt::Instruction* opt::GraphicsRobustAccessPass::WidenInteger(
    bool sign_extend, uint32_t bit_width, Instruction* value,
    Instruction* before_inst) {
  analysis::Integer unsigned_type_for_query(bit_width, false);
  auto* type_mgr = context()->get_type_mgr();
  auto* unsigned_type = type_mgr->GetRegisteredType(&unsigned_type_for_query);
  auto type_id = context()->get_type_mgr()->GetId(unsigned_type);
  auto conversion_id = TakeNextId();
  auto* conversion = InsertInst(
      before_inst, (sign_extend ? spv::Op::OpSConvert : spv::Op::OpUConvert),
      type_id, conversion_id, {{SPV_OPERAND_TYPE_ID, {value->result_id()}}});
  return conversion;
}

Instruction* GraphicsRobustAccessPass::MakeUMinInst(
    const analysis::TypeManager& tm, Instruction* x, Instruction* y,
    Instruction* where) {
  // Get IDs of instructions we'll be referencing. Evaluate them before calling
  // the function so we force a deterministic ordering in case both of them need
  // to take a new ID.
  const uint32_t glsl_insts_id = GetGlslInsts();
  uint32_t smin_id = TakeNextId();
  const auto xwidth = tm.GetType(x->type_id())->AsInteger()->width();
  const auto ywidth = tm.GetType(y->type_id())->AsInteger()->width();
  assert(xwidth == ywidth);
  (void)xwidth;
  (void)ywidth;
  auto* smin_inst = InsertInst(
      where, spv::Op::OpExtInst, x->type_id(), smin_id,
      {
          {SPV_OPERAND_TYPE_ID, {glsl_insts_id}},
          {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {GLSLstd450UMin}},
          {SPV_OPERAND_TYPE_ID, {x->result_id()}},
          {SPV_OPERAND_TYPE_ID, {y->result_id()}},
      });
  return smin_inst;
}

Instruction* GraphicsRobustAccessPass::MakeSClampInst(
    const analysis::TypeManager& tm, Instruction* x, Instruction* min,
    Instruction* max, Instruction* where) {
  // Get IDs of instructions we'll be referencing. Evaluate them before calling
  // the function so we force a deterministic ordering in case both of them need
  // to take a new ID.
  const uint32_t glsl_insts_id = GetGlslInsts();
  uint32_t clamp_id = TakeNextId();
  const auto xwidth = tm.GetType(x->type_id())->AsInteger()->width();
  const auto minwidth = tm.GetType(min->type_id())->AsInteger()->width();
  const auto maxwidth = tm.GetType(max->type_id())->AsInteger()->width();
  assert(xwidth == minwidth);
  assert(xwidth == maxwidth);
  (void)xwidth;
  (void)minwidth;
  (void)maxwidth;
  auto* clamp_inst = InsertInst(
      where, spv::Op::OpExtInst, x->type_id(), clamp_id,
      {
          {SPV_OPERAND_TYPE_ID, {glsl_insts_id}},
          {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {GLSLstd450SClamp}},
          {SPV_OPERAND_TYPE_ID, {x->result_id()}},
          {SPV_OPERAND_TYPE_ID, {min->result_id()}},
          {SPV_OPERAND_TYPE_ID, {max->result_id()}},
      });
  return clamp_inst;
}

Instruction* GraphicsRobustAccessPass::MakeRuntimeArrayLengthInst(
    Instruction* access_chain, uint32_t operand_index) {
  // The Index parameter to the access chain at |operand_index| is indexing
  // *into* the runtime-array.  To get the number of elements in the runtime
  // array we need a pointer to the Block-decorated struct that contains the
  // runtime array. So conceptually we have to go 2 steps backward in the
  // access chain.  The two steps backward might forces us to traverse backward
  // across multiple dominating instructions.
  auto* type_mgr = context()->get_type_mgr();

  // How many access chain indices do we have to unwind to find the pointer
  // to the struct containing the runtime array?
  uint32_t steps_remaining = 2;
  // Find or create an instruction computing the pointer to the structure
  // containing the runtime array.
  // Walk backward through pointer address calculations until we either get
  // to exactly the right base pointer, or to an access chain instruction
  // that we can replicate but truncate to compute the address of the right
  // struct.
  Instruction* current_access_chain = access_chain;
  Instruction* pointer_to_containing_struct = nullptr;
  while (steps_remaining > 0) {
    switch (current_access_chain->opcode()) {
      case spv::Op::OpCopyObject:
        // Whoops. Walk right through this one.
        current_access_chain =
            GetDef(current_access_chain->GetSingleWordInOperand(0));
        break;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain: {
        const int first_index_operand = 3;
        // How many indices in this access chain contribute to getting us
        // to an element in the runtime array?
        const auto num_contributing_indices =
            current_access_chain == access_chain
                ? operand_index - (first_index_operand - 1)
                : current_access_chain->NumInOperands() - 1 /* skip the base */;
        Instruction* base =
            GetDef(current_access_chain->GetSingleWordInOperand(0));
        if (num_contributing_indices == steps_remaining) {
          // The base pointer points to the structure.
          pointer_to_containing_struct = base;
          steps_remaining = 0;
          break;
        } else if (num_contributing_indices < steps_remaining) {
          // Peel off the index and keep going backward.
          steps_remaining -= num_contributing_indices;
          current_access_chain = base;
        } else {
          // This access chain has more indices than needed.  Generate a new
          // access chain instruction, but truncating the list of indices.
          const int base_operand = 2;
          // We'll use the base pointer and the indices up to but not including
          // the one indexing into the runtime array.
          Instruction::OperandList ops;
          // Use the base pointer
          ops.push_back(current_access_chain->GetOperand(base_operand));
          const uint32_t num_indices_to_keep =
              num_contributing_indices - steps_remaining - 1;
          for (uint32_t i = 0; i <= num_indices_to_keep; i++) {
            ops.push_back(
                current_access_chain->GetOperand(first_index_operand + i));
          }
          // Compute the type of the result of the new access chain.  Start at
          // the base and walk the indices in a forward direction.
          auto* constant_mgr = context()->get_constant_mgr();
          std::vector<uint32_t> indices_for_type;
          for (uint32_t i = 0; i < ops.size() - 1; i++) {
            uint32_t index_for_type_calculation = 0;
            Instruction* index =
                GetDef(current_access_chain->GetSingleWordOperand(
                    first_index_operand + i));
            if (auto* index_constant =
                    constant_mgr->GetConstantFromInst(index)) {
              // We only need 32 bits. For the type calculation, it's sufficient
              // to take the zero-extended value. It only matters for the struct
              // case, and struct member indices are unsigned.
              index_for_type_calculation =
                  uint32_t(index_constant->GetZeroExtendedValue());
            } else {
              // Indexing into a variably-sized thing like an array.  Use 0.
              index_for_type_calculation = 0;
            }
            indices_for_type.push_back(index_for_type_calculation);
          }
          auto* base_ptr_type = type_mgr->GetType(base->type_id())->AsPointer();
          auto* base_pointee_type = base_ptr_type->pointee_type();
          auto* new_access_chain_result_pointee_type =
              type_mgr->GetMemberType(base_pointee_type, indices_for_type);
          const uint32_t new_access_chain_type_id = type_mgr->FindPointerToType(
              type_mgr->GetId(new_access_chain_result_pointee_type),
              base_ptr_type->storage_class());

          // Create the instruction and insert it.
          const auto new_access_chain_id = TakeNextId();
          auto* new_access_chain =
              InsertInst(current_access_chain, current_access_chain->opcode(),
                         new_access_chain_type_id, new_access_chain_id, ops);
          pointer_to_containing_struct = new_access_chain;
          steps_remaining = 0;
          break;
        }
      } break;
      default:
        Fail() << "Unhandled access chain in logical addressing mode passes "
                  "through "
               << current_access_chain->PrettyPrint(
                      SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET |
                      SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
        return nullptr;
    }
  }
  assert(pointer_to_containing_struct);
  auto* pointee_type =
      type_mgr->GetType(pointer_to_containing_struct->type_id())
          ->AsPointer()
          ->pointee_type();

  auto* struct_type = pointee_type->AsStruct();
  const uint32_t member_index_of_runtime_array =
      uint32_t(struct_type->element_types().size() - 1);
  // Create the length-of-array instruction before the original access chain,
  // but after the generation of the pointer to the struct.
  const auto array_len_id = TakeNextId();
  analysis::Integer uint_type_for_query(32, false);
  auto* uint_type = type_mgr->GetRegisteredType(&uint_type_for_query);
  auto* array_len = InsertInst(
      access_chain, spv::Op::OpArrayLength, type_mgr->GetId(uint_type),
      array_len_id,
      {{SPV_OPERAND_TYPE_ID, {pointer_to_containing_struct->result_id()}},
       {SPV_OPERAND_TYPE_LITERAL_INTEGER, {member_index_of_runtime_array}}});
  return array_len;
}

spv_result_t GraphicsRobustAccessPass::ClampCoordinateForImageTexelPointer(
    opt::Instruction* image_texel_pointer) {
  // TODO(dneto): Write tests for this code.
  // TODO(dneto): Use signed-clamp
  (void)(image_texel_pointer);
  return SPV_SUCCESS;

  // Do not compile this code until it is ready to be used.
#if 0
  // Example:
  //   %texel_ptr = OpImageTexelPointer %texel_ptr_type %image_ptr %coord
  //   %sample
  //
  // We want to clamp %coord components between vector-0 and the result
  // of OpImageQuerySize acting on the underlying image.  So insert:
  //     %image = OpLoad %image_type %image_ptr
  //     %query_size = OpImageQuerySize %query_size_type %image
  //
  // For a multi-sampled image, %sample is the sample index, and we need
  // to clamp it between zero and the number of samples in the image.
  //     %sample_count = OpImageQuerySamples %uint %image
  //     %max_sample_index = OpISub %uint %sample_count %uint_1
  // For non-multi-sampled images, the sample index must be constant zero.

  auto* def_use_mgr = context()->get_def_use_mgr();
  auto* type_mgr = context()->get_type_mgr();
  auto* constant_mgr = context()->get_constant_mgr();

  auto* image_ptr = GetDef(image_texel_pointer->GetSingleWordInOperand(0));
  auto* image_ptr_type = GetDef(image_ptr->type_id());
  auto image_type_id = image_ptr_type->GetSingleWordInOperand(1);
  auto* image_type = GetDef(image_type_id);
  auto* coord = GetDef(image_texel_pointer->GetSingleWordInOperand(1));
  auto* samples = GetDef(image_texel_pointer->GetSingleWordInOperand(2));

  // We will modify the module, at least by adding image query instructions.
  module_status_.modified = true;

  // Declare the ImageQuery capability if the module doesn't already have it.
  auto* feature_mgr = context()->get_feature_mgr();
  if (!feature_mgr->HasCapability(spv::Capability::ImageQuery)) {
    auto cap = MakeUnique<Instruction>(
        context(), spv::Op::OpCapability, 0, 0,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_CAPABILITY, {spv::Capability::ImageQuery}}});
    def_use_mgr->AnalyzeInstDefUse(cap.get());
    context()->AddCapability(std::move(cap));
    feature_mgr->Analyze(context()->module());
  }

  // OpImageTexelPointer is used to translate a coordinate and sample index
  // into an address for use with an atomic operation.  That is, it may only
  // used with what Vulkan calls a "storage image"
  // (OpTypeImage parameter Sampled=2).
  // Note: A storage image never has a level-of-detail associated with it.

  // Constraints on the sample id:
  //  - Only 2D images can be multi-sampled: OpTypeImage parameter MS=1
  //    only if Dim=2D.
  //  - Non-multi-sampled images (OpTypeImage parameter MS=0) must use
  //    sample ID to a constant 0.

  // The coordinate is treated as unsigned, and should be clamped against the
  // image "size", returned by OpImageQuerySize. (Note: OpImageQuerySizeLod
  // is only usable with a sampled image, i.e. its image type has Sampled=1).

  // Determine the result type for the OpImageQuerySize.
  // For non-arrayed images:
  //   non-Cube:
  //     - Always the same as the coordinate type
  //   Cube:
  //     - Use all but the last component of the coordinate (which is the face
  //       index from 0 to 5).
  // For arrayed images (in Vulkan the Dim is 1D, 2D, or Cube):
  //   non-Cube:
  //     - A vector with the components in the coordinate, and one more for
  //       the layer index.
  //   Cube:
  //     - The same as the coordinate type: 3-element integer vector.
  //     - The third component from the size query is the layer count.
  //     - The third component in the texel pointer calculation is
  //       6 * layer + face, where 0 <= face < 6.
  //   Cube: Use all but the last component of the coordinate (which is the face
  //   index from 0 to 5).
  const auto dim = SpvDim(image_type->GetSingleWordInOperand(1));
  const bool arrayed = image_type->GetSingleWordInOperand(3) == 1;
  const bool multisampled = image_type->GetSingleWordInOperand(4) != 0;
  const auto query_num_components = [dim, arrayed, this]() -> int {
    const int arrayness_bonus = arrayed ? 1 : 0;
    int num_coords = 0;
    switch (dim) {
      case spv::Dim::Buffer:
      case SpvDim1D:
        num_coords = 1;
        break;
      case spv::Dim::Cube:
        // For cube, we need bounds for x, y, but not face.
      case spv::Dim::Rect:
      case SpvDim2D:
        num_coords = 2;
        break;
      case SpvDim3D:
        num_coords = 3;
        break;
      case spv::Dim::SubpassData:
      case spv::Dim::Max:
        return Fail() << "Invalid image dimension for OpImageTexelPointer: "
                      << int(dim);
        break;
    }
    return num_coords + arrayness_bonus;
  }();
  const auto* coord_component_type = [type_mgr, coord]() {
    const analysis::Type* coord_type = type_mgr->GetType(coord->type_id());
    if (auto* vector_type = coord_type->AsVector()) {
      return vector_type->element_type()->AsInteger();
    }
    return coord_type->AsInteger();
  }();
  // For now, only handle 32-bit case for coordinates.
  if (!coord_component_type) {
    return Fail() << " Coordinates for OpImageTexelPointer are not integral: "
                  << image_texel_pointer->PrettyPrint(
                         SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  }
  if (coord_component_type->width() != 32) {
    return Fail() << " Expected OpImageTexelPointer coordinate components to "
                     "be 32-bits wide. They are "
                  << coord_component_type->width() << " bits. "
                  << image_texel_pointer->PrettyPrint(
                         SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  }
  const auto* query_size_type =
      [type_mgr, coord_component_type,
       query_num_components]() -> const analysis::Type* {
    if (query_num_components == 1) return coord_component_type;
    analysis::Vector proposed(coord_component_type, query_num_components);
    return type_mgr->GetRegisteredType(&proposed);
  }();

  const uint32_t image_id = TakeNextId();
  auto* image =
      InsertInst(image_texel_pointer, spv::Op::OpLoad, image_type_id, image_id,
                 {{SPV_OPERAND_TYPE_ID, {image_ptr->result_id()}}});

  const uint32_t query_size_id = TakeNextId();
  auto* query_size =
      InsertInst(image_texel_pointer, spv::Op::OpImageQuerySize,
                 type_mgr->GetTypeInstruction(query_size_type), query_size_id,
                 {{SPV_OPERAND_TYPE_ID, {image->result_id()}}});

  auto* component_1 = constant_mgr->GetConstant(coord_component_type, {1});
  const uint32_t component_1_id =
      constant_mgr->GetDefiningInstruction(component_1)->result_id();
  auto* component_0 = constant_mgr->GetConstant(coord_component_type, {0});
  const uint32_t component_0_id =
      constant_mgr->GetDefiningInstruction(component_0)->result_id();

  // If the image is a cube array, then the last component of the queried
  // size is the layer count.  In the query, we have to accommodate folding
  // in the face index ranging from 0 through 5. The inclusive upper bound
  // on the third coordinate therefore is multiplied by 6.
  auto* query_size_including_faces = query_size;
  if (arrayed && (dim == spv::Dim::Cube)) {
    // Multiply the last coordinate by 6.
    auto* component_6 = constant_mgr->GetConstant(coord_component_type, {6});
    const uint32_t component_6_id =
        constant_mgr->GetDefiningInstruction(component_6)->result_id();
    assert(query_num_components == 3);
    auto* multiplicand = constant_mgr->GetConstant(
        query_size_type, {component_1_id, component_1_id, component_6_id});
    auto* multiplicand_inst =
        constant_mgr->GetDefiningInstruction(multiplicand);
    const auto query_size_including_faces_id = TakeNextId();
    query_size_including_faces = InsertInst(
        image_texel_pointer, spv::Op::OpIMul,
        type_mgr->GetTypeInstruction(query_size_type),
        query_size_including_faces_id,
        {{SPV_OPERAND_TYPE_ID, {query_size_including_faces->result_id()}},
         {SPV_OPERAND_TYPE_ID, {multiplicand_inst->result_id()}}});
  }

  // Make a coordinate-type with all 1 components.
  auto* coordinate_1 =
      query_num_components == 1
          ? component_1
          : constant_mgr->GetConstant(
                query_size_type,
                std::vector<uint32_t>(query_num_components, component_1_id));
  // Make a coordinate-type with all 1 components.
  auto* coordinate_0 =
      query_num_components == 0
          ? component_0
          : constant_mgr->GetConstant(
                query_size_type,
                std::vector<uint32_t>(query_num_components, component_0_id));

  const uint32_t query_max_including_faces_id = TakeNextId();
  auto* query_max_including_faces = InsertInst(
      image_texel_pointer, spv::Op::OpISub,
      type_mgr->GetTypeInstruction(query_size_type),
      query_max_including_faces_id,
      {{SPV_OPERAND_TYPE_ID, {query_size_including_faces->result_id()}},
       {SPV_OPERAND_TYPE_ID,
        {constant_mgr->GetDefiningInstruction(coordinate_1)->result_id()}}});

  // Clamp the coordinate
  auto* clamp_coord = MakeSClampInst(
      *type_mgr, coord, constant_mgr->GetDefiningInstruction(coordinate_0),
      query_max_including_faces, image_texel_pointer);
  image_texel_pointer->SetInOperand(1, {clamp_coord->result_id()});

  // Clamp the sample index
  if (multisampled) {
    // Get the sample count via OpImageQuerySamples
    const auto query_samples_id = TakeNextId();
    auto* query_samples = InsertInst(
        image_texel_pointer, spv::Op::OpImageQuerySamples,
        constant_mgr->GetDefiningInstruction(component_0)->type_id(),
        query_samples_id, {{SPV_OPERAND_TYPE_ID, {image->result_id()}}});

    const auto max_samples_id = TakeNextId();
    auto* max_samples = InsertInst(image_texel_pointer, spv::Op::OpImageQuerySamples,
                                   query_samples->type_id(), max_samples_id,
                                   {{SPV_OPERAND_TYPE_ID, {query_samples_id}},
                                    {SPV_OPERAND_TYPE_ID, {component_1_id}}});

    auto* clamp_samples = MakeSClampInst(
        *type_mgr, samples, constant_mgr->GetDefiningInstruction(coordinate_0),
        max_samples, image_texel_pointer);
    image_texel_pointer->SetInOperand(2, {clamp_samples->result_id()});

  } else {
    // Just replace it with 0.  Don't even check what was there before.
    image_texel_pointer->SetInOperand(2, {component_0_id});
  }

  def_use_mgr->AnalyzeInstUse(image_texel_pointer);

  return SPV_SUCCESS;
#endif
}

opt::Instruction* GraphicsRobustAccessPass::InsertInst(
    opt::Instruction* where_inst, spv::Op opcode, uint32_t type_id,
    uint32_t result_id, const Instruction::OperandList& operands) {
  module_status_.modified = true;
  auto* result = where_inst->InsertBefore(
      MakeUnique<Instruction>(context(), opcode, type_id, result_id, operands));
  context()->get_def_use_mgr()->AnalyzeInstDefUse(result);
  auto* basic_block = context()->get_instr_block(where_inst);
  context()->set_instr_block(result, basic_block);
  return result;
}

}  // namespace opt
}  // namespace spvtools
