// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"

#include "source/fuzz/comparator_deep_blocks_first.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_flatten_conditional_branch.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that randomly selects conditional branches to flatten and
// flattens them, if possible.
FuzzerPassFlattenConditionalBranches::FuzzerPassFlattenConditionalBranches(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassFlattenConditionalBranches::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    // Get all the selection headers that we want to flatten. We need to collect
    // all of them first, because, since we are changing the structure of the
    // module, it's not safe to modify them while iterating.
    std::vector<opt::BasicBlock*> selection_headers;

    for (auto& block : function) {
      // Randomly decide whether to consider this block.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfFlatteningConditionalBranch())) {
        continue;
      }

      // Only consider this block if it is the header of a conditional, with a
      // non-irrelevant condition.
      if (block.GetMergeInst() &&
          block.GetMergeInst()->opcode() == spv::Op::OpSelectionMerge &&
          block.terminator()->opcode() == spv::Op::OpBranchConditional &&
          !GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
              block.terminator()->GetSingleWordInOperand(0))) {
        selection_headers.emplace_back(&block);
      }
    }

    // Sort the headers so that those that are more deeply nested are considered
    // first, possibly enabling outer conditionals to be flattened.
    std::sort(selection_headers.begin(), selection_headers.end(),
              ComparatorDeepBlocksFirst(GetIRContext()));

    // Apply the transformation to the headers which can be flattened.
    for (auto header : selection_headers) {
      // Make a set to keep track of the instructions that need fresh ids.
      std::set<opt::Instruction*> instructions_that_need_ids;

      // Do not consider this header if the conditional cannot be flattened.
      if (!TransformationFlattenConditionalBranch::
              GetProblematicInstructionsIfConditionalCanBeFlattened(
                  GetIRContext(), header, *GetTransformationContext(),
                  &instructions_that_need_ids)) {
        continue;
      }

      uint32_t convergence_block_id =
          TransformationFlattenConditionalBranch::FindConvergenceBlock(
              GetIRContext(), *header);

      // If the SPIR-V version is restricted so that OpSelect can only work on
      // scalar, pointer and vector types then we cannot apply this
      // transformation to a header whose convergence block features OpPhi
      // instructions on different types, as we cannot convert such instructions
      // to OpSelect instructions.
      if (TransformationFlattenConditionalBranch::
              OpSelectArgumentsAreRestricted(GetIRContext())) {
        if (!GetIRContext()
                 ->cfg()
                 ->block(convergence_block_id)
                 ->WhileEachPhiInst(
                     [this](opt::Instruction* phi_instruction) -> bool {
                       switch (GetIRContext()
                                   ->get_def_use_mgr()
                                   ->GetDef(phi_instruction->type_id())
                                   ->opcode()) {
                         case spv::Op::OpTypeBool:
                         case spv::Op::OpTypeInt:
                         case spv::Op::OpTypeFloat:
                         case spv::Op::OpTypePointer:
                         case spv::Op::OpTypeVector:
                           return true;
                         default:
                           return false;
                       }
                     })) {
          // An OpPhi is performed on a type not supported by OpSelect; we
          // cannot flatten this selection.
          continue;
        }
      }

      // If the construct's convergence block features OpPhi instructions with
      // vector result types then we may be *forced*, by the SPIR-V version, to
      // turn these into component-wise OpSelect instructions, or we might wish
      // to do so anyway.  The following booleans capture whether we will opt
      // to use a component-wise select even if we don't have to.
      bool use_component_wise_2d_select_even_if_optional =
          GetFuzzerContext()->ChooseEven();
      bool use_component_wise_3d_select_even_if_optional =
          GetFuzzerContext()->ChooseEven();
      bool use_component_wise_4d_select_even_if_optional =
          GetFuzzerContext()->ChooseEven();

      // If we do need to perform any component-wise selections, we will need a
      // fresh id for a boolean vector representing the selection's condition
      // repeated N times, where N is the vector dimension.
      uint32_t fresh_id_for_bvec2_selector = 0;
      uint32_t fresh_id_for_bvec3_selector = 0;
      uint32_t fresh_id_for_bvec4_selector = 0;

      GetIRContext()
          ->cfg()
          ->block(convergence_block_id)
          ->ForEachPhiInst([this, &fresh_id_for_bvec2_selector,
                            &fresh_id_for_bvec3_selector,
                            &fresh_id_for_bvec4_selector,
                            use_component_wise_2d_select_even_if_optional,
                            use_component_wise_3d_select_even_if_optional,
                            use_component_wise_4d_select_even_if_optional](
                               opt::Instruction* phi_instruction) {
            opt::Instruction* type_instruction =
                GetIRContext()->get_def_use_mgr()->GetDef(
                    phi_instruction->type_id());
            switch (type_instruction->opcode()) {
              case spv::Op::OpTypeVector: {
                uint32_t dimension =
                    type_instruction->GetSingleWordInOperand(1);
                switch (dimension) {
                  case 2:
                    PrepareForOpPhiOnVectors(
                        dimension,
                        use_component_wise_2d_select_even_if_optional,
                        &fresh_id_for_bvec2_selector);
                    break;
                  case 3:
                    PrepareForOpPhiOnVectors(
                        dimension,
                        use_component_wise_3d_select_even_if_optional,
                        &fresh_id_for_bvec3_selector);
                    break;
                  case 4:
                    PrepareForOpPhiOnVectors(
                        dimension,
                        use_component_wise_4d_select_even_if_optional,
                        &fresh_id_for_bvec4_selector);
                    break;
                  default:
                    assert(false && "Invalid vector dimension.");
                }
                break;
              }
              default:
                break;
            }
          });

      // Some instructions will require to be enclosed inside conditionals
      // because they have side effects (for example, loads and stores). Some of
      // this have no result id, so we require instruction descriptors to
      // identify them. Each of them is associated with the necessary ids for it
      // via a SideEffectWrapperInfo message.
      std::vector<protobufs::SideEffectWrapperInfo> wrappers_info;

      for (auto instruction : instructions_that_need_ids) {
        protobufs::SideEffectWrapperInfo wrapper_info;
        *wrapper_info.mutable_instruction() =
            MakeInstructionDescriptor(GetIRContext(), instruction);
        wrapper_info.set_merge_block_id(GetFuzzerContext()->GetFreshId());
        wrapper_info.set_execute_block_id(GetFuzzerContext()->GetFreshId());

        // If the instruction has a non-void result id, we need to define more
        // fresh ids and provide an id of the suitable type whose value can be
        // copied in order to create a placeholder id.
        if (TransformationFlattenConditionalBranch::InstructionNeedsPlaceholder(
                GetIRContext(), *instruction)) {
          wrapper_info.set_actual_result_id(GetFuzzerContext()->GetFreshId());
          wrapper_info.set_alternative_block_id(
              GetFuzzerContext()->GetFreshId());
          wrapper_info.set_placeholder_result_id(
              GetFuzzerContext()->GetFreshId());

          // The id will be a zero constant if the type allows it, and an
          // OpUndef otherwise. We want to avoid using OpUndef, if possible, to
          // avoid undefined behaviour in the module as much as possible.
          if (fuzzerutil::CanCreateConstant(GetIRContext(),
                                            instruction->type_id())) {
            wrapper_info.set_value_to_copy_id(
                FindOrCreateZeroConstant(instruction->type_id(), true));
          } else {
            wrapper_info.set_value_to_copy_id(
                FindOrCreateGlobalUndef(instruction->type_id()));
          }
        }

        wrappers_info.push_back(std::move(wrapper_info));
      }

      // Apply the transformation, evenly choosing whether to lay out the true
      // branch or the false branch first.
      ApplyTransformation(TransformationFlattenConditionalBranch(
          header->id(), GetFuzzerContext()->ChooseEven(),
          fresh_id_for_bvec2_selector, fresh_id_for_bvec3_selector,
          fresh_id_for_bvec4_selector, wrappers_info));
    }
  }
}

void FuzzerPassFlattenConditionalBranches::PrepareForOpPhiOnVectors(
    uint32_t vector_dimension, bool use_vector_select_if_optional,
    uint32_t* fresh_id_for_bvec_selector) {
  if (*fresh_id_for_bvec_selector != 0) {
    // We already have a fresh id for a component-wise OpSelect of this
    // dimension
    return;
  }
  if (TransformationFlattenConditionalBranch::OpSelectArgumentsAreRestricted(
          GetIRContext()) ||
      use_vector_select_if_optional) {
    // We either have to, or have chosen to, perform a component-wise select, so
    // we ensure that the right boolean vector type is available, and grab a
    // fresh id.
    FindOrCreateVectorType(FindOrCreateBoolType(), vector_dimension);
    *fresh_id_for_bvec_selector = GetFuzzerContext()->GetFreshId();
  }
}

}  // namespace fuzz
}  // namespace spvtools
