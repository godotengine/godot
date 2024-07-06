// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_add_image_sample_unused_components.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddImageSampleUnusedComponents::
    TransformationAddImageSampleUnusedComponents(
        protobufs::TransformationAddImageSampleUnusedComponents message)
    : message_(std::move(message)) {}

TransformationAddImageSampleUnusedComponents::
    TransformationAddImageSampleUnusedComponents(
        uint32_t coordinate_with_unused_components_id,
        const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_coordinate_with_unused_components_id(
      coordinate_with_unused_components_id);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationAddImageSampleUnusedComponents::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto image_sample_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);

  // The image sample instruction must be defined.
  if (image_sample_instruction == nullptr) {
    return false;
  }

  // The instruction must be an image sample instruction.
  if (!spvOpcodeIsImageSample(image_sample_instruction->opcode())) {
    return false;
  }

  uint32_t coordinate_id = image_sample_instruction->GetSingleWordInOperand(1);
  auto coordinate_instruction =
      ir_context->get_def_use_mgr()->GetDef(coordinate_id);
  auto coordinate_type =
      ir_context->get_type_mgr()->GetType(coordinate_instruction->type_id());

  // It must be possible to add unused components.
  if (coordinate_type->AsVector() &&
      coordinate_type->AsVector()->element_count() == 4) {
    return false;
  }

  auto coordinate_with_unused_components_instruction =
      ir_context->get_def_use_mgr()->GetDef(
          message_.coordinate_with_unused_components_id());

  // The coordinate with unused components instruction must be defined.
  if (coordinate_with_unused_components_instruction == nullptr) {
    return false;
  }

  // It must be an OpCompositeConstruct instruction such that it can be checked
  // that the original components are present.
  if (coordinate_with_unused_components_instruction->opcode() !=
      spv::Op::OpCompositeConstruct) {
    return false;
  }

  // The first constituent must be the original coordinate.
  if (coordinate_with_unused_components_instruction->GetSingleWordInOperand(
          0) != coordinate_id) {
    return false;
  }

  auto coordinate_with_unused_components_type =
      ir_context->get_type_mgr()->GetType(
          coordinate_with_unused_components_instruction->type_id());

  // |coordinate_with_unused_components_type| must be a vector.
  if (!coordinate_with_unused_components_type->AsVector()) {
    return false;
  }

  return true;
}

void TransformationAddImageSampleUnusedComponents::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Sets the coordinate operand.
  auto image_sample_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  image_sample_instruction->SetInOperand(
      1, {message_.coordinate_with_unused_components_id()});
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationAddImageSampleUnusedComponents::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_image_sample_unused_components() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationAddImageSampleUnusedComponents::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
