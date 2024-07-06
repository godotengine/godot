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

#include "source/fuzz/fuzzer_pass_add_image_sample_unused_components.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_image_sample_unused_components.h"
#include "source/fuzz/transformation_composite_construct.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddImageSampleUnusedComponents::
    FuzzerPassAddImageSampleUnusedComponents(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddImageSampleUnusedComponents::Apply() {
  // SPIR-V module to help understand the transformation.
  //
  //       OpCapability Shader
  //  %1 = OpExtInstImport "GLSL.std.450"
  //       OpMemoryModel Logical GLSL450
  //       OpEntryPoint Fragment %15 "main" %12 %14
  //       OpExecutionMode %15 OriginUpperLeft
  //
  // ; Decorations
  //        OpDecorate %12 Location 0 ; Input color variable location
  //        OpDecorate %13 DescriptorSet 0 ; Image coordinate variable
  //        descriptor set OpDecorate %13 Binding 0 ; Image coordinate
  //        variable binding OpDecorate %14 Location 0 ; Fragment color
  //        variable location
  //
  // ; Types
  //  %2 = OpTypeVoid
  //  %3 = OpTypeFunction %2
  //  %4 = OpTypeFloat 32
  //  %5 = OpTypeVector %4 2
  //  %6 = OpTypeVector %4 4
  //  %7 = OpTypeImage %4 2D 0 0 0 1 Rgba32f
  //  %8 = OpTypeSampledImage %7
  //  %9 = OpTypePointer Input %5
  // %10 = OpTypePointer UniformConstant %8
  // %11 = OpTypePointer Output %6
  //
  // ; Variables
  // %12 = OpVariable %9 Input ; Input image coordinate variable
  // %13 = OpVariable %10 UniformConstant ; Image variable
  // %14 = OpVariable %11 Output ; Fragment color variable
  //
  // ; main function
  // %15 = OpFunction %2 None %3
  // %16 = OpLabel
  // %17 = OpLoad %5 %12
  // %18 = OpLoad %8 %13
  // %19 = OpImageSampleImplicitLod %6 %18 %17
  //       OpStore %14 %19
  //       OpReturn
  //       OpFunctionEnd

  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    // |instruction| %19 = OpImageSampleImplicitLod %6 %18 %17
    if (!spvOpcodeIsImageSample(instruction->opcode())) {
      return;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfAddingImageSampleUnusedComponents())) {
      return;
    }

    // Gets image sample coordinate information.
    // |coordinate_instruction| %17 = OpLoad %5 %12
    uint32_t coordinate_id = instruction->GetSingleWordInOperand(1);
    auto coordinate_instruction =
        GetIRContext()->get_def_use_mgr()->GetDef(coordinate_id);
    auto coordinate_type = GetIRContext()->get_type_mgr()->GetType(
        coordinate_instruction->type_id());

    // If the coordinate is a 4-dimensional vector, then no unused components
    // may be added.
    if (coordinate_type->AsVector() &&
        coordinate_type->AsVector()->element_count() == 4) {
      return;
    }

    // If the coordinate is a scalar, then at most 3 unused components may be
    // added. If the coordinate is a vector, then the maximum number of unused
    // components depends on the vector size.
    // For the sample module, the coordinate type instruction is %5 =
    // OpTypeVector %4 2, thus |max_unused_component_count| = 4 - 2 = 2.
    uint32_t max_unused_component_count =
        coordinate_type->AsInteger() || coordinate_type->AsFloat()
            ? 3
            : 4 - coordinate_type->AsVector()->element_count();

    // |unused_component_count| may be 1 or 2.
    uint32_t unused_component_count =
        GetFuzzerContext()->GetRandomUnusedComponentCountForImageSample(
            max_unused_component_count);

    // Gets a type for the zero-unused components.
    uint32_t zero_constant_type_id;
    switch (unused_component_count) {
      case 1:
        // If the coordinate is an integer or float, then the unused components
        // type is the same as the coordinate. If the coordinate is a vector,
        // then the unused components type is the same as the vector components
        // type.
        zero_constant_type_id =
            coordinate_type->AsInteger() || coordinate_type->AsFloat()
                ? coordinate_instruction->type_id()
                : GetIRContext()->get_type_mgr()->GetId(
                      coordinate_type->AsVector()->element_type());
        break;
      case 2:
      case 3:
        // If the coordinate is an integer or float, then the unused components
        // type is the same as the coordinate. If the coordinate is a vector,
        // then the unused components type is the same as the coordinate
        // components type.
        // |zero_constant_type_id| %5 = OpTypeVector %4 2
        zero_constant_type_id =
            coordinate_type->AsInteger() || coordinate_type->AsFloat()
                ? FindOrCreateVectorType(coordinate_instruction->type_id(),
                                         unused_component_count)
                : FindOrCreateVectorType(
                      GetIRContext()->get_type_mgr()->GetId(
                          coordinate_type->AsVector()->element_type()),
                      unused_component_count);
        break;
      default:
        assert(false && "Should be unreachable.");
        zero_constant_type_id = 0;
        break;
    }

    // Gets |coordinate_type| again because the module may have changed due to
    // the use of FindOrCreateVectorType above.
    coordinate_type = GetIRContext()->get_type_mgr()->GetType(
        coordinate_instruction->type_id());

    // If the new vector type with unused components does not exist, then create
    // it. |coordinate_with_unused_components_type_id| %6 = OpTypeVector %4 4
    uint32_t coordinate_with_unused_components_type_id =
        coordinate_type->AsInteger() || coordinate_type->AsFloat()
            ? FindOrCreateVectorType(coordinate_instruction->type_id(),
                                     1 + unused_component_count)
            : FindOrCreateVectorType(
                  GetIRContext()->get_type_mgr()->GetId(
                      coordinate_type->AsVector()->element_type()),
                  coordinate_type->AsVector()->element_count() +
                      unused_component_count);

    // Inserts an OpCompositeConstruct instruction which
    // represents the coordinate with unused components.
    // |coordinate_with_unused_components_id|
    // %22 = OpCompositeConstruct %6 %17 %21
    uint32_t coordinate_with_unused_components_id =
        GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationCompositeConstruct(
        coordinate_with_unused_components_type_id,
        {coordinate_instruction->result_id(),
         // FindOrCreateZeroConstant
         // %20 = OpConstant %4 0
         // %21 = OpConstantComposite %5 %20 %20
         FindOrCreateZeroConstant(zero_constant_type_id, true)},
        MakeInstructionDescriptor(GetIRContext(), instruction),
        coordinate_with_unused_components_id));

    // Tries to add unused components to the image sample coordinate.
    // %19 = OpImageSampleImplicitLod %6 %18 %22
    ApplyTransformation(TransformationAddImageSampleUnusedComponents(
        coordinate_with_unused_components_id,
        MakeInstructionDescriptor(GetIRContext(), instruction)));
  });
}

}  // namespace fuzz
}  // namespace spvtools
