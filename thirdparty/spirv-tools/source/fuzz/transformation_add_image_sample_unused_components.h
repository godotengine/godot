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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_IMAGE_SAMPLE_UNUSED_COMPONENTS_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_IMAGE_SAMPLE_UNUSED_COMPONENTS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddImageSampleUnusedComponents : public Transformation {
 public:
  explicit TransformationAddImageSampleUnusedComponents(
      protobufs::TransformationAddImageSampleUnusedComponents message);

  TransformationAddImageSampleUnusedComponents(
      uint32_t coordinate_with_unused_components_id,
      const protobufs::InstructionDescriptor& instruction_descriptor);

  // - |coordinate_with_unused_components_id| must identify a vector such that
  //   the first components match the components of the image sample coordinate.
  // - |message_.instruction_descriptor| must identify an image sample
  //   instruction
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Add unused components to an image sample coordinate by replacing the
  // coordinate with |coordinate_with_unused_components_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddImageSampleUnusedComponents message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_IMAGE_SAMPLE_UNUSED_COMPONENTS_H_
