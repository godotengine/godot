// Copyright (c) 2021 Google LLC
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

#ifndef SOURCE_OPT_CONVERT_TO_SAMPLED_IMAGE_PASS_H_
#define SOURCE_OPT_CONVERT_TO_SAMPLED_IMAGE_PASS_H_

#include <memory>
#include <unordered_set>
#include <utility>

#include "source/opt/pass.h"
#include "source/opt/types.h"

namespace spvtools {
namespace opt {

// A struct for a pair of descriptor set and binding.
struct DescriptorSetAndBinding {
  uint32_t descriptor_set;
  uint32_t binding;

  bool operator==(const DescriptorSetAndBinding& descriptor_set_binding) const {
    return descriptor_set_binding.descriptor_set == descriptor_set &&
           descriptor_set_binding.binding == binding;
  }
};

// See optimizer.hpp for documentation.
class ConvertToSampledImagePass : public Pass {
 public:
  // Hashing functor for the pair of descriptor set and binding.
  struct DescriptorSetAndBindingHash {
    size_t operator()(
        const DescriptorSetAndBinding& descriptor_set_binding) const {
      return std::hash<uint32_t>()(descriptor_set_binding.descriptor_set) ^
             std::hash<uint32_t>()(descriptor_set_binding.binding);
    }
  };

  using SetOfDescriptorSetAndBindingPairs =
      std::unordered_set<DescriptorSetAndBinding, DescriptorSetAndBindingHash>;
  using DescriptorSetBindingToInstruction =
      std::unordered_map<DescriptorSetAndBinding, Instruction*,
                         DescriptorSetAndBindingHash>;

  explicit ConvertToSampledImagePass(
      const std::vector<DescriptorSetAndBinding>& descriptor_set_binding_pairs)
      : descriptor_set_binding_pairs_(descriptor_set_binding_pairs.begin(),
                                      descriptor_set_binding_pairs.end()) {}

  const char* name() const override { return "convert-to-sampled-image"; }
  Status Process() override;

  // Parses the given null-terminated C string to get a vector of descriptor set
  // and binding pairs. Returns a unique pointer to the vector of descriptor set
  // and binding pairs built from the given |str| on success. Returns a nullptr
  // if the given string is not valid for building the vector of pairs.
  // A valid string for building the vector of pairs should follow the rule
  // below:
  //
  //  "<descriptor set>:<binding> <descriptor set>:<binding> ..."
  //  Example:
  //    "3:5 2:1 0:4"
  //
  //  Entries are separated with blank spaces (i.e.:' ', '\n', '\r', '\t',
  //  '\f', '\v'). Each entry corresponds to a descriptor set and binding pair.
  //  Multiple spaces between, before or after entries are allowed. However,
  //  spaces are not allowed within a descriptor set or binding.
  //
  //  In each entry, the descriptor set and binding are separated by ':'.
  //  Missing ':' in any entry is invalid. And it is invalid to have blank
  //  spaces in between the descriptor set and ':' or ':' and the binding.
  //
  //  <descriptor set>: the descriptor set.
  //    The text must represent a valid uint32_t number.
  //
  //  <binding>: the binding.
  //    The text must represent a valid uint32_t number.
  static std::unique_ptr<std::vector<DescriptorSetAndBinding>>
  ParseDescriptorSetBindingPairsString(const char* str);

 private:
  // Collects resources to convert to sampled image and saves them in
  // |descriptor_set_binding_pair_to_sampler| if the resource is a sampler and
  // saves them in |descriptor_set_binding_pair_to_image| if the resource is an
  // image. Returns false if two samplers or two images have the same descriptor
  // set and binding. Otherwise, returns true.
  bool CollectResourcesToConvert(
      DescriptorSetBindingToInstruction* descriptor_set_binding_pair_to_sampler,
      DescriptorSetBindingToInstruction* descriptor_set_binding_pair_to_image)
      const;

  // Finds an OpDecorate with DescriptorSet decorating |inst| and another
  // OpDecorate with Binding decorating |inst|. Stores the descriptor set and
  // binding in |descriptor_set_binding|. Returns whether it successfully finds
  // the descriptor set and binding or not.
  bool GetDescriptorSetBinding(
      const Instruction& inst,
      DescriptorSetAndBinding* descriptor_set_binding) const;

  // Returns whether |descriptor_set_binding| is a pair of a descriptor set
  // and a binding that we have to convert resources with it to a sampled image
  // or not.
  bool ShouldResourceBeConverted(
      const DescriptorSetAndBinding& descriptor_set_binding) const;

  // Returns the pointee type of the type of variable |variable|. If |variable|
  // is not an OpVariable instruction, just returns nullptr.
  const analysis::Type* GetVariableType(const Instruction& variable) const;

  // Returns the storage class of |variable|.
  spv::StorageClass GetStorageClass(const Instruction& variable) const;

  // Finds |inst|'s users whose opcode is |user_opcode| or users of OpCopyObject
  // instructions of |inst| whose opcode is |user_opcode| and puts them in
  // |uses|.
  void FindUses(const Instruction* inst, std::vector<Instruction*>* uses,
                spv::Op user_opcode) const;

  // Finds OpImage* instructions using |image| or OpCopyObject instructions that
  // copy |image| and puts them in |uses|.
  void FindUsesOfImage(const Instruction* image,
                       std::vector<Instruction*>* uses) const;

  // Creates an OpImage instruction that extracts the image from the sampled
  // image |sampled_image|.
  Instruction* CreateImageExtraction(Instruction* sampled_image);

  // Converts |image_variable| whose type is an image pointer to sampled image
  // type. Updates users of |image_variable| accordingly. If some instructions
  // e.g., OpImageRead use |image_variable| as an Image operand, creates an
  // image extracted from the sampled image using OpImage and replace the Image
  // operands of the users with the extracted image. If some OpSampledImage
  // instructions use |image_variable| and sampler whose descriptor set and
  // binding are the same with |image_variable|, just combines |image_variable|
  // and the sampler to a sampled image.
  Pass::Status UpdateImageVariableToSampledImage(
      Instruction* image_variable,
      const DescriptorSetAndBinding& descriptor_set_binding);

  // Returns the id of type sampled image type whose image type is the one of
  // |image_variable|.
  uint32_t GetSampledImageTypeForImage(Instruction* image_variable);

  // Moves |inst| next to the OpType* instruction with |type_id|.
  void MoveInstructionNextToType(Instruction* inst, uint32_t type_id);

  // Converts |image_variable| whose type is an image pointer to sampled image
  // with the type id |sampled_image_type_id|. Returns whether it successfully
  // converts the type of |image_variable| or not.
  bool ConvertImageVariableToSampledImage(Instruction* image_variable,
                                          uint32_t sampled_image_type_id);

  // Replaces |sampled_image_load| instruction used by OpImage* with the image
  // extracted from |sampled_image_load|. Returns the extracted image or nullptr
  // if it does not have uses.
  Instruction* UpdateImageUses(Instruction* sampled_image_load);

  // Returns true if the sampler of |sampled_image_inst| is decorated by a
  // descriptor set and a binding |descriptor_set_binding|.
  bool IsSamplerOfSampledImageDecoratedByDescriptorSetBinding(
      Instruction* sampled_image_inst,
      const DescriptorSetAndBinding& descriptor_set_binding);

  // Replaces OpSampledImage instructions using |image_load| with |image_load|
  // if the sampler of the OpSampledImage instruction has descriptor set and
  // binding |image_descriptor_set_binding|. Otherwise, replaces |image_load|
  // with |image_extraction|.
  void UpdateSampledImageUses(
      Instruction* image_load, Instruction* image_extraction,
      const DescriptorSetAndBinding& image_descriptor_set_binding);

  // Checks the uses of |sampler_variable|. When a sampler is used by
  // OpSampledImage instruction, the corresponding image must be
  // |image_to_be_combined_with| that should be already converted to a sampled
  // image by UpdateImageVariableToSampledImage() method.
  Pass::Status CheckUsesOfSamplerVariable(
      const Instruction* sampler_variable,
      Instruction* image_to_be_combined_with);

  // Returns true if Image operand of |sampled_image_inst| is the image of
  // |image_variable|.
  bool DoesSampledImageReferenceImage(Instruction* sampled_image_inst,
                                      Instruction* image_variable);

  // A set of pairs of descriptor set and binding. If an image and/or a sampler
  // have a pair of descriptor set and binding that is an element of
  // |descriptor_set_binding_pairs_|, they/it will be converted to a sampled
  // image by this pass.
  const SetOfDescriptorSetAndBindingPairs descriptor_set_binding_pairs_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONVERT_TO_SAMPLED_IMAGE_PASS_H_
