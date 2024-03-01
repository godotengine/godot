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

#ifndef SOURCE_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_
#define SOURCE_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SetSpecConstantDefaultValuePass : public Pass {
 public:
  using SpecIdToValueStrMap = std::unordered_map<uint32_t, std::string>;
  using SpecIdToValueBitPatternMap =
      std::unordered_map<uint32_t, std::vector<uint32_t>>;
  using SpecIdToInstMap = std::unordered_map<uint32_t, Instruction*>;

  // Constructs a pass instance with a map from spec ids to default values
  // in the form of string.
  explicit SetSpecConstantDefaultValuePass(
      const SpecIdToValueStrMap& default_values)
      : spec_id_to_value_str_(default_values),
        spec_id_to_value_bit_pattern_() {}
  explicit SetSpecConstantDefaultValuePass(SpecIdToValueStrMap&& default_values)
      : spec_id_to_value_str_(std::move(default_values)),
        spec_id_to_value_bit_pattern_() {}

  // Constructs a pass instance with a map from spec ids to default values in
  // the form of bit pattern.
  explicit SetSpecConstantDefaultValuePass(
      const SpecIdToValueBitPatternMap& default_values)
      : spec_id_to_value_str_(),
        spec_id_to_value_bit_pattern_(default_values) {}
  explicit SetSpecConstantDefaultValuePass(
      SpecIdToValueBitPatternMap&& default_values)
      : spec_id_to_value_str_(),
        spec_id_to_value_bit_pattern_(std::move(default_values)) {}

  const char* name() const override { return "set-spec-const-default-value"; }
  Status Process() override;

  // Parses the given null-terminated C string to get a mapping from Spec Id to
  // default value strings. Returns a unique pointer of the mapping from spec
  // ids to spec constant default value strings built from the given |str| on
  // success. Returns a nullptr if the given string is not valid for building
  // the mapping.
  // A valid string for building the mapping should follow the rule below:
  //
  //  "<spec id A>:<default value for A> <spec id B>:<default value for B> ..."
  //  Example:
  //    "200:0x11   201:3.14   202:1.4728"
  //
  //  Entries are separated with blank spaces (i.e.:' ', '\n', '\r', '\t',
  //  '\f', '\v'). Each entry corresponds to a Spec Id and default value pair.
  //  Multiple spaces between, before or after entries are allowed. However,
  //  spaces are not allowed within spec id or the default value string because
  //  spaces are always considered as delimiter to separate entries.
  //
  //  In each entry, the spec id and value string is separated by ':'. Missing
  //  ':' in any entry is invalid. And it is invalid to have blank spaces in
  //  between the spec id and ':' or the default value and ':'.
  //
  //  <spec id>: specifies the spec id value.
  //    The text must represent a valid uint32_t number.
  //    Hex format with '0x' prefix is allowed.
  //    Empty <spec id> is not allowed.
  //    One spec id value can only be defined once, multiple default values
  //      defined for the same spec id is not allowed. Spec ids with same value
  //      but different formats (e.g. 0x100 and 256) are considered the same.
  //
  //  <default value>: the default value string.
  //    Spaces before and after default value text is allowed.
  //    Spaces within the text is not allowed.
  //    Empty <default value> is not allowed.
  static std::unique_ptr<SpecIdToValueStrMap> ParseDefaultValuesString(
      const char* str);

 private:
  // The mappings from spec ids to default values. Two maps are defined here,
  // each to be used for one specific form of the default values. Only one of
  // them will be populated in practice.

  // The mapping from spec ids to their string-form default values to be set.
  const SpecIdToValueStrMap spec_id_to_value_str_;
  // The mapping from spec ids to their bitpattern-form default values to be
  // set.
  const SpecIdToValueBitPatternMap spec_id_to_value_bit_pattern_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_
