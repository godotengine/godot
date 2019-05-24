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

#ifndef SOURCE_VAL_DECORATION_H_
#define SOURCE_VAL_DECORATION_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "source/latest_version_spirv_header.h"

namespace spvtools {
namespace val {

// An object of this class represents a specific decoration including its
// parameters (if any). Decorations are used by OpDecorate and OpMemberDecorate,
// and they describe certain properties that can be assigned to one or several
// <id>s.
//
// A Decoration object contains the decoration type (an enum), associated
// literal parameters, and struct member index. If the decoration does not apply
// to a struct member, then the index is kInvalidIndex. A Decoration object does
// not store the target Id, i.e. the Id to which it applies. It is
// possible for the same decoration to be applied to several <id>s (and they
// might be assigned using separate SPIR-V instructions, possibly using an
// assignment through GroupDecorate).
//
// Example 1: Decoration for an object<id> with no parameters:
// OpDecorate %obj Flat
//            dec_type_ = SpvDecorationFlat
//              params_ = empty vector
// struct_member_index_ = kInvalidMember
//
// Example 2: Decoration for an object<id> with two parameters:
// OpDecorate %obj LinkageAttributes "link" Import
//            dec_type_ = SpvDecorationLinkageAttributes
//              params_ = vector { link, Import }
// struct_member_index_ = kInvalidMember
//
// Example 3: Decoration for a member of a structure with one parameter:
// OpMemberDecorate %struct 2 Offset 2
//            dec_type_ = SpvDecorationOffset
//              params_ = vector { 2 }
// struct_member_index_ = 2
//
class Decoration {
 public:
  enum { kInvalidMember = -1 };
  Decoration(SpvDecoration t,
             const std::vector<uint32_t>& parameters = std::vector<uint32_t>(),
             uint32_t member_index = kInvalidMember)
      : dec_type_(t), params_(parameters), struct_member_index_(member_index) {}

  void set_struct_member_index(uint32_t index) { struct_member_index_ = index; }
  int struct_member_index() const { return struct_member_index_; }
  SpvDecoration dec_type() const { return dec_type_; }
  std::vector<uint32_t>& params() { return params_; }
  const std::vector<uint32_t>& params() const { return params_; }

  inline bool operator==(const Decoration& rhs) const {
    return (dec_type_ == rhs.dec_type_ && params_ == rhs.params_ &&
            struct_member_index_ == rhs.struct_member_index_);
  }

 private:
  SpvDecoration dec_type_;
  std::vector<uint32_t> params_;

  // If the decoration applies to a member of a structure type, then the index
  // of the member is stored here. Otherwise, this is kInvalidIndex.
  int struct_member_index_;
};

}  // namespace val
}  // namespace spvtools

#endif  // SOURCE_VAL_DECORATION_H_
