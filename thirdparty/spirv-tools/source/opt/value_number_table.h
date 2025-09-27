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

#ifndef SOURCE_OPT_VALUE_NUMBER_TABLE_H_
#define SOURCE_OPT_VALUE_NUMBER_TABLE_H_

#include <cstdint>
#include <unordered_map>

#include "source/opt/instruction.h"

namespace spvtools {
namespace opt {

class IRContext;

// Returns true if the two instructions compute the same value.  Used by the
// value number table to compare two instructions.
class ComputeSameValue {
 public:
  bool operator()(const Instruction& lhs, const Instruction& rhs) const;
};

// The hash function used in the value number table.
class ValueTableHash {
 public:
  std::size_t operator()(const Instruction& inst) const;
};

// This class implements the value number analysis.  It is using a hash-based
// approach to value numbering.  It is essentially doing dominator-tree value
// numbering described in
//
//   Preston Briggs, Keith D. Cooper, and L. Taylor Simpson. 1997. Value
//   numbering. Softw. Pract. Exper. 27, 6 (June 1997), 701-724.
//   https://www.cs.rice.edu/~keith/Promo/CRPC-TR94517.pdf.gz
//
// The main difference is that because we do not perform redundancy elimination
// as we build the value number table, we do not have to deal with cleaning up
// the scope.
class ValueNumberTable {
 public:
  ValueNumberTable(IRContext* ctx) : context_(ctx), next_value_number_(1) {
    BuildDominatorTreeValueNumberTable();
  }

  // Returns the value number of the value computed by |inst|.  |inst| must have
  // a result id that will hold the computed value.  If no value number has been
  // assigned to the result id, then the return value is 0.
  uint32_t GetValueNumber(Instruction* inst) const;

  // Returns the value number of the value contain in |id|.  Returns 0 if it
  // has not been assigned a value number.
  uint32_t GetValueNumber(uint32_t id) const;

  IRContext* context() const { return context_; }

 private:
  // Assigns a value number to every result id in the module.
  void BuildDominatorTreeValueNumberTable();

  // Returns the new value number.
  uint32_t TakeNextValueNumber() { return next_value_number_++; }

  // Assigns a new value number to the result of |inst| if it does not already
  // have one.  Return the value number for |inst|.  |inst| must have a result
  // id.
  uint32_t AssignValueNumber(Instruction* inst);

  std::unordered_map<Instruction, uint32_t, ValueTableHash, ComputeSameValue>
      instruction_to_value_;
  std::unordered_map<uint32_t, uint32_t> id_to_value_;
  IRContext* context_;
  uint32_t next_value_number_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_VALUE_NUMBER_TABLE_H_
