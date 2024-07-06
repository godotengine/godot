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

#ifndef SOURCE_OPT_STRENGTH_REDUCTION_PASS_H_
#define SOURCE_OPT_STRENGTH_REDUCTION_PASS_H_

#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class StrengthReductionPass : public Pass {
 public:
  const char* name() const override { return "strength-reduction"; }
  Status Process() override;

 private:
  // Replaces multiple by power of 2 with an equivalent bit shift.
  // Returns true if something changed.
  bool ReplaceMultiplyByPowerOf2(BasicBlock::iterator*);

  // Scan the types and constants in the module looking for the integer
  // types that we are
  // interested in.  The shift operation needs a small unsigned integer.  We
  // need to find
  // them or create them.  We do not want duplicates.
  void FindIntTypesAndConstants();

  // Get the id for the given constant.  If it does not exist, it will be
  // created.  The parameter must be between 0 and 32 inclusive.
  uint32_t GetConstantId(uint32_t);

  // Replaces certain instructions in function bodies with presumably cheaper
  // ones. Returns true if something changed.
  bool ScanFunctions();

  // Type ids for the types of interest, or 0 if they do not exist.
  uint32_t int32_type_id_;
  uint32_t uint32_type_id_;

  // constant_ids[i] is the id for unsigned integer constant i.
  // We set the limit at 32 because a bit shift of a 32-bit integer does not
  // need a value larger than 32.
  uint32_t constant_ids_[33];
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_STRENGTH_REDUCTION_PASS_H_
