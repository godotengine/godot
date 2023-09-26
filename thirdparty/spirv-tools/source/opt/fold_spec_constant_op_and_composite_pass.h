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

#ifndef SOURCE_OPT_FOLD_SPEC_CONSTANT_OP_AND_COMPOSITE_PASS_H_
#define SOURCE_OPT_FOLD_SPEC_CONSTANT_OP_AND_COMPOSITE_PASS_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "source/opt/constants.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class FoldSpecConstantOpAndCompositePass : public Pass {
 public:
  FoldSpecConstantOpAndCompositePass() = default;

  const char* name() const override { return "fold-spec-const-op-composite"; }

  // Iterates through the types-constants-globals section of the given module,
  // finds the Spec Constants defined with OpSpecConstantOp and
  // OpSpecConstantComposite instructions. If the result value of those spec
  // constants can be folded, fold them to their corresponding normal constants.
  Status Process() override;

 private:
  // Processes the OpSpecConstantOp instruction pointed by the given
  // instruction iterator, folds it to normal constants if possible. Returns
  // true if the spec constant is folded to normal constants. New instructions
  // will be inserted before the OpSpecConstantOp instruction pointed by the
  // instruction iterator. The instruction iterator, which is passed by
  // pointer, will still point to the original OpSpecConstantOp instruction. If
  // folding is done successfully, the original OpSpecConstantOp instruction
  // will be changed to Nop and new folded instruction will be inserted before
  // it.
  bool ProcessOpSpecConstantOp(Module::inst_iterator* pos);

  // Returns the result of folding the OpSpecConstantOp instruction
  // |inst_iter_ptr| using the instruction folder.
  Instruction* FoldWithInstructionFolder(Module::inst_iterator* inst_iter_ptr);

  // Try to fold the OpSpecConstantOp <component wise operations> instruction
  // pointed by the given instruction iterator to a normal constant defining
  // instruction. Returns the pointer to the new constant defining instruction
  // if succeeded, otherwise return nullptr.
  Instruction* DoComponentWiseOperation(Module::inst_iterator* inst_iter_ptr);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FOLD_SPEC_CONSTANT_OP_AND_COMPOSITE_PASS_H_
