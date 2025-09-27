// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_OPT_WRAP_OPKILL_H_
#define SOURCE_OPT_WRAP_OPKILL_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class WrapOpKill : public Pass {
 public:
  WrapOpKill() : void_type_id_(0) {}

  const char* name() const override { return "wrap-opkill"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisBuiltinVarId |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Replaces the OpKill or OpTerminateInvocation instruction |inst| with a
  // function call to a function that contains a single instruction, a clone of
  // |inst|.  An OpUnreachable instruction will be placed after the function
  // call. Return true if successful.
  bool ReplaceWithFunctionCall(Instruction* inst);

  // Returns the id of the void type.
  uint32_t GetVoidTypeId();

  // Returns the id of the function type for a void function with no parameters.
  uint32_t GetVoidFunctionTypeId();

  // Return the id of a function that has return type void, has no parameters,
  // and contains a single instruction, which is |opcode|, either OpKill or
  // OpTerminateInvocation.  Returns 0 if the function could not be generated.
  uint32_t GetKillingFuncId(spv::Op opcode);

  // Returns the id of the return type for the function that contains |inst|.
  // Returns 0 if |inst| is not in a function.
  uint32_t GetOwningFunctionsReturnType(Instruction* inst);

  // The id of the void type.  If its value is 0, then the void type has not
  // been found or created yet.
  uint32_t void_type_id_;

  // The function that is a single instruction, which is an OpKill.  The
  // function has a void return type and takes no parameters. If the function is
  // |nullptr|, then the function has not been generated.
  std::unique_ptr<Function> opkill_function_;
  // The function that is a single instruction, which is an
  // OpTerminateInvocation. The function has a void return type and takes no
  // parameters. If the function is |nullptr|, then the function has not been
  // generated.
  std::unique_ptr<Function> opterminateinvocation_function_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_WRAP_OPKILL_H_
