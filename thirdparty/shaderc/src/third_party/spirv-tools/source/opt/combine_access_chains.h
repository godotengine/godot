// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_OPT_COMBINE_ACCESS_CHAINS_H_
#define SOURCE_OPT_COMBINE_ACCESS_CHAINS_H_

#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CombineAccessChains : public Pass {
 public:
  const char* name() const override { return "combine-access-chains"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

 private:
  // Combine access chains in |function|. Blocks are processed in reverse
  // post-order. Returns true if the function is modified.
  bool ProcessFunction(Function& function);

  // Combines an access chain (normal, in bounds or pointer) |inst| if its base
  // pointer is another access chain. Returns true if the access chain was
  // modified.
  bool CombineAccessChain(Instruction* inst);

  // Returns the value of |constant_inst| as a uint32_t.
  uint32_t GetConstantValue(const analysis::Constant* constant_inst);

  // Returns the array stride of |inst|'s type.
  uint32_t GetArrayStride(const Instruction* inst);

  // Returns the type by resolving the index operands |inst|. |inst| must be an
  // access chain instruction.
  const analysis::Type* GetIndexedType(Instruction* inst);

  // Populates |new_operands| with the operands for the combined access chain.
  // Returns false if the access chains cannot be combined.
  bool CreateNewInputOperands(Instruction* ptr_input, Instruction* inst,
                              std::vector<Operand>* new_operands);

  // Combines the last index of |ptr_input| with the element operand of |inst|.
  // Adds the combined operand to |new_operands|.
  bool CombineIndices(Instruction* ptr_input, Instruction* inst,
                      std::vector<Operand>* new_operands);

  // Returns the opcode to use for the combined access chain.
  SpvOp UpdateOpcode(SpvOp base_opcode, SpvOp input_opcode);

  // Returns true if |opcode| is a pointer access chain.
  bool IsPtrAccessChain(SpvOp opcode);

  // Returns true if |inst| (an access chain) has 64-bit indices.
  bool Has64BitIndices(Instruction* inst);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_COMBINE_ACCESS_CHAINS_H_
