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

#ifndef SOURCE_OPT_REPLACE_INVALID_OPC_H_
#define SOURCE_OPT_REPLACE_INVALID_OPC_H_

#include <string>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// This pass will runs on shader modules only.  It will replace the result of
// instructions that are valid for shader modules, but not the current shader
// stage, with a constant value.  If the instruction does not have a return
// value, the instruction will simply be deleted.
class ReplaceInvalidOpcodePass : public Pass {
 public:
  const char* name() const override { return "replace-invalid-opcode"; }
  Status Process() override;

 private:
  // Returns the execution model that is used by every entry point in the
  // module. If more than one execution model is used in the module, then the
  // return value is spv::ExecutionModel::Max.
  spv::ExecutionModel GetExecutionModel();

  // Replaces all instructions in |function| that are invalid with execution
  // model |mode|, but valid for another shader model, with a special constant
  // value.  See |GetSpecialConstant|.
  bool RewriteFunction(Function* function, spv::ExecutionModel mode);

  // Returns true if |inst| is valid for fragment shaders only.
  bool IsFragmentShaderOnlyInstruction(Instruction* inst);

  // Replaces all uses of the result of |inst|, if there is one, with the id of
  // a special constant.  Then |inst| is killed.  |inst| cannot be a block
  // terminator because the basic block will then become invalid.  |inst| is no
  // longer valid after calling this function.
  void ReplaceInstruction(Instruction* inst, const char* source,
                          uint32_t line_number, uint32_t column_number);

  // Returns the id of a constant with type |type_id|.  The type must be an
  // integer, float, or vector.  For scalar types, the hex representation of the
  // constant will be the concatenation of 0xDEADBEEF with itself until the
  // width of the type has been reached. For a vector, each element of the
  // constant will be constructed the same way.
  uint32_t GetSpecialConstant(uint32_t type_id);
  std::string BuildWarningMessage(spv::Op opcode);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REPLACE_INVALID_OPC_H_
