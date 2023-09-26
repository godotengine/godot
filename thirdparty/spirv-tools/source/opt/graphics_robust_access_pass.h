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

#ifndef SOURCE_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_
#define SOURCE_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_

#include <map>
#include <unordered_map>

#include "constants.h"
#include "def_use_manager.h"
#include "instruction.h"
#include "module.h"
#include "pass.h"
#include "source/diagnostic.h"
#include "type_manager.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class GraphicsRobustAccessPass : public Pass {
 public:
  GraphicsRobustAccessPass();
  const char* name() const override { return "graphics-robust-access"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes |
           IRContext::kAnalysisIdToFuncMapping;
  }

 private:
  // Records failure for the current module, and returns a stream
  // that can be used to provide user error information to the message
  // consumer.
  spvtools::DiagnosticStream Fail();

  // Returns SPV_SUCCESS if this pass can correctly process the module,
  // as far as we can tell from capabilities and the memory model.
  // Otherwise logs a message and returns a failure code.
  spv_result_t IsCompatibleModule();

  // Transform the current module, if possible. Failure and modification
  // status is recorded in the |_| member. On failure, error information is
  // posted to the message consumer.  The return value has no significance.
  spv_result_t ProcessCurrentModule();

  // Process the given function.  Updates the state value |_|.  Returns true
  // if the module was modified.  This can log a failure.
  bool ProcessAFunction(opt::Function*);

  // Clamps indices in the OpAccessChain or OpInBoundsAccessChain instruction
  // |access_chain|. Inserts instructions before the given instruction.  Updates
  // analyses and records that the module is modified.  This can log a failure.
  void ClampIndicesForAccessChain(Instruction* access_chain);

  // Returns the id of the instruction importing the "GLSL.std.450" extended
  // instruction set. If it does not yet exist, the import instruction is
  // created and inserted into the module, and updates |_.modified| and
  // |_.glsl_insts_id|.
  uint32_t GetGlslInsts();

  // Returns an instruction which is constant with the given value of the given
  // type. Ignores any value bits beyond the width of the type.
  Instruction* GetValueForType(uint64_t value, const analysis::Integer* type);

  // Converts an integer value to an unsigned wider integer type, using either
  // sign extension or zero extension.  The new instruction is inserted
  // immediately before |before_inst|, and is analyzed for definitions and uses.
  // Returns the newly inserted instruction.  Assumes the |value| is an integer
  // scalar of a narrower type than |bit_width| bits.
  Instruction* WidenInteger(bool sign_extend, uint32_t bit_width,
                            Instruction* value, Instruction* before_inst);

  // Returns a new instruction that invokes the UMin GLSL.std.450 extended
  // instruction with the two given operands.  That is, the result of the
  // instruction is:
  //  - |x| if |x| is unsigned-less than |y|
  //  - |y| otherwise
  // We assume that |x| and |y| are scalar integer types with the same
  // width.  The instruction is inserted before |where|.
  opt::Instruction* MakeUMinInst(const analysis::TypeManager& tm,
                                 Instruction* x, Instruction* y,
                                 Instruction* where);

  // Returns a new instruction that invokes the SClamp GLSL.std.450 extended
  // instruction with the three given operands.  That is, the result of the
  // instruction is:
  //  - |min| if |x| is signed-less than |min|
  //  - |max| if |x| is signed-more than |max|
  //  - |x| otherwise.
  // We assume that |min| is signed-less-or-equal to |max|, and that the
  // operands all have the same scalar integer type.  The instruction is
  // inserted before |where|.
  opt::Instruction* MakeSClampInst(const analysis::TypeManager& tm,
                                   Instruction* x, Instruction* min,
                                   Instruction* max, Instruction* where);

  // Returns a new instruction which evaluates to the length the runtime array
  // referenced by the access chain at the specified index.  The instruction is
  // inserted before the access chain instruction.  Returns a null pointer in
  // some cases if assumptions are violated (rather than asserting out).
  opt::Instruction* MakeRuntimeArrayLengthInst(Instruction* access_chain,
                                               uint32_t operand_index);

  // Clamps the coordinate for an OpImageTexelPointer so it stays within
  // the bounds of the size of the image.  Updates analyses and records that
  // the module is modified.  Returns a status code to indicate success
  // or failure.  If assumptions are not met, returns an error status code
  // and emits a diagnostic.
  spv_result_t ClampCoordinateForImageTexelPointer(
      opt::Instruction* image_texel_pointer);

  // Gets the instruction that defines the given id.
  opt::Instruction* GetDef(uint32_t id) {
    return context()->get_def_use_mgr()->GetDef(id);
  }

  // Returns a new instruction inserted before |where_inst|, and created from
  // the remaining arguments. Registers the definitions and uses of the new
  // instruction and also records its block.
  opt::Instruction* InsertInst(opt::Instruction* where_inst, spv::Op opcode,
                               uint32_t type_id, uint32_t result_id,
                               const Instruction::OperandList& operands);

  // State required for the current module.
  struct PerModuleState {
    // This pass modified the module.
    bool modified = false;
    // True if there is an error processing the current module, e.g. if
    // preconditions are not met.
    bool failed = false;
    // The id of the GLSL.std.450 extended instruction set.  Zero if it does
    // not exist.
    uint32_t glsl_insts_id = 0;
  } module_status_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_
