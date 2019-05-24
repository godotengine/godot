// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef SOURCE_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_
#define SOURCE_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalAccessChainConvertPass : public MemPass {
 public:
  LocalAccessChainConvertPass();

  const char* name() const override { return "convert-local-access-chains"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

  using ProcessFunction = std::function<bool(Function*)>;

 private:
  // Return true if all refs through |ptrId| are only loads or stores and
  // cache ptrId in supported_ref_ptrs_. TODO(dnovillo): This function is
  // replicated in other passes and it's slightly different in every pass. Is it
  // possible to make one common implementation?
  bool HasOnlySupportedRefs(uint32_t ptrId);

  // Search |func| and cache function scope variables of target type that are
  // not accessed with non-constant-index access chains. Also cache non-target
  // variables.
  void FindTargetVars(Function* func);

  // Build instruction from |opcode|, |typeId|, |resultId|, and |in_opnds|.
  // Append to |newInsts|.
  void BuildAndAppendInst(SpvOp opcode, uint32_t typeId, uint32_t resultId,
                          const std::vector<Operand>& in_opnds,
                          std::vector<std::unique_ptr<Instruction>>* newInsts);

  // Build load of variable in |ptrInst| and append to |newInsts|.
  // Return var in |varId| and its pointee type in |varPteTypeId|.
  uint32_t BuildAndAppendVarLoad(
      const Instruction* ptrInst, uint32_t* varId, uint32_t* varPteTypeId,
      std::vector<std::unique_ptr<Instruction>>* newInsts);

  // Append literal integer operands to |in_opnds| corresponding to constant
  // integer operands from access chain |ptrInst|. Assumes all indices in
  // access chains are OpConstant.
  void AppendConstantOperands(const Instruction* ptrInst,
                              std::vector<Operand>* in_opnds);

  // Create a load/insert/store equivalent to a store of
  // |valId| through (constant index) access chaing |ptrInst|.
  // Append to |newInsts|.
  void GenAccessChainStoreReplacement(
      const Instruction* ptrInst, uint32_t valId,
      std::vector<std::unique_ptr<Instruction>>* newInsts);

  // For the (constant index) access chain |address_inst|, create an
  // equivalent load and extract that replaces |original_load|.  The result id
  // of the extract will be the same as the original result id of
  // |original_load|.
  void ReplaceAccessChainLoad(const Instruction* address_inst,
                              Instruction* original_load);

  // Return true if all indices of access chain |acp| are OpConstant integers
  bool IsConstantIndexAccessChain(const Instruction* acp) const;

  // Identify all function scope variables of target type which are
  // accessed only with loads, stores and access chains with constant
  // indices. Convert all loads and stores of such variables into equivalent
  // loads, stores, extracts and inserts. This unifies access to these
  // variables to a single mode and simplifies analysis and optimization.
  // See IsTargetType() for targeted types.
  //
  // Nested access chains and pointer access chains are not currently
  // converted.
  bool ConvertLocalAccessChains(Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize();
  Pass::Status ProcessImpl();

  // Variables with only supported references, ie. loads and stores using
  // variable directly or through non-ptr access chains.
  std::unordered_set<uint32_t> supported_ref_ptrs_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_
