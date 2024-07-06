// Copyright (c) 2022 Google LLC
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

#ifndef SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_
#define SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SpreadVolatileSemantics : public Pass {
 public:
  SpreadVolatileSemantics() {}

  const char* name() const override { return "spread-volatile-semantics"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisDecorations |
           IRContext::kAnalysisInstrToBlockMapping;
  }

 private:
  // Returns true if it does not have an execution model. Linkage shaders do not
  // have an execution model.
  bool HasNoExecutionModel() {
    return get_module()->entry_points().empty() &&
           context()->get_feature_mgr()->HasCapability(
               spv::Capability::Linkage);
  }

  // Iterates interface variables and spreads the Volatile semantics if it has
  // load instructions for the Volatile semantics.
  Pass::Status SpreadVolatileSemanticsToVariables(
      const bool is_vk_memory_model_enabled);

  // Returns whether |var_id| is the result id of a target builtin variable for
  // the volatile semantics for |execution_model| based on the Vulkan spec
  // VUID-StandaloneSpirv-VulkanMemoryModel-04678 or
  // VUID-StandaloneSpirv-VulkanMemoryModel-04679.
  bool IsTargetForVolatileSemantics(uint32_t var_id,
                                    spv::ExecutionModel execution_model);

  // Collects interface variables that need the volatile semantics.
  // |is_vk_memory_model_enabled| is true if VulkanMemoryModel capability is
  // enabled.
  void CollectTargetsForVolatileSemantics(
      const bool is_vk_memory_model_enabled);

  // Reports an error if an interface variable is used by two entry points and
  // it needs the Volatile decoration for one but not for another. Returns true
  // if the error must be reported.
  bool HasInterfaceInConflictOfVolatileSemantics();

  // Returns whether the variable whose result is |var_id| is used by a
  // non-volatile load or a pointer to it is used by a non-volatile load in
  // |entry_point| or not.
  bool IsTargetUsedByNonVolatileLoadInEntryPoint(uint32_t var_id,
                                                 Instruction* entry_point);

  // Visits load instructions of pointers to variable whose result id is
  // |var_id| if the load instructions are in reachable functions from entry
  // points. |handle_load| is a function to do some actions for the load
  // instructions. Finishes the traversal and returns false if |handle_load|
  // returns false for a load instruction. Otherwise, returns true after running
  // |handle_load| for all the load instructions.
  bool VisitLoadsOfPointersToVariableInEntries(
      uint32_t var_id, const std::function<bool(Instruction*)>& handle_load,
      const std::unordered_set<uint32_t>& function_ids);

  // Sets Memory Operands of OpLoad instructions that load |var| or pointers
  // of |var| as Volatile if the function id of the OpLoad instruction is
  // included in |entry_function_ids|.
  void SetVolatileForLoadsInEntries(
      Instruction* var, const std::unordered_set<uint32_t>& entry_function_ids);

  // Adds OpDecorate Volatile for |var| if it does not exist.
  void DecorateVarWithVolatile(Instruction* var);

  // Returns a set of entry function ids to spread the volatile semantics for
  // the variable with the result id |var_id|.
  std::unordered_set<uint32_t> EntryFunctionsToSpreadVolatileSemanticsForVar(
      uint32_t var_id) {
    auto itr = var_ids_to_entry_fn_for_volatile_semantics_.find(var_id);
    if (itr == var_ids_to_entry_fn_for_volatile_semantics_.end()) return {};
    return itr->second;
  }

  // Specifies that we have to spread the volatile semantics for the
  // variable with the result id |var_id| for the entry point |entry_point|.
  void MarkVolatileSemanticsForVariable(uint32_t var_id,
                                        Instruction* entry_point);

  // Result ids of variables to entry function ids for the volatile semantics
  // spread.
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>>
      var_ids_to_entry_fn_for_volatile_semantics_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_
