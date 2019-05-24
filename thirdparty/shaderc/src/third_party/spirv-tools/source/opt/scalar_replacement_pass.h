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

#ifndef SOURCE_OPT_SCALAR_REPLACEMENT_PASS_H_
#define SOURCE_OPT_SCALAR_REPLACEMENT_PASS_H_

#include <cstdio>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/function.h"
#include "source/opt/pass.h"
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class ScalarReplacementPass : public Pass {
 private:
  static const uint32_t kDefaultLimit = 100;

 public:
  ScalarReplacementPass(uint32_t limit = kDefaultLimit)
      : max_num_elements_(limit) {
    name_[0] = '\0';
    strcat(name_, "scalar-replacement=");
    sprintf(&name_[strlen(name_)], "%d", max_num_elements_);
  }

  const char* name() const override { return name_; }

  // Attempts to scalarize all appropriate function scope variables. Returns
  // SuccessWithChange if any change is made.
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Small container for tracking statistics about variables.
  //
  // TODO(alanbaker): Develop some useful heuristics to tune this pass.
  struct VariableStats {
    uint32_t num_partial_accesses;
    uint32_t num_full_accesses;
  };

  // Attempts to scalarize all appropriate function scope variables in
  // |function|. Returns SuccessWithChange if any changes are mode.
  Status ProcessFunction(Function* function);

  // Returns true if |varInst| can be scalarized.
  //
  // Examines the use chain of |varInst| to verify all uses are valid for
  // scalarization.
  bool CanReplaceVariable(const Instruction* varInst) const;

  // Returns true if |typeInst| is an acceptable type to scalarize.
  //
  // Allows all aggregate types except runtime arrays. Additionally, checks the
  // that the number of elements that would be scalarized is within bounds.
  bool CheckType(const Instruction* typeInst) const;

  // Returns true if all the decorations for |varInst| are acceptable for
  // scalarization.
  bool CheckAnnotations(const Instruction* varInst) const;

  // Returns true if all the decorations for |typeInst| are acceptable for
  // scalarization.
  bool CheckTypeAnnotations(const Instruction* typeInst) const;

  // Returns true if the uses of |inst| are acceptable for scalarization.
  //
  // Recursively checks all the uses of |inst|. For |inst| specifically, only
  // allows SpvOpAccessChain, SpvOpInBoundsAccessChain, SpvOpLoad and
  // SpvOpStore. Access chains must have the first index be a compile-time
  // constant. Subsequent uses of access chains (including other access chains)
  // are checked in a more relaxed manner.
  bool CheckUses(const Instruction* inst) const;

  // Helper function for the above |CheckUses|.
  //
  // This version tracks some stats about the current OpVariable. These stats
  // are used to drive heuristics about when to scalarize.
  bool CheckUses(const Instruction* inst, VariableStats* stats) const;

  // Relaxed helper function for |CheckUses|.
  bool CheckUsesRelaxed(const Instruction* inst) const;

  // Transfers appropriate decorations from |source| to |replacements|.
  void TransferAnnotations(const Instruction* source,
                           std::vector<Instruction*>* replacements);

  // Scalarizes |inst| and updates its uses.
  //
  // |inst| must be an OpVariable. It is replaced with an OpVariable for each
  // for element of the composite type. Uses of |inst| are updated as
  // appropriate. If the replacement variables are themselves scalarizable, they
  // get added to |worklist| for further processing. If any replacement
  // variable ends up with no uses it is erased. Returns false if any
  // subsequent access chain is out of bounds.
  bool ReplaceVariable(Instruction* inst, std::queue<Instruction*>* worklist);

  // Returns the underlying storage type for |inst|.
  //
  // |inst| must be an OpVariable. Returns the type that is pointed to by
  // |inst|.
  Instruction* GetStorageType(const Instruction* inst) const;

  // Returns true if the load can be scalarized.
  //
  // |inst| must be an OpLoad. Returns true if |index| is the pointer operand of
  // |inst| and the load is not from volatile memory.
  bool CheckLoad(const Instruction* inst, uint32_t index) const;

  // Returns true if the store can be scalarized.
  //
  // |inst| must be an OpStore. Returns true if |index| is the pointer operand
  // of |inst| and the store is not to volatile memory.
  bool CheckStore(const Instruction* inst, uint32_t index) const;

  // Creates a variable of type |typeId| from the |index|'th element of
  // |varInst|. The new variable is added to |replacements|.
  void CreateVariable(uint32_t typeId, Instruction* varInst, uint32_t index,
                      std::vector<Instruction*>* replacements);

  // Populates |replacements| with a new OpVariable for each element of |inst|.
  //
  // |inst| must be an OpVariable of a composite type. New variables are
  // initialized the same as the corresponding index in |inst|. |replacements|
  // will contain a variable for each element of the composite with matching
  // indexes (i.e. the 0'th element of |inst| is the 0'th entry of
  // |replacements|).
  void CreateReplacementVariables(Instruction* inst,
                                  std::vector<Instruction*>* replacements);

  // Returns the value of an OpConstant of integer type.
  //
  // |constant| must use two or fewer words to generate the value.
  size_t GetConstantInteger(const Instruction* constant) const;

  // Returns the integer literal for |op|.
  size_t GetIntegerLiteral(const Operand& op) const;

  // Returns the array length for |arrayInst|.
  size_t GetArrayLength(const Instruction* arrayInst) const;

  // Returns the number of elements in |type|.
  //
  // |type| must be a vector or matrix type.
  size_t GetNumElements(const Instruction* type) const;

  // Returns true if |id| is a specialization constant.
  //
  // |id| must be registered definition.
  bool IsSpecConstant(uint32_t id) const;

  // Returns an id for a pointer to |id|.
  uint32_t GetOrCreatePointerType(uint32_t id);

  // Creates the initial value for the |index| element of |source| in |newVar|.
  //
  // If there is an initial value for |source| for element |index|, it is
  // appended as an operand on |newVar|. If the initial value is OpUndef, no
  // initial value is added to |newVar|.
  void GetOrCreateInitialValue(Instruction* source, uint32_t index,
                               Instruction* newVar);

  // Replaces the load to the entire composite.
  //
  // Generates a load for each replacement variable and then creates a new
  // composite by combining all of the loads.
  //
  // |load| must be a load.
  void ReplaceWholeLoad(Instruction* load,
                        const std::vector<Instruction*>& replacements);

  // Replaces the store to the entire composite.
  //
  // Generates a composite extract and store for each element in the scalarized
  // variable from the original store data input.
  void ReplaceWholeStore(Instruction* store,
                         const std::vector<Instruction*>& replacements);

  // Replaces an access chain to the composite variable with either a direct use
  // of the appropriate replacement variable or another access chain with the
  // replacement variable as the base and one fewer indexes. Returns false if
  // the chain has an out of bounds access.
  bool ReplaceAccessChain(Instruction* chain,
                          const std::vector<Instruction*>& replacements);

  // Returns a set containing the which components of the result of |inst| are
  // potentially used.  If the return value is |nullptr|, then every components
  // is possibly used.
  std::unique_ptr<std::unordered_set<uint64_t>> GetUsedComponents(
      Instruction* inst);

  // Returns an instruction defining a null constant with type |type_id|.  If
  // one already exists, it is returned.  Otherwise a new one is created.
  Instruction* CreateNullConstant(uint32_t type_id);

  // Maps storage type to a pointer type enclosing that type.
  std::unordered_map<uint32_t, uint32_t> pointee_to_pointer_;

  // Maps type id to OpConstantNull for that type.
  std::unordered_map<uint32_t, uint32_t> type_to_null_;

  // Limit on the number of members in an object that will be replaced.
  // 0 means there is no limit.
  uint32_t max_num_elements_;
  bool IsLargerThanSizeLimit(size_t length) const;
  char name_[55];
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SCALAR_REPLACEMENT_PASS_H_
