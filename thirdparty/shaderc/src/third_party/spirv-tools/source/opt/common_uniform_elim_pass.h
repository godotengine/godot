// Copyright (c) 2016 The Khronos Group Inc.
// Copyright (c) 2016 Valve Corporation
// Copyright (c) 2016 LunarG Inc.
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

#ifndef SOURCE_OPT_COMMON_UNIFORM_ELIM_PASS_H_
#define SOURCE_OPT_COMMON_UNIFORM_ELIM_PASS_H_

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CommonUniformElimPass : public Pass {
  using cbb_ptr = const BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<BasicBlock*>*(const BasicBlock*)>;

  CommonUniformElimPass();

  const char* name() const override { return "eliminate-common-uniform"; }
  Status Process() override;

 private:
  // Returns true if |opcode| is a non-ptr access chain op
  bool IsNonPtrAccessChain(const SpvOp opcode) const;

  // Returns true if |typeInst| is a sampler or image type or a struct
  // containing one, recursively.
  bool IsSamplerOrImageType(const Instruction* typeInst) const;

  // Returns true if |varId| is a variable containing a sampler or image.
  bool IsSamplerOrImageVar(uint32_t varId) const;

  // Given a load or store pointed at by |ip|, return the top-most
  // non-CopyObj in its pointer operand. Also return the base pointer
  // in |objId|.
  Instruction* GetPtr(Instruction* ip, uint32_t* objId);

  // Return true if variable is uniform
  bool IsUniformVar(uint32_t varId);

  // Given the type id for a struct type, checks if the struct type
  // or any struct member is volatile decorated
  bool IsVolatileStruct(uint32_t type_id);

  // Given an OpAccessChain instruction, return true
  // if the accessed variable belongs to a volatile
  // decorated object or member of a struct type
  bool IsAccessChainToVolatileStructType(const Instruction& AccessChainInst);

  // Given an OpLoad instruction, return true if
  // OpLoad has a Volatile Memory Access flag or if
  // the resulting type is a volatile decorated struct
  bool IsVolatileLoad(const Instruction& loadInst);

  // Return true if any uses of |id| are decorate ops.
  bool HasUnsupportedDecorates(uint32_t id) const;

  // Return true if all uses of |id| are only name or decorate ops.
  bool HasOnlyNamesAndDecorates(uint32_t id) const;

  // Delete inst if it has no uses. Assumes inst has a resultId.
  void DeleteIfUseless(Instruction* inst);

  // Replace all instances of load's id with replId and delete load
  // and its access chain, if any
  Instruction* ReplaceAndDeleteLoad(Instruction* loadInst, uint32_t replId,
                                    Instruction* ptrInst);

  // For the (constant index) access chain ptrInst, create an
  // equivalent load and extract
  void GenACLoadRepl(const Instruction* ptrInst,
                     std::vector<std::unique_ptr<Instruction>>* newInsts,
                     uint32_t* resultId);

  // Return true if all indices are constant
  bool IsConstantIndexAccessChain(Instruction* acp);

  // Convert all uniform access chain loads into load/extract.
  bool UniformAccessChainConvert(Function* func);

  // Compute structured successors for function |func|.
  // A block's structured successors are the blocks it branches to
  // together with its declared merge block if it has one.
  // When order matters, the merge block always appears first.
  // This assures correct depth first search in the presence of early
  // returns and kills. If the successor vector contain duplicates
  // if the merge block, they are safely ignored by DFS.
  //
  // TODO(dnovillo): This pass computes structured successors slightly different
  // than the implementation in class Pass. Can this be re-factored?
  void ComputeStructuredSuccessors(Function* func);

  // Compute structured block order for |func| into |structuredOrder|. This
  // order has the property that dominators come before all blocks they
  // dominate and merge blocks come after all blocks that are in the control
  // constructs of their header.
  //
  // TODO(dnovillo): This pass computes structured order slightly different
  // than the implementation in class Pass. Can this be re-factored?
  void ComputeStructuredOrder(Function* func, std::list<BasicBlock*>* order);

  // Eliminate loads of uniform variables which have previously been loaded.
  // If first load is in control flow, move it to first block of function.
  // Most effective if preceded by UniformAccessChainRemoval().
  bool CommonUniformLoadElimination(Function* func);

  // Eliminate loads of uniform sampler and image variables which have
  // previously
  // been loaded in the same block for types whose loads cannot cross blocks.
  bool CommonUniformLoadElimBlock(Function* func);

  // Eliminate duplicated extracts of same id. Extract may be moved to same
  // block as the id definition. This is primarily intended for extracts
  // from uniform loads. Most effective if preceded by
  // CommonUniformLoadElimination().
  bool CommonExtractElimination(Function* func);

  // For function |func|, first change all uniform constant index
  // access chain loads into equivalent composite extracts. Then consolidate
  // identical uniform loads into one uniform load. Finally, consolidate
  // identical uniform extracts into one uniform extract. This may require
  // moving a load or extract to a point which dominates all uses.
  // Return true if func is modified.
  //
  // This pass requires the function to have structured control flow ie shader
  // capability. It also requires logical addressing ie Addresses capability
  // is not enabled. It also currently does not support any extensions.
  //
  // This function currently only optimizes loads with a single index.
  bool EliminateCommonUniform(Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  // Return true if |op| is a decorate for non-type instruction
  inline bool IsNonTypeDecorate(uint32_t op) const {
    return (op == SpvOpDecorate || op == SpvOpDecorateId);
  }

  // Return true if |inst| is an instruction that loads uniform variable and
  // can be replaced with other uniform load instruction.
  bool IsUniformLoadToBeRemoved(Instruction* inst) {
    if (inst->opcode() == SpvOpLoad) {
      uint32_t varId;
      Instruction* ptrInst = GetPtr(inst, &varId);
      if (ptrInst->opcode() == SpvOpVariable && IsUniformVar(varId) &&
          !IsSamplerOrImageVar(varId) &&
          !HasUnsupportedDecorates(inst->result_id()) && !IsVolatileLoad(*inst))
        return true;
    }
    return false;
  }

  void Initialize();
  Pass::Status ProcessImpl();

  // Map from uniform variable id to its common load id
  std::unordered_map<uint32_t, uint32_t> uniform2load_id_;

  // Map of extract composite ids to map of indices to insts
  // TODO(greg-lunarg): Consider std::vector.
  std::unordered_map<uint32_t,
                     std::unordered_map<uint32_t, std::list<Instruction*>>>
      comp2idx2inst_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;

  // Map from block to its structured successor blocks. See
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>
      block2structured_succs_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_COMMON_UNIFORM_ELIM_PASS_H_
