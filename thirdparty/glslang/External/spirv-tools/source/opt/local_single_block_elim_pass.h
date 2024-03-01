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

#ifndef SOURCE_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_
#define SOURCE_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalSingleBlockLoadStoreElimPass : public MemPass {
 public:
  LocalSingleBlockLoadStoreElimPass();

  const char* name() const override { return "eliminate-local-single-block"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Return true if all uses of |varId| are only through supported reference
  // operations ie. loads and store. Also cache in supported_ref_ptrs_.
  // TODO(dnovillo): This function is replicated in other passes and it's
  // slightly different in every pass. Is it possible to make one common
  // implementation?
  bool HasOnlySupportedRefs(uint32_t varId);

  // On all entry point functions, within each basic block, eliminate
  // loads and stores to function variables where possible. For
  // loads, if previous load or store to same variable, replace
  // load id with previous id and delete load. Finally, check if
  // remaining stores are useless, and delete store and variable
  // where possible. Assumes logical addressing.
  bool LocalSingleBlockLoadStoreElim(Function* func);

  // Initialize extensions allowlist
  void InitExtensions();

  // Return true if all extensions in this module are supported by this pass.
  bool AllExtensionsSupported() const;

  void Initialize();
  Pass::Status ProcessImpl();

  // Map from function scope variable to a store of that variable in the
  // current block whose value is currently valid. This map is cleared
  // at the start of each block and incrementally updated as the block
  // is scanned. The stores are candidates for elimination. The map is
  // conservatively cleared when a function call is encountered.
  std::unordered_map<uint32_t, Instruction*> var2store_;

  // Map from function scope variable to a load of that variable in the
  // current block whose value is currently valid. This map is cleared
  // at the start of each block and incrementally updated as the block
  // is scanned. The stores are candidates for elimination. The map is
  // conservatively cleared when a function call is encountered.
  std::unordered_map<uint32_t, Instruction*> var2load_;

  // Set of variables whose most recent store in the current block cannot be
  // deleted, for example, if there is a load of the variable which is
  // dependent on the store and is not replaced and deleted by this pass,
  // for example, a load through an access chain. A variable is removed
  // from this set each time a new store of that variable is encountered.
  std::unordered_set<uint32_t> pinned_vars_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_allowlist_;

  // Variables that are only referenced by supported operations for this
  // pass ie. loads and stores.
  std::unordered_set<uint32_t> supported_ref_ptrs_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_
