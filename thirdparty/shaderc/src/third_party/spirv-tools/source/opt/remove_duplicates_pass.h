// Copyright (c) 2017 Pierre Moreau
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

#ifndef SOURCE_OPT_REMOVE_DUPLICATES_PASS_H_
#define SOURCE_OPT_REMOVE_DUPLICATES_PASS_H_

#include <unordered_map>
#include <vector>

#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

using IdDecorationsList =
    std::unordered_map<uint32_t, std::vector<Instruction*>>;

// See optimizer.hpp for documentation.
class RemoveDuplicatesPass : public Pass {
 public:
  const char* name() const override { return "remove-duplicates"; }
  Status Process() override;

  // TODO(pierremoreau): Move this function somewhere else (e.g. pass.h or
  // within the type manager)
  // Returns whether two types are equal, and have the same decorations.
  static bool AreTypesEqual(const Instruction& inst1, const Instruction& inst2,
                            IRContext* context);

 private:
  // Remove duplicate capabilities from the module
  //
  // Returns true if the module was modified, false otherwise.
  bool RemoveDuplicateCapabilities() const;
  // Remove duplicate extended instruction imports from the module
  //
  // Returns true if the module was modified, false otherwise.
  bool RemoveDuplicatesExtInstImports() const;
  // Remove duplicate types from the module
  //
  // Returns true if the module was modified, false otherwise.
  bool RemoveDuplicateTypes() const;
  // Remove duplicate decorations from the module
  //
  // Returns true if the module was modified, false otherwise.
  bool RemoveDuplicateDecorations() const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REMOVE_DUPLICATES_PASS_H_
