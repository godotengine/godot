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

#ifndef SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_
#define SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "source/opt/def_use_manager.h"
#include "source/opt/inline_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlineExhaustivePass : public InlinePass {
 public:
  InlineExhaustivePass();
  Status Process() override;

  const char* name() const override { return "inline-entry-points-exhaustive"; }

 private:
  // Exhaustively inline all function calls in func as well as in
  // all code that is inlined into func. Returns the status.
  Status InlineExhaustive(Function* func);

  void Initialize();
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_
