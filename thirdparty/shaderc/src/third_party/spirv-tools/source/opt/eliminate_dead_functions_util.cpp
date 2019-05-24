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

#include "eliminate_dead_functions_util.h"

namespace spvtools {
namespace opt {

namespace eliminatedeadfunctionsutil {

Module::iterator EliminateFunction(IRContext* context,
                                   Module::iterator* func_iter) {
  (*func_iter)
      ->ForEachInst([context](Instruction* inst) { context->KillInst(inst); },
                    true);
  return func_iter->Erase();
}

}  // namespace eliminatedeadfunctionsutil
}  // namespace opt
}  // namespace spvtools
