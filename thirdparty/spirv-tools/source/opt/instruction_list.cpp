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

#include "source/opt/instruction_list.h"

namespace spvtools {
namespace opt {

InstructionList::iterator InstructionList::iterator::InsertBefore(
    std::vector<std::unique_ptr<Instruction>>&& list) {
  Instruction* first_node = list.front().get();
  for (auto& i : list) {
    i.release()->InsertBefore(node_);
  }
  list.clear();
  return iterator(first_node);
}

InstructionList::iterator InstructionList::iterator::InsertBefore(
    std::unique_ptr<Instruction>&& i) {
  i.get()->InsertBefore(node_);
  return iterator(i.release());
}
}  // namespace opt
}  // namespace spvtools
