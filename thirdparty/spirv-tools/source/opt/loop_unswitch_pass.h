// Copyright (c) 2018 Google LLC.
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

#ifndef SOURCE_OPT_LOOP_UNSWITCH_PASS_H_
#define SOURCE_OPT_LOOP_UNSWITCH_PASS_H_

#include "source/opt/loop_descriptor.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Implements the loop unswitch optimization.
// The loop unswitch hoists invariant "if" statements if the conditions are
// constant within the loop and clones the loop for each branch.
class LoopUnswitchPass : public Pass {
 public:
  const char* name() const override { return "loop-unswitch"; }

  // Processes the given |module|. Returns Status::Failure if errors occur when
  // processing. Returns the corresponding Status::Success if processing is
  // successful to indicate whether changes have been made to the module.
  Pass::Status Process() override;

 private:
  bool ProcessFunction(Function* f);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_UNSWITCH_PASS_H_
