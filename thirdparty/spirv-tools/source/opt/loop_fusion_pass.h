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

#ifndef SOURCE_OPT_LOOP_FUSION_PASS_H_
#define SOURCE_OPT_LOOP_FUSION_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Implements a loop fusion pass.
// This pass will look for adjacent loops that are compatible and legal to be
// fused. It will fuse all such loops as long as the register usage for the
// fused loop stays under the threshold defined by |max_registers_per_loop|.
class LoopFusionPass : public Pass {
 public:
  explicit LoopFusionPass(size_t max_registers_per_loop)
      : Pass(), max_registers_per_loop_(max_registers_per_loop) {}

  const char* name() const override { return "loop-fusion"; }

  // Processes the given |module|. Returns Status::Failure if errors occur when
  // processing. Returns the corresponding Status::Success if processing is
  // successful to indicate whether changes have been made to the module.
  Status Process() override;

 private:
  // Fuse loops in |function| if compatible, legal and the fused loop won't use
  // too many registers.
  bool ProcessFunction(Function* function);

  // The maximum number of registers a fused loop is allowed to use.
  size_t max_registers_per_loop_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_FUSION_PASS_H_
