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

#ifndef SOURCE_OPT_LOOP_UTILS_H_
#define SOURCE_OPT_LOOP_UTILS_H_

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"

namespace spvtools {

namespace opt {

// Class to gather some metrics about a Region Of Interest (ROI).
// So far it counts the number of instructions in a ROI (excluding debug
// and label instructions) per basic block and in total.
struct CodeMetrics {
  void Analyze(const Loop& loop);

  // The number of instructions per basic block in the ROI.
  std::unordered_map<uint32_t, size_t> block_sizes_;

  // Number of instruction in the ROI.
  size_t roi_size_;
};

// LoopUtils is used to encapsulte loop optimizations and from the passes which
// use them. Any pass which needs a loop optimization should do it through this
// or through a pass which is using this.
class LoopUtils {
 public:
  // Holds a auxiliary results of the loop cloning procedure.
  struct LoopCloningResult {
    using ValueMapTy = std::unordered_map<uint32_t, uint32_t>;
    using BlockMapTy = std::unordered_map<uint32_t, BasicBlock*>;
    using PtrMap = std::unordered_map<Instruction*, Instruction*>;

    PtrMap ptr_map_;

    // Mapping between the original loop ids and the new one.
    ValueMapTy value_map_;
    // Mapping between original loop blocks to the cloned one.
    BlockMapTy old_to_new_bb_;
    // Mapping between the cloned loop blocks to original one.
    BlockMapTy new_to_old_bb_;
    // List of cloned basic block.
    std::vector<std::unique_ptr<BasicBlock>> cloned_bb_;
  };

  LoopUtils(IRContext* context, Loop* loop)
      : context_(context),
        loop_desc_(
            context->GetLoopDescriptor(loop->GetHeaderBlock()->GetParent())),
        loop_(loop),
        function_(*loop_->GetHeaderBlock()->GetParent()) {}

  // The converts the current loop to loop closed SSA form.
  // In the loop closed SSA, all loop exiting values go through a dedicated Phi
  // instruction. For instance:
  //
  // for (...) {
  //   A1 = ...
  //   if (...)
  //     A2 = ...
  //   A = phi A1, A2
  // }
  // ... = op A ...
  //
  // Becomes
  //
  // for (...) {
  //   A1 = ...
  //   if (...)
  //     A2 = ...
  //   A = phi A1, A2
  // }
  // C = phi A
  // ... = op C ...
  //
  // This makes some loop transformations (such as loop unswitch) simpler
  // (removes the needs to take care of exiting variables).
  void MakeLoopClosedSSA();

  // Create dedicate exit basic block. This ensure all exit basic blocks has the
  // loop as sole predecessors.
  // By construction, structured control flow already has a dedicated exit
  // block.
  // Preserves: CFG, def/use and instruction to block mapping.
  void CreateLoopDedicatedExits();

  // Clone |loop_| and remap its instructions. Newly created blocks
  // will be added to the |cloning_result.cloned_bb_| list, correctly ordered to
  // be inserted into a function.
  // It is assumed that |ordered_loop_blocks| is compatible with the result of
  // |Loop::ComputeLoopStructuredOrder|. If the preheader and merge block are in
  // the list they will also be cloned. If not, the resulting loop will share
  // them with the original loop.
  // The function preserves the def/use, cfg and instr to block analyses.
  // The cloned loop nest will be added to the loop descriptor and will have
  // ownership.
  Loop* CloneLoop(LoopCloningResult* cloning_result,
                  const std::vector<BasicBlock*>& ordered_loop_blocks) const;
  // Clone |loop_| and remap its instructions, as above. Overload to compute
  // loop block ordering within method rather than taking in as parameter.
  Loop* CloneLoop(LoopCloningResult* cloning_result) const;

  // Clone the |loop_| and make the new loop branch to the second loop on exit.
  Loop* CloneAndAttachLoopToHeader(LoopCloningResult* cloning_result);

  // Perform a partial unroll of |loop| by given |factor|. This will copy the
  // body of the loop |factor| times. So a |factor| of one would give a new loop
  // with the original body plus one unrolled copy body.
  bool PartiallyUnroll(size_t factor);

  // Fully unroll |loop|.
  bool FullyUnroll();

  // This function validates that |loop| meets the assumptions made by the
  // implementation of the loop unroller. As the implementation accommodates
  // more types of loops this function can reduce its checks.
  //
  // The conditions checked to ensure the loop can be unrolled are as follows:
  // 1. That the loop is in structured order.
  // 2. That the continue block is a branch to the header.
  // 3. That the only phi used in the loop is the induction variable.
  //  TODO(stephen@codeplay.com): This is a temporary measure, after the loop is
  //  converted into LCSAA form and has a single entry and exit we can rewrite
  //  the other phis.
  // 4. That this is an inner most loop, or that loops contained within this
  // loop have already been fully unrolled.
  // 5. That each instruction in the loop is only used within the loop.
  // (Related to the above phi condition).
  bool CanPerformUnroll();

  // Maintains the loop descriptor object after the unroll functions have been
  // called, otherwise the analysis should be invalidated.
  void Finalize();

  // Returns the context associate to |loop_|.
  IRContext* GetContext() { return context_; }
  // Returns the loop descriptor owning |loop_|.
  LoopDescriptor* GetLoopDescriptor() { return loop_desc_; }
  // Returns the loop on which the object operates on.
  Loop* GetLoop() const { return loop_; }
  // Returns the function that |loop_| belong to.
  Function* GetFunction() const { return &function_; }

 private:
  IRContext* context_;
  LoopDescriptor* loop_desc_;
  Loop* loop_;
  Function& function_;

  // Populates the loop nest of |new_loop| according to |loop_| nest.
  void PopulateLoopNest(Loop* new_loop,
                        const LoopCloningResult& cloning_result) const;

  // Populates |new_loop| descriptor according to |old_loop|'s one.
  void PopulateLoopDesc(Loop* new_loop, Loop* old_loop,
                        const LoopCloningResult& cloning_result) const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_UTILS_H_
