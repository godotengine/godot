//
// Copyright (C) 2016 Google, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// The SPIR-V spec requires code blocks to appear in an order satisfying the
// dominator-tree direction (ie, dominator before the dominated).  This is,
// actually, easy to achieve: any pre-order CFG traversal algorithm will do it.
// Because such algorithms visit a block only after traversing some path to it
// from the root, they necessarily visit the block's idom first.
//
// But not every graph-traversal algorithm outputs blocks in an order that
// appears logical to human readers.  The problem is that unrelated branches may
// be interspersed with each other, and merge blocks may come before some of the
// branches being merged.
//
// A good, human-readable order of blocks may be achieved by performing
// depth-first search but delaying merge nodes until after all their branches
// have been visited.  This is implemented below by the inReadableOrder()
// function.

#include "spvIR.h"

#include <cassert>
#include <unordered_set>

using spv::Block;
using spv::Id;

namespace {
// Traverses CFG in a readable order, invoking a pre-set callback on each block.
// Use by calling visit() on the root block.
class ReadableOrderTraverser {
public:
    ReadableOrderTraverser(std::function<void(Block*, spv::ReachReason, Block*)> callback)
      : callback_(callback) {}
    // Visits the block if it hasn't been visited already and isn't currently
    // being delayed.  Invokes callback(block, why, header), then descends into its
    // successors.  Delays merge-block and continue-block processing until all
    // the branches have been completed.  If |block| is an unreachable merge block or
    // an unreachable continue target, then |header| is the corresponding header block.
    void visit(Block* block, spv::ReachReason why, Block* header)
    {
        assert(block);
        if (why == spv::ReachViaControlFlow) {
            reachableViaControlFlow_.insert(block);
        }
        if (visited_.count(block) || delayed_.count(block))
            return;
        callback_(block, why, header);
        visited_.insert(block);
        Block* mergeBlock = nullptr;
        Block* continueBlock = nullptr;
        auto mergeInst = block->getMergeInstruction();
        if (mergeInst) {
            Id mergeId = mergeInst->getIdOperand(0);
            mergeBlock = block->getParent().getParent().getInstruction(mergeId)->getBlock();
            delayed_.insert(mergeBlock);
            if (mergeInst->getOpCode() == spv::Op::OpLoopMerge) {
                Id continueId = mergeInst->getIdOperand(1);
                continueBlock =
                    block->getParent().getParent().getInstruction(continueId)->getBlock();
                delayed_.insert(continueBlock);
            }
        }
        if (why == spv::ReachViaControlFlow) {
            const auto& successors = block->getSuccessors();
            for (auto it = successors.cbegin(); it != successors.cend(); ++it)
                visit(*it, why, nullptr);
        }
        if (continueBlock) {
            const spv::ReachReason continueWhy =
                (reachableViaControlFlow_.count(continueBlock) > 0)
                    ? spv::ReachViaControlFlow
                    : spv::ReachDeadContinue;
            delayed_.erase(continueBlock);
            visit(continueBlock, continueWhy, block);
        }
        if (mergeBlock) {
            const spv::ReachReason mergeWhy =
                (reachableViaControlFlow_.count(mergeBlock) > 0)
                    ? spv::ReachViaControlFlow
                    : spv::ReachDeadMerge;
            delayed_.erase(mergeBlock);
            visit(mergeBlock, mergeWhy, block);
        }
    }

private:
    std::function<void(Block*, spv::ReachReason, Block*)> callback_;
    // Whether a block has already been visited or is being delayed.
    std::unordered_set<Block *> visited_, delayed_;

    // The set of blocks that actually are reached via control flow.
    std::unordered_set<Block *> reachableViaControlFlow_;
};
}

void spv::inReadableOrder(Block* root, std::function<void(Block*, spv::ReachReason, Block*)> callback)
{
    ReadableOrderTraverser(callback).visit(root, spv::ReachViaControlFlow, nullptr);
}
