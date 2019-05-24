// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_VAL_CONSTRUCT_H_
#define SOURCE_VAL_CONSTRUCT_H_

#include <cstdint>
#include <set>
#include <vector>

#include "source/val/basic_block.h"

namespace spvtools {
namespace val {

/// Functor for ordering BasicBlocks. BasicBlock pointers must not be null.
struct less_than_id {
  bool operator()(const BasicBlock* lhs, const BasicBlock* rhs) const {
    return lhs->id() < rhs->id();
  }
};

enum class ConstructType : int {
  kNone = 0,
  /// The set of blocks dominated by a selection header, minus the set of blocks
  /// dominated by the header's merge block
  kSelection,
  /// The set of blocks dominated by an OpLoopMerge's Continue Target and post
  /// dominated by the corresponding back
  kContinue,
  ///  The set of blocks dominated by a loop header, minus the set of blocks
  ///  dominated by the loop's merge block, minus the loop's corresponding
  ///  continue construct
  kLoop,
  ///  The set of blocks dominated by an OpSwitch's Target or Default, minus the
  ///  set of blocks dominated by the OpSwitch's merge block (this construct is
  ///  only defined for those OpSwitch Target or Default that are not equal to
  ///  the OpSwitch's corresponding merge block)
  kCase
};

class Function;

/// @brief This class tracks the CFG constructs as defined in the SPIR-V spec
class Construct {
 public:
  Construct(ConstructType type, BasicBlock* dominator,
            BasicBlock* exit = nullptr,
            std::vector<Construct*> constructs = std::vector<Construct*>());

  /// Returns the type of the construct
  ConstructType type() const;

  const std::vector<Construct*>& corresponding_constructs() const;
  std::vector<Construct*>& corresponding_constructs();
  void set_corresponding_constructs(std::vector<Construct*> constructs);

  /// Returns the dominator block of the construct.
  ///
  /// This is usually the header block or the first block of the construct.
  const BasicBlock* entry_block() const;

  /// Returns the dominator block of the construct.
  ///
  /// This is usually the header block or the first block of the construct.
  BasicBlock* entry_block();

  /// Returns the exit block of the construct.
  ///
  /// For a continue construct it is  the backedge block of the corresponding
  /// loop construct. For the case  construct it is the block that branches to
  /// the OpSwitch merge block or  other case blocks. Otherwise it is the merge
  /// block of the corresponding  header block
  const BasicBlock* exit_block() const;

  /// Returns the exit block of the construct.
  ///
  /// For a continue construct it is  the backedge block of the corresponding
  /// loop construct. For the case  construct it is the block that branches to
  /// the OpSwitch merge block or  other case blocks. Otherwise it is the merge
  /// block of the corresponding  header block
  BasicBlock* exit_block();

  /// Sets the exit block for this construct. This is useful for continue
  /// constructs which do not know the back-edge block during construction
  void set_exit(BasicBlock* exit_block);

  // Returns whether the exit block of this construct is the merge block
  // for an OpLoopMerge or OpSelectionMerge
  bool ExitBlockIsMergeBlock() const {
    return type_ == ConstructType::kLoop || type_ == ConstructType::kSelection;
  }

  using ConstructBlockSet = std::set<BasicBlock*, less_than_id>;

  // Returns the basic blocks in this construct. This function should not
  // be called before the exit block is set and dominators have been
  // calculated.
  ConstructBlockSet blocks(Function* function) const;

 private:
  /// The type of the construct
  ConstructType type_;

  /// These are the constructs that are related to this construct. These
  /// constructs can be the continue construct, for the corresponding loop
  /// construct, the case construct that are part of the same OpSwitch
  /// instruction
  ///
  /// Here is a table that describes what constructs are included in
  /// @p corresponding_constructs_
  /// | this construct | corresponding construct          |
  /// |----------------|----------------------------------|
  /// | loop           | continue                         |
  /// | continue       | loop                             |
  /// | case           | other cases in the same OpSwitch |
  ///
  /// kContinue and kLoop constructs will always have corresponding
  /// constructs even if they are represented by the same block
  std::vector<Construct*> corresponding_constructs_;

  /// @brief Dominator block for the construct
  ///
  /// The dominator block for the construct. Depending on the construct this may
  /// be a selection header, a continue target of a loop, a loop header or a
  /// Target or Default block of a switch
  BasicBlock* entry_block_;

  /// @brief Exiting block for the construct
  ///
  /// The exit block for the construct. This can be a merge block for the loop
  /// and selection constructs, a back-edge block for a continue construct, or
  /// the branching block for the case construct
  BasicBlock* exit_block_;
};

}  // namespace val
}  // namespace spvtools

#endif  // SOURCE_VAL_CONSTRUCT_H_
